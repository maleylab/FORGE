from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any
import yaml

from labtools.workflow.state import WorkflowState
from labtools.workflow.policy import RestartPolicy
from labtools.orca.restart import RestartEngine
from labtools.workflow.provenance import write_stage_provenance
from labtools.workflow.utils import run_orca
import labtools.orca.classify as classify


def write_fail_sentinel(jobdir: Path, stage: str, attempts: int) -> None:
    (jobdir / "FAIL").write_text(
        f"Stage '{stage}' failed permanently after {attempts} attempts.\n"
    )


def write_failure_json(jobdir: Path, metadata: Dict[str, Any]) -> None:
    p = jobdir / ".forge" / "failure.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(metadata, indent=2))


def _write_final_xyz(stage_dir: Path, rec: dict) -> None:
    geom = rec.get("final_geometry")
    if not geom:
        # fallback: ORCA often writes `<stage>.xyz`
        alt = stage_dir / f"{stage_dir.name}.xyz"
        if alt.exists():
            raw = alt.read_text().splitlines()
            xyz = [ln for ln in raw if ln.strip()]
            # skip header if present
            if xyz and xyz[0].strip().isdigit():
                xyz = xyz[2:]
            geom = xyz
        else:
            return

    out = stage_dir / "final.xyz"
    with out.open("w") as f:
        f.write(f"{len(geom)}\n")
        f.write("FORGE final geometry\n")
        for line in geom:
            f.write(f"{line}\n")


def workflow_step(jobdir: Path) -> None:
    jobdir = Path(jobdir)
    job_yaml = jobdir / "job.yaml"

    if not job_yaml.exists():
        raise FileNotFoundError(f"Missing job.yaml in {jobdir}")

    job_config = yaml.safe_load(job_yaml.read_text())

    from labtools.workflow.engine import WorkflowEngine
    engine = WorkflowEngine(job_config, jobdir)

    restart_policy = RestartPolicy.from_config(job_config.get("restart", {}))
    restart_engine = RestartEngine(restart_policy)

    try:
        state = WorkflowState.load(jobdir)
    except FileNotFoundError:
        state = engine.initialize_state()

    if state.is_finished:
        return

    stages = engine.resolve_stages()
    stage = stages[state.stage_index]
    stage_name = stage["name"]

    stage_dir = jobdir / stage_name
    stage_out = stage_dir / "output.out"

    if not stage_dir.exists():
        stage_dir.mkdir()
        state.mark_stage_start()
        state.save(jobdir)
        write_stage_provenance(jobdir, f"{stage_name}_start")
        inp = engine.render_stage_input(stage, stage_dir)
        run_orca(inp, nprocs=job_config.get("cpus", 1))
        return

    if not stage_out.exists():
        return

    import labtools.orca.collect as collect
    try:
        parsed = collect.parse_orca_out_file(stage_out)
        if isinstance(parsed, dict) and "parsed" in parsed:
            parsed = parsed["parsed"]

        rec = {
            "termination": "success",
            "errors": parsed.get("errors", []),
            "final_geometry": parsed.get("final_geometry"),
        }

        if rec["errors"]:
            rec["termination"] = "failure"

    except Exception:
        rec = {
            "termination": "failure",
            "errors": ["parse_error"],
            "final_geometry": None,
        }

    success = rec["termination"] == "success"

    if success:
        _write_final_xyz(stage_dir, rec)
        state.mark_stage_success(final_geometry=rec.get("final_geometry"))
        write_stage_provenance(jobdir, f"{stage_name}_success")
        state.advance_stage()
        state.save(jobdir)
        return

    fail_type = classify.classify_failure(rec, stage_out)
    write_stage_provenance(jobdir, f"{stage_name}_fail_{fail_type}")

    allowed = restart_policy.allow_restart(stage_name)
    max_restarts = restart_policy.max_restarts_for(stage_name)

    if fail_type == "BAD_INPUT":
        allowed = False

    if not allowed:
        metadata = {
            "job_id": state.job_id,
            "stage": stage_name,
            "attempts": state.attempt,
            "fail_type": fail_type,
            "reason": "restart_not_allowed",
        }
        write_fail_sentinel(jobdir, stage_name, state.attempt)
        write_failure_json(jobdir, metadata)
        state.mark_fail(fail_type, "restart_not_allowed")
        state.append_history(jobdir, metadata)
        state.save(jobdir)
        return

    if state.attempt >= max_restarts:
        metadata = {
            "job_id": state.job_id,
            "stage": stage_name,
            "attempts": state.attempt,
            "fail_type": fail_type,
            "reason": "max_restarts_exceeded",
        }
        write_fail_sentinel(jobdir, stage_name, state.attempt)
        write_failure_json(jobdir, metadata)
        state.mark_fail(fail_type, "max_restarts_exceeded")
        state.append_history(jobdir, metadata)
        state.save(jobdir)
        return

    state.increment_attempt()
    state.save(jobdir)

    overrides = restart_engine.generate_restart(
        stage_dir=stage_dir,
        rec={"attempt": state.attempt, **rec},
        fail_type=fail_type,
    )

    inp = engine.render_stage_input(stage, stage_dir, overrides=overrides)
    run_orca(inp, nprocs=job_config.get("cpus", 1))
    write_stage_provenance(jobdir, f"{stage_name}_restart_{state.attempt}")
