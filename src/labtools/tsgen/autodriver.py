"""Autonomous pipeline-native TSGen2 driver.

This module is the replacement for the legacy monolithic TSGenController path.
It does not own chemistry or execution logic directly. Instead, it advances the
artifact-backed TSGen stages:

    L0 render -> submit -> wait -> collect -> promote
    L1 render -> submit -> wait -> collect -> promote
    L2 render -> submit -> wait -> collect -> promote
    verify

The driver is intended to be run either interactively for small tests or inside a
lightweight SLURM driver job for production TSGen campaigns. The ORCA jobs are
still submitted as normal worker jobs; this process only coordinates stage
frontiers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from labtools.pipeline.artifacts import ArtifactRef
from labtools.pipeline.builtin import OrcaCollectStage, OrcaSubmitStage
from labtools.pipeline.run import PipelineRun
from labtools.pipeline.slurm_status import gather_submit_status
from labtools.tsgen.pipeline_stages import (
    TSGenL0PromoteStage,
    TSGenL0RenderStage,
    TSGenL1PromoteStage,
    TSGenL1RenderStage,
    TSGenL2PromoteStage,
    TSGenL2RenderStage,
    TSGenVerifyStage,
)


TERMINAL_SCHEDULER_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "DRY_RUN",
    "VALIDATED",
    "UNSUBMITTED",
    "UNKNOWN",
}
ACTIVE_SCHEDULER_STATES = {"PENDING", "RUNNING"}


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _artifact_ref_from_manifest(kind: str, manifest: Path) -> ArtifactRef:
    obj = _load_manifest(manifest)
    artifact_id = str(((obj.get("artifact") or {}).get("id") or manifest.parent.name))
    return ArtifactRef(artifact_id=artifact_id, kind=kind, relpath=f"{kind}/{artifact_id}")


def _iter_artifact_manifests(run_dir: Path, kind: str) -> Iterable[Path]:
    kind_dir = run_dir / "artifacts" / kind
    if not kind_dir.is_dir():
        return []
    return sorted(kind_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_artifact_ref(run_dir: Path, kind: str, **matches: str) -> Optional[ArtifactRef]:
    for man in _iter_artifact_manifests(run_dir, kind):
        obj = _load_manifest(man)
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        ok = True
        for key, expected in matches.items():
            if str(data.get(key) or "") != str(expected):
                ok = False
                break
        if ok:
            return _artifact_ref_from_manifest(kind, man)
    return None


def _latest_artifact_data(run_dir: Path, kind: str, **matches: str) -> Optional[Dict[str, Any]]:
    for man in _iter_artifact_manifests(run_dir, kind):
        obj = _load_manifest(man)
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        ok = True
        for key, expected in matches.items():
            if str(data.get(key) or "") != str(expected):
                ok = False
                break
        if ok:
            data = dict(data)
            data["_artifact_id"] = str(((obj.get("artifact") or {}).get("id") or man.parent.name))
            data["_manifest"] = str(man)
            return data
    return None


def _latest_submit_for_render(run_dir: Path, render_ref: ArtifactRef) -> Optional[Dict[str, Any]]:
    for man in _iter_artifact_manifests(run_dir, "SubmitBatch"):
        obj = _load_manifest(man)
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        rb = data.get("render_batch_ref") if isinstance(data.get("render_batch_ref"), dict) else {}
        if str(rb.get("artifact_id") or "") == str(render_ref.artifact_id):
            data = dict(data)
            data["_artifact_id"] = str(((obj.get("artifact") or {}).get("id") or man.parent.name))
            data["_manifest"] = str(man)
            return data
    return None


def _latest_collect_for_submit(run_dir: Path, submit_artifact_id: str) -> Optional[Dict[str, Any]]:
    for man in _iter_artifact_manifests(run_dir, "CollectBatch"):
        obj = _load_manifest(man)
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        sb = data.get("submit_batch_ref") if isinstance(data.get("submit_batch_ref"), dict) else {}
        if str(sb.get("artifact_id") or "") == str(submit_artifact_id):
            data = dict(data)
            data["_artifact_id"] = str(((obj.get("artifact") or {}).get("id") or man.parent.name))
            data["_manifest"] = str(man)
            return data
    return None


def _candidate_batch_exists(run_dir: Path, stage: str) -> bool:
    return _latest_artifact_ref(run_dir, "TSGenCandidateBatch", tsgen_stage=stage) is not None


def _verified_batch_exists(run_dir: Path) -> bool:
    return _latest_artifact_ref(run_dir, "TSGenVerifiedBatch") is not None


def _ensure_run_dir(run_dir: Path, *, config: Dict[str, Any], pipeline_name: str) -> None:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        pr = PipelineRun(run_dir=run_dir, pipeline_name=pipeline_name, config=config, stages=[])
        pr.init(allow_exists=False)
        return

    if not run_dir.is_dir():
        raise RuntimeError(f"Run path exists but is not a directory: {run_dir}")

    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "workspace").mkdir(exist_ok=True)

    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        pr = PipelineRun(run_dir=run_dir, pipeline_name=pipeline_name, config=config, stages=[])
        pr.init(allow_exists=True)


def _run_stages(run_dir: Path, *, config: Dict[str, Any], name: str, stages: List[Any]) -> Dict[str, Any]:
    pr = PipelineRun(
        run_dir=run_dir.expanduser().resolve(),
        pipeline_name=name,
        config=config,
        stages=stages,
    )
    result = pr.run(resume=False)
    if not result.get("ok"):
        failed = result.get("failed_stage") or "unknown"
        msg = result.get("message") or ""
        raise RuntimeError(f"{name} failed at {failed}: {msg}")
    return result


def _wait_for_latest_submit(
    run_dir: Path,
    *,
    poll_seconds: int,
    max_wait_seconds: Optional[int],
    dry_run: bool,
    validate_only: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if dry_run or validate_only:
        return {"skipped": True, "reason": "dry_run_or_validate_only"}

    t0 = time.time()
    last_status: Dict[str, Any] = {}
    while True:
        status = gather_submit_status(run_dir)
        last_status = status
        counts = dict(status.get("counts") or {})
        active = sum(int(counts.get(s, 0) or 0) for s in ACTIVE_SCHEDULER_STATES)
        per_job = list(status.get("per_job") or [])

        if verbose:
            count_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no jobs"
            print(f"[{_utc_ts()}] wait: {count_str}", flush=True)

        if per_job and active == 0:
            return status

        if max_wait_seconds is not None and (time.time() - t0) > max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for submitted jobs after {max_wait_seconds} s")

        time.sleep(max(5, int(poll_seconds)))


def _render_submit_wait_collect(
    *,
    run_dir: Path,
    config: Dict[str, Any],
    stage: str,
    render_stage_obj: Any,
    poll_seconds: int,
    max_wait_seconds: Optional[int],
    verbose: bool,
) -> None:
    render_ref = _latest_artifact_ref(run_dir, "RenderBatch", tsgen_stage=stage)
    if render_ref is None:
        if verbose:
            print(f"[{_utc_ts()}] {stage}: render", flush=True)
        _run_stages(run_dir, config=config, name=f"tsgen_{stage.lower()}_render", stages=[render_stage_obj])
        render_ref = _latest_artifact_ref(run_dir, "RenderBatch", tsgen_stage=stage)
        if render_ref is None:
            raise RuntimeError(f"{stage}: render completed but no matching RenderBatch was found")
    else:
        if verbose:
            print(f"[{_utc_ts()}] {stage}: existing RenderBatch {render_ref.artifact_id}", flush=True)

    submit = _latest_submit_for_render(run_dir, render_ref)
    if submit is None:
        if verbose:
            print(f"[{_utc_ts()}] {stage}: submit", flush=True)
        _run_stages(run_dir, config=config, name=f"tsgen_{stage.lower()}_submit", stages=[OrcaSubmitStage()])
        submit = _latest_submit_for_render(run_dir, render_ref)
        if submit is None:
            raise RuntimeError(f"{stage}: submit completed but no matching SubmitBatch was found")
    else:
        if verbose:
            print(f"[{_utc_ts()}] {stage}: existing SubmitBatch {submit.get('_artifact_id')}", flush=True)

    _wait_for_latest_submit(
        run_dir,
        poll_seconds=poll_seconds,
        max_wait_seconds=max_wait_seconds,
        dry_run=bool(config.get("dry_run", False)),
        validate_only=bool(config.get("validate_only", False)),
        verbose=verbose,
    )

    submit_id = str(submit.get("_artifact_id") or "")
    collect = _latest_collect_for_submit(run_dir, submit_id)
    if collect is None:
        if verbose:
            print(f"[{_utc_ts()}] {stage}: collect", flush=True)
        _run_stages(run_dir, config=config, name=f"tsgen_{stage.lower()}_collect", stages=[OrcaCollectStage()])
    else:
        if verbose:
            print(f"[{_utc_ts()}] {stage}: existing CollectBatch {collect.get('_artifact_id')}", flush=True)


def _promote_if_needed(
    *,
    run_dir: Path,
    config: Dict[str, Any],
    stage: str,
    promote_stage_obj: Any,
    verbose: bool,
) -> None:
    if _candidate_batch_exists(run_dir, stage):
        if verbose:
            print(f"[{_utc_ts()}] {stage}: existing TSGenCandidateBatch", flush=True)
        return
    if verbose:
        print(f"[{_utc_ts()}] {stage}: promote", flush=True)
    _run_stages(run_dir, config=config, name=f"tsgen_{stage.lower()}_promote", stages=[promote_stage_obj])


def run_tsgen_autonomous(
    *,
    plan: Path,
    run_dir: Path,
    profile: str = "medium",
    sbatch_template: str = "single_orca_job.sbatch.j2",
    execution_backend: str = "slurm",
    n_drones: int = 1,
    poll_seconds: int = 60,
    max_wait_minutes: Optional[int] = None,
    dry_run: bool = False,
    validate_only: bool = False,
    l0_max_promote: int = 3,
    l1_max_promote: int = 0,
    l2_max_promote: int = 0,
    require_one_imag_l2: bool = True,
    verify_require_one_imag: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run TSGen2 end-to-end using pipeline-native stages.

    This function is intentionally restartable at coarse stage boundaries. If a
    run directory already contains completed stage artifacts, those stages are
    reused rather than regenerated.
    """
    plan = plan.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if not plan.is_file():
        raise FileNotFoundError(f"TSGen plan not found: {plan}")

    max_wait_seconds = None if max_wait_minutes is None else int(max_wait_minutes) * 60
    config: Dict[str, Any] = {
        "plan": str(plan),
        "plan_path": str(plan),
        "profile": profile,
        "sbatch_template": sbatch_template,
        "execution_backend": execution_backend,
        "backend": execution_backend,
        "n_drones": int(n_drones),
        "dry_run": bool(dry_run),
        "validate_only": bool(validate_only),
        "l0_max_promote": int(l0_max_promote),
        "l1_max_promote": int(l1_max_promote),
        "l2_max_promote": int(l2_max_promote),
        "require_one_imag": bool(require_one_imag_l2),
        "verify_require_one_imag": bool(verify_require_one_imag),
        "autonomous_driver": True,
    }

    _ensure_run_dir(run_dir, config=config, pipeline_name="tsgen2_autonomous")

    phases: List[Tuple[str, Any, Any]] = [
        ("L0", TSGenL0RenderStage(), TSGenL0PromoteStage()),
        ("L1", TSGenL1RenderStage(), TSGenL1PromoteStage()),
        ("L2", TSGenL2RenderStage(), TSGenL2PromoteStage()),
    ]

    for stage, render_stage, promote_stage in phases:
        if not _candidate_batch_exists(run_dir, stage):
            _render_submit_wait_collect(
                run_dir=run_dir,
                config=config,
                stage=stage,
                render_stage_obj=render_stage,
                poll_seconds=poll_seconds,
                max_wait_seconds=max_wait_seconds,
                verbose=verbose,
            )
            _promote_if_needed(
                run_dir=run_dir,
                config=config,
                stage=stage,
                promote_stage_obj=promote_stage,
                verbose=verbose,
            )
        elif verbose:
            print(f"[{_utc_ts()}] {stage}: complete", flush=True)

    if not _verified_batch_exists(run_dir):
        if verbose:
            print(f"[{_utc_ts()}] VERIFY: exploratory verification", flush=True)
        # Avoid leaking the L2 promotion option into verify if a future verify
        # option diverges. TSGenVerifyStage currently reads require_one_imag.
        verify_config = dict(config)
        verify_config["require_one_imag"] = bool(verify_require_one_imag)
        _run_stages(run_dir, config=verify_config, name="tsgen_verify", stages=[TSGenVerifyStage()])
    elif verbose:
        print(f"[{_utc_ts()}] VERIFY: existing TSGenVerifiedBatch", flush=True)

    return {
        "ok": True,
        "run_dir": str(run_dir),
        "plan": str(plan),
        "completed": ["L0", "L1", "L2", "VERIFY"],
    }
