"""
Pipeline-native TSGen2 stages.

This module wraps the TSGen2 stage functions in FORGE's artifact-backed
pipeline interface. TSGen2 is represented as explicit render, collect, promote,
and verify stages rather than a monolithic controller.

The promotion stages emit generic TSGenCandidateBatch artifacts. Later TSGen
stages consume TSGenCandidateBatch artifacts rather than in-memory workflow
state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from labtools.pipeline.artifacts import Artifact, ArtifactRef, sha256_file
from labtools.pipeline.stages import BaseStage
from labtools.tsgen.plan import TSGenPlan
from labtools.tsgen.tsgen_orca import parse_frequencies_and_modes


TSGEN_CANDIDATE_KIND = "TSGenCandidateBatch"


def _cfg(context: Dict[str, Any], key: str, default: Any = None, *, required: bool = False) -> Any:
    """Read a stage option from direct context first, then PipelineRun.config."""
    if key in context:
        return context[key]
    run = context.get("run")
    if run is not None:
        cfg = getattr(run, "config", None)
        if isinstance(cfg, dict) and key in cfg:
            return cfg[key]
    if required:
        raise KeyError(key)
    return default


def _run_dir(context: Dict[str, Any]) -> Path:
    return Path(str(context["run_dir"])).expanduser().resolve()


def _workspace_dir(context: Dict[str, Any]) -> Path:
    return Path(str(context["workspace_dir"])).expanduser().resolve()


def _artifact_file_path(run_dir: Path, ref: ArtifactRef, relpath: str) -> Path:
    return (run_dir / "artifacts" / ref.relpath / relpath).resolve()


def _find_latest_artifact_ref(run_dir: Path, kind: str) -> Optional[ArtifactRef]:
    kind_dir = run_dir / "artifacts" / kind
    if not kind_dir.is_dir():
        return None
    manifests = sorted(kind_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        return None
    man_path = manifests[0]
    artifact_id = man_path.parent.name
    relpath = f"{kind}/{artifact_id}"
    return ArtifactRef(artifact_id=artifact_id, kind=kind, relpath=relpath)



def _load_json_artifact_payload(run_dir: Path, ref: ArtifactRef, relpath_key: str, *, default_relpath: str) -> Dict[str, Any]:
    """Load a JSON payload copied into an artifact, falling back to artifact.data."""
    art = Artifact.load(run_dir, ref)
    relpath = str(art.data.get(relpath_key) or default_relpath)
    path = _artifact_file_path(run_dir, ref, relpath)
    if path.is_file():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    return dict(art.data or {})


def _latest_candidate_ref(run_dir: Path, *, stage: Optional[str] = None) -> Optional[ArtifactRef]:
    """Find the newest TSGenCandidateBatch, optionally constrained by tsgen_stage."""
    kind_dir = run_dir / "artifacts" / TSGEN_CANDIDATE_KIND
    if not kind_dir.is_dir():
        return None
    manifests = sorted(kind_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for man_path in manifests:
        artifact_id = man_path.parent.name
        ref = ArtifactRef(artifact_id=artifact_id, kind=TSGEN_CANDIDATE_KIND, relpath=f"{TSGEN_CANDIDATE_KIND}/{artifact_id}")
        if stage is None:
            return ref
        try:
            art = Artifact.load(run_dir, ref)
        except Exception:
            continue
        if str((art.data or {}).get("tsgen_stage") or "") == stage:
            return ref
    return None


def _candidate_ref_for_stage(
    run_dir: Path,
    candidate_ref: Optional[ArtifactRef],
    *,
    stage: str,
) -> Optional[ArtifactRef]:
    """Return candidate_ref only if it matches stage; otherwise find latest matching stage.

    Pipeline commands can leave a generic tsgen_candidate_batch_ref in context.
    In a completed run that generic reference can point at the newest batch
    (for example L2) while L1 rendering explicitly needs the latest L0 batch.
    """
    if candidate_ref is not None:
        try:
            art = Artifact.load(run_dir, candidate_ref)
            if str((art.data or {}).get("tsgen_stage") or "") == stage:
                return candidate_ref
        except Exception:
            pass
    return _latest_candidate_ref(run_dir, stage=stage)
def _artifact_ref_dict(ref: ArtifactRef) -> Dict[str, str]:
    return {"artifact_id": ref.artifact_id, "kind": ref.kind, "relpath": ref.relpath}


def _load_tsgen_plan(context: Dict[str, Any], *, force_work_dir: Optional[Path] = None) -> TSGenPlan:
    plan_value = _cfg(context, "plan", None)
    if plan_value is None:
        plan_value = _cfg(context, "plan_path", required=True)
    plan_path = Path(str(plan_value)).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"TSGen plan not found: {plan_path}")

    import yaml

    data = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"TSGen plan must parse to a mapping: {plan_path}")

    if force_work_dir is not None:
        data = dict(data)
        data["work_dir"] = str(force_work_dir)

    return TSGenPlan(**data)


def _read_collect_rows(run_dir: Path, collect_ref: ArtifactRef) -> List[Dict[str, Any]]:
    collect_art = Artifact.load(run_dir, collect_ref)
    records_rel = str(collect_art.data.get("records_jsonl") or "")
    if not records_rel:
        return []
    records_path = _artifact_file_path(run_dir, collect_ref, records_rel)
    if not records_path.is_file():
        return []

    rows: List[Dict[str, Any]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _collected_job_dirs(run_dir: Path, collect_ref: Optional[ArtifactRef]) -> Optional[Set[str]]:
    """Return collected job-dir names, or None when no CollectBatch is available."""
    if collect_ref is None:
        return None
    rows = _read_collect_rows(run_dir, collect_ref)
    return {
        str(row.get("job_dir") or "")
        for row in rows
        if str(row.get("collection_state") or "") == "COLLECTED"
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")




def _resolve_stage_jobs_outdir(context: Dict[str, Any], stage: str, *config_keys: str) -> Path:
    """Resolve a TSGen stage jobs directory from explicit config or workspace/<stage>."""
    outdir_value = None
    for key in config_keys:
        outdir_value = _cfg(context, key, None)
        if outdir_value is not None and str(outdir_value).strip() != "":
            break
    if outdir_value is None or str(outdir_value).strip() == "":
        jobs_outdir = _workspace_dir(context) / stage
    else:
        jobs_outdir = Path(str(outdir_value)).expanduser().resolve()
    jobs_outdir.mkdir(parents=True, exist_ok=True)
    return jobs_outdir


def _load_promoted_candidate_inputs(
    run_dir: Path,
    candidate_ref: ArtifactRef,
    *,
    source_stage: str,
    target_stage: str,
) -> tuple[List[Dict[str, Any]], List[Path], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load promoted candidate XYZ paths for the next TSGen render stage."""
    candidate_payload = _load_json_artifact_payload(
        run_dir,
        candidate_ref,
        "candidate_batch_json",
        default_relpath="candidate_batch.json",
    )
    promoted: List[Dict[str, Any]] = list(candidate_payload.get("promoted") or [])
    if not promoted:
        raise RuntimeError(f"{source_stage} TSGenCandidateBatch contains no promoted candidates for {target_stage} rendering")

    upstream_xyz: List[Path] = []
    accepted_inputs: List[Dict[str, Any]] = []
    rejected_inputs: List[Dict[str, Any]] = []
    for idx, cand in enumerate(promoted):
        xyz_value = cand.get("xyz") if isinstance(cand, dict) else None
        if not xyz_value:
            rejected_inputs.append({"candidate_index": idx, "reason": "missing_xyz_field", "candidate": cand})
            continue
        xyz = Path(str(xyz_value)).expanduser().resolve()
        if not xyz.is_file():
            rejected_inputs.append({"candidate_index": idx, "reason": "xyz_not_found", "xyz": str(xyz), "candidate": cand})
            continue
        upstream_xyz.append(xyz)
        accepted_inputs.append({"candidate_index": idx, "xyz": str(xyz), "candidate": cand})

    if not upstream_xyz:
        raise RuntimeError(f"No promoted {source_stage} candidate XYZ files are available for {target_stage} rendering")

    return promoted, upstream_xyz, accepted_inputs, rejected_inputs


def _job_io_paths(job_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Return the first .out and .xyz files in a job directory, if present."""
    out_files = sorted(job_dir.glob("*.out"))
    xyz_files = sorted(job_dir.glob("*.xyz"))
    out_path = out_files[0] if out_files else None
    xyz_path = xyz_files[0] if xyz_files else None
    return out_path, xyz_path


def _frequency_fields(out_path: Path) -> Dict[str, Any]:
    """Parse imaginary-frequency metadata used by TSGen candidate ranking."""
    freqs, _ = parse_frequencies_and_modes(out_path)
    imags = [float(f) for f in freqs if float(f) < 0.0]
    imag_cm1 = min(imags) if imags else None
    return {
        "n_imag": len(imags),
        "imag_cm1": imag_cm1,
        "score": abs(float(imag_cm1)) if imag_cm1 is not None else None,
    }


def _scan_stage_jobs_for_candidates(
    *,
    jobs_outdir: Path,
    job_dirs: Sequence[str],
    collected_job_dirs: Set[str],
    source_stage: str,
    debug: bool,
    require_one_imag: Optional[bool],
    parse_error_debug_reason: str = "debug_accept_freq_parse_error",
    accept_reason: str = "exploratory_accept",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scan collected stage jobs and return candidate/rejected rows.

    require_one_imag=None means do not reject candidates by imaginary-count.
    require_one_imag=True rejects non-TS-like candidates unless debug=True.
    """
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for jd in job_dirs:
        if jd not in collected_job_dirs:
            rejected.append({"job_dir": jd, "reason": "not_collected"})
            continue

        job_dir = jobs_outdir / jd
        out_path, xyz_path = _job_io_paths(job_dir)

        if xyz_path is None:
            rejected.append({"job_dir": jd, "reason": "missing_xyz"})
            continue
        if out_path is None:
            rejected.append({"job_dir": jd, "reason": "missing_out"})
            continue

        row: Dict[str, Any] = {
            "source_stage": source_stage,
            "job_dir": jd,
            "job_path": str(job_dir),
            "xyz": str(xyz_path),
            "out": str(out_path),
            "status": "candidate",
            "promoted": False,
        }

        try:
            row.update(_frequency_fields(out_path))
        except Exception as e:
            if debug:
                row["freq_parse_error"] = str(e)
                row["n_imag"] = None
                row["imag_cm1"] = None
                row["score"] = None
                row["reason"] = parse_error_debug_reason
                candidates.append(row)
                continue
            rejected.append({"job_dir": jd, "reason": "freq_parse_error", "message": str(e)})
            continue

        if debug:
            row["reason"] = "debug_accept"
            candidates.append(row)
            continue

        if require_one_imag is True and row.get("n_imag") != 1:
            row["reason"] = "bad_imag_count"
            rejected.append(row)
            continue

        row["reason"] = accept_reason
        candidates.append(row)

    return candidates, rejected


def _promote_ranked_candidates(candidates: Sequence[Dict[str, Any]], max_promote: int) -> List[Dict[str, Any]]:
    """Rank candidates by score and return the promoted subset; max_promote<=0 promotes all."""
    ranked = sorted(candidates, key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return ranked[:max_promote] if max_promote > 0 else list(ranked)


def _mark_promoted(candidates: Sequence[Dict[str, Any]], promoted: Sequence[Dict[str, Any]]) -> None:
    """Mark candidate rows in-place as promoted when their job_dir is in the promoted set."""
    promoted_keys = {str(row.get("job_dir") or "") for row in promoted}
    for row in candidates:
        row["promoted"] = str(row.get("job_dir") or "") in promoted_keys
        if row["promoted"]:
            row["status"] = "promoted"


def _write_candidate_batch(
    *,
    context: Dict[str, Any],
    run_dir: Path,
    stage: str,
    next_stage: str,
    policy: str,
    max_promote: int,
    candidates: List[Dict[str, Any]],
    promoted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    parents: Sequence[ArtifactRef],
) -> ArtifactRef:
    """Write a standard TSGenCandidateBatch/v1 artifact."""
    candidate_json = _workspace_dir(context) / f"tsgen_{stage.lower()}_candidate_batch.json"
    payload = {
        "artifact_schema": "TSGenCandidateBatch/v1",
        "tsgen_stage": stage,
        "source_stage": stage,
        "next_stage": next_stage,
        "policy": policy,
        "max_promote": max_promote,
        "n_candidates": len(candidates),
        "n_promoted": len(promoted),
        "n_rejected": len(rejected),
        "candidates": candidates,
        "promoted": promoted,
        "rejected": rejected,
    }
    candidate_json.parent.mkdir(parents=True, exist_ok=True)
    candidate_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    art = Artifact(
        kind=TSGEN_CANDIDATE_KIND,
        data={
            **payload,
            "candidate_batch_json": "candidate_batch.json",
        },
        parents=list(parents),
    )
    ref = art.write(run_dir=run_dir, copy_files=[(candidate_json, "candidate_batch.json")])
    context["tsgen_candidate_batch_ref"] = ref
    context[f"tsgen_{stage.lower()}_candidate_batch_ref"] = ref
    return ref

class TSGenL0RenderStage(BaseStage):
    """Render TSGen L0 seed jobs and publish a standard RenderBatch artifact."""

    def __init__(self):
        super().__init__(name="tsgen_l0_render")

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        plan_value = _cfg(context, "plan", None)
        if plan_value is None:
            plan_value = _cfg(context, "plan_path", required=True)
        plan_path = Path(str(plan_value)).expanduser().resolve()
        outdir_value = _cfg(context, "outdir", None)
        return {
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path) if plan_path.is_file() else "",
            "stage": "L0",
            "outdir": str(outdir_value) if outdir_value else "",
        }

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        from labtools.tsgen.stages.l0_seed import run_l0

        run_dir = _run_dir(context)

        outdir_value = _cfg(context, "outdir", None)
        if outdir_value is None or str(outdir_value).strip() == "":
            jobs_outdir = _workspace_dir(context) / "L0"
        else:
            jobs_outdir = Path(str(outdir_value)).expanduser().resolve()

        jobs_outdir.mkdir(parents=True, exist_ok=True)

        # run_l0() creates plan.stage_dir("L0") under plan.work_dir, so the
        # forced work_dir must be the parent of the desired L0 jobs directory.
        plan = _load_tsgen_plan(context, force_work_dir=jobs_outdir.parent)
        jobs = run_l0(plan)

        job_dirs = sorted([Path(str(j["jobdir"])).name for j in jobs])
        rows: List[Dict[str, Any]] = []
        for j in jobs:
            row = dict(j)
            for key in ("jobdir", "input"):
                if key in row:
                    row[key] = str(row[key])
            rows.append(row)

        jobs_jsonl = _workspace_dir(context) / "tsgen_l0_jobs.jsonl"
        _write_jsonl(jobs_jsonl, rows)

        plan_value = _cfg(context, "plan", _cfg(context, "plan_path", ""))
        art = Artifact(
            kind="RenderBatch",
            data={
                "domain": "tsgen2",
                "tsgen_stage": "L0",
                "plan": str(Path(str(plan_value)).expanduser().resolve()),
                "jobs_outdir": str(jobs_outdir),
                "job_dirs": job_dirs,
                "n_jobs": len(job_dirs),
                "jobs_jsonl": "jobs.jsonl",
            },
        )
        ref = art.write(run_dir=run_dir, copy_files=[(jobs_jsonl, "jobs.jsonl")])
        context["render_batch_ref"] = ref
        context["tsgen_l0_render_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": "tsgen_l0_render_complete",
                    "stage": self.name,
                    "n_jobs": len(job_dirs),
                    "jobs_outdir": str(jobs_outdir),
                }
            )

        return [ref]


class TSGenL0PromoteStage(BaseStage):
    """Promote L0 seeds and emit a generic TSGenCandidateBatch artifact.

    The default policy mirrors the legacy controller's L0 policy: rank completed
    L0 jobs by largest absolute imaginary frequency magnitude.
    """

    def __init__(self):
        super().__init__(name="tsgen_l0_promote")

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = _run_dir(context)
        collect_ref = context.get("collect_batch_ref") or _find_latest_artifact_ref(run_dir, "CollectBatch")
        render_ref = context.get("render_batch_ref") or _find_latest_artifact_ref(run_dir, "RenderBatch")
        return {
            "collect_batch_id": getattr(collect_ref, "artifact_id", ""),
            "render_batch_id": getattr(render_ref, "artifact_id", ""),
            "artifact_kind": TSGEN_CANDIDATE_KIND,
            "max_promote": int(_cfg(context, "max_promote", _cfg(context, "l0_max_promote", 3))),
            "debug": bool(_cfg(context, "debug", False)),
        }

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        run_dir = _run_dir(context)
        collect_ref = context.get("collect_batch_ref") or _find_latest_artifact_ref(run_dir, "CollectBatch")
        render_ref = context.get("render_batch_ref") or _find_latest_artifact_ref(run_dir, "RenderBatch")
        if render_ref is None:
            raise RuntimeError("No RenderBatch artifact available for TSGen L0 promotion")

        render_art = Artifact.load(run_dir, render_ref)
        jobs_outdir = Path(str(render_art.data["jobs_outdir"])).expanduser().resolve()
        job_dirs: List[str] = list(render_art.data.get("job_dirs") or [])
        collected_job_dirs = _collected_job_dirs(run_dir, collect_ref)

        max_promote = int(_cfg(context, "max_promote", _cfg(context, "l0_max_promote", 3)))
        debug = bool(_cfg(context, "debug", False))

        candidates: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for jd in job_dirs:
            if collected_job_dirs is not None and jd not in collected_job_dirs:
                rejected.append({"job_dir": jd, "reason": "not_collected"})
                continue

            job_dir = jobs_outdir / jd
            out_files = sorted(job_dir.glob("*.out"))
            xyz_files = sorted(job_dir.glob("*.xyz"))
            out_path = out_files[0] if out_files else None
            xyz_path = xyz_files[0] if xyz_files else None

            if xyz_path is None:
                rejected.append({"job_dir": jd, "reason": "missing_xyz"})
                continue

            if debug:
                candidates.append(
                    {
                        "source_stage": "L0",
                        "job_dir": jd,
                        "job_path": str(job_dir),
                        "xyz": str(xyz_path),
                        "out": str(out_path) if out_path else None,
                        "score": None,
                        "imag_cm1": None,
                        "n_imag": None,
                        "status": "candidate",
                        "reason": "debug_accept",
                    }
                )
                continue

            if out_path is None:
                rejected.append({"job_dir": jd, "reason": "missing_out"})
                continue

            try:
                freqs, _ = parse_frequencies_and_modes(out_path)
            except Exception as e:
                rejected.append({"job_dir": jd, "reason": "parse_error", "message": str(e)})
                continue

            imags = [float(f) for f in freqs if float(f) < 0.0]
            if not imags:
                rejected.append({"job_dir": jd, "reason": "no_imaginary_frequency"})
                continue

            imag = min(imags)
            score = abs(imag)
            candidates.append(
                {
                    "source_stage": "L0",
                    "job_dir": jd,
                    "job_path": str(job_dir),
                    "xyz": str(xyz_path),
                    "out": str(out_path),
                    "score": score,
                    "imag_cm1": imag,
                    "n_imag": len(imags),
                    "status": "candidate",
                    "reason": "ranked_by_largest_abs_imaginary_frequency",
                }
            )

        if debug:
            promoted = candidates[:max_promote]
        else:
            promoted = sorted(candidates, key=lambda row: float(row.get("score") or 0.0), reverse=True)[:max_promote]

        promoted_keys = {str(row.get("job_dir") or "") for row in promoted}
        for row in candidates:
            row["promoted"] = str(row.get("job_dir") or "") in promoted_keys
            if row["promoted"]:
                row["status"] = "promoted"

        candidate_json = _workspace_dir(context) / "tsgen_l0_candidate_batch.json"
        candidate_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema": "TSGenCandidateBatch/v1",
            "tsgen_stage": "L0",
            "source_stage": "L0",
            "next_stage": "L1",
            "policy": "largest_abs_imaginary_frequency",
            "max_promote": max_promote,
            "n_candidates": len(candidates),
            "n_promoted": len(promoted),
            "n_rejected": len(rejected),
            "candidates": candidates,
            "promoted": promoted,
            "rejected": rejected,
        }
        candidate_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        parents = [render_ref]
        if collect_ref is not None:
            parents.append(collect_ref)

        art = Artifact(
            kind=TSGEN_CANDIDATE_KIND,
            data={
                "artifact_schema": "TSGenCandidateBatch/v1",
                "tsgen_stage": "L0",
                "source_stage": "L0",
                "next_stage": "L1",
                "policy": "largest_abs_imaginary_frequency",
                "max_promote": max_promote,
                "n_candidates": len(candidates),
                "n_promoted": len(promoted),
                "n_rejected": len(rejected),
                "candidates": candidates,
                "promoted": promoted,
                "rejected": rejected,
                "candidate_batch_json": "candidate_batch.json",
                # Temporary compatibility for any scripts written against the
                # first prototype's promotion artifact payload.
                "promotion_json": "candidate_batch.json",
            },
            parents=parents,
        )
        ref = art.write(run_dir=run_dir, copy_files=[(candidate_json, "candidate_batch.json")])
        context["tsgen_candidate_batch_ref"] = ref
        context["tsgen_l0_candidate_batch_ref"] = ref
        context["tsgen_l0_promotion_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": "tsgen_l0_candidate_batch_complete",
                    "stage": self.name,
                    "n_candidates": len(candidates),
                    "n_promoted": len(promoted),
                    "candidate_batch_ref": _artifact_ref_dict(ref),
                }
            )

        return [ref]


class TSGenCandidateRenderStageBase(BaseStage):
    """Base class for rendering TSGen jobs from a promoted candidate batch."""

    stage: str = ""
    source_stage: str = ""
    outdir_key: str = ""
    runner_module: str = ""
    runner_name: str = ""

    def __init__(self, *, name: str):
        super().__init__(name=name)

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = _run_dir(context)
        candidate_ref = _candidate_ref_for_stage(
            run_dir,
            context.get("tsgen_candidate_batch_ref"),
            stage=self.source_stage,
        )
        plan_value = _cfg(context, "plan", None)
        if plan_value is None:
            plan_value = _cfg(context, "plan_path", required=True)
        plan_path = Path(str(plan_value)).expanduser().resolve()
        outdir_value = _cfg(context, "outdir", _cfg(context, self.outdir_key, None))
        return {
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path) if plan_path.is_file() else "",
            "candidate_batch_id": getattr(candidate_ref, "artifact_id", ""),
            "source_stage": self.source_stage,
            "stage": self.stage,
            "outdir": str(outdir_value) if outdir_value else "",
        }

    def _runner(self):
        import importlib

        module = importlib.import_module(self.runner_module)
        return getattr(module, self.runner_name)

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        run_dir = _run_dir(context)
        candidate_ref = _candidate_ref_for_stage(
            run_dir,
            context.get("tsgen_candidate_batch_ref"),
            stage=self.source_stage,
        )
        if candidate_ref is None:
            raise RuntimeError(
                f"No {self.source_stage} TSGenCandidateBatch artifact available for TSGen {self.stage} rendering"
            )

        promoted, upstream_xyz, accepted_inputs, rejected_inputs = _load_promoted_candidate_inputs(
            run_dir,
            candidate_ref,
            source_stage=self.source_stage,
            target_stage=self.stage,
        )

        jobs_outdir = _resolve_stage_jobs_outdir(context, self.stage, "outdir", self.outdir_key)

        # TSGen stage runners create plan.stage_dir(stage) under plan.work_dir,
        # so the forced work_dir must be the parent of the desired jobs directory.
        plan = _load_tsgen_plan(context, force_work_dir=jobs_outdir.parent)
        jobs = self._runner()(plan, upstream_xyz)

        job_dirs = sorted([Path(str(j["jobdir"])).name for j in jobs])
        rows: List[Dict[str, Any]] = []
        for j, src in zip(jobs, accepted_inputs):
            row = dict(j)
            for key in ("jobdir", "input"):
                if key in row:
                    row[key] = str(row[key])
            row["parent_candidate_index"] = src["candidate_index"]
            row["parent_xyz"] = src["xyz"]
            row["parent_candidate"] = src["candidate"]
            rows.append(row)

        stage_lower = self.stage.lower()
        jobs_jsonl = _workspace_dir(context) / f"tsgen_{stage_lower}_jobs.jsonl"
        _write_jsonl(jobs_jsonl, rows)

        lineage_json = _workspace_dir(context) / f"tsgen_{stage_lower}_lineage.json"
        lineage_payload = {
            "artifact_schema": f"TSGen{self.stage}RenderLineage/v1",
            "source_candidate_batch": _artifact_ref_dict(candidate_ref),
            f"n_promoted_{self.source_stage.lower()}": len(promoted),
            f"n_{stage_lower}_inputs": len(accepted_inputs),
            f"n_{stage_lower}_jobs": len(job_dirs),
            "accepted_inputs": accepted_inputs,
            "rejected_inputs": rejected_inputs,
        }
        lineage_json.write_text(json.dumps(lineage_payload, indent=2), encoding="utf-8")

        plan_value = _cfg(context, "plan", _cfg(context, "plan_path", ""))
        art = Artifact(
            kind="RenderBatch",
            data={
                "domain": "tsgen2",
                "tsgen_stage": self.stage,
                "source_candidate_batch": _artifact_ref_dict(candidate_ref),
                "plan": str(Path(str(plan_value)).expanduser().resolve()),
                "jobs_outdir": str(jobs_outdir),
                "job_dirs": job_dirs,
                "n_jobs": len(job_dirs),
                "jobs_jsonl": "jobs.jsonl",
                "lineage_json": "lineage.json",
                "rejected_inputs": rejected_inputs,
            },
            parents=[candidate_ref],
        )
        ref = art.write(run_dir=run_dir, copy_files=[(jobs_jsonl, "jobs.jsonl"), (lineage_json, "lineage.json")])
        context["render_batch_ref"] = ref
        context[f"tsgen_{stage_lower}_render_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": f"tsgen_{stage_lower}_render_complete",
                    "stage": self.name,
                    "n_jobs": len(job_dirs),
                    "jobs_outdir": str(jobs_outdir),
                    "source_candidate_batch": _artifact_ref_dict(candidate_ref),
                }
            )

        return [ref]


class TSGenL1RenderStage(TSGenCandidateRenderStageBase):
    """Render TSGen L1 OptTS jobs from promoted L0 candidates."""

    stage = "L1"
    source_stage = "L0"
    outdir_key = "l1_outdir"
    runner_module = "labtools.tsgen.stages.l1_opt"
    runner_name = "run_l1"

    def __init__(self):
        super().__init__(name="tsgen_l1_render")


class TSGenL2RenderStage(TSGenCandidateRenderStageBase):
    """Render TSGen L2 high-level OptTS/Freq jobs from promoted L1 candidates."""

    stage = "L2"
    source_stage = "L1"
    outdir_key = "l2_outdir"
    runner_module = "labtools.tsgen.stages.l2_opt"
    runner_name = "run_l2"

    def __init__(self):
        super().__init__(name="tsgen_l2_render")


def _latest_artifact_ref_with_data(run_dir: Path, kind: str, **matches: str) -> Optional[ArtifactRef]:
    """Find newest artifact of a kind whose artifact.data string fields match."""
    kind_dir = run_dir / "artifacts" / kind
    if not kind_dir.is_dir():
        return None
    manifests = sorted(kind_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for man_path in manifests:
        artifact_id = man_path.parent.name
        ref = ArtifactRef(artifact_id=artifact_id, kind=kind, relpath=f"{kind}/{artifact_id}")
        try:
            art = Artifact.load(run_dir, ref)
        except Exception:
            continue
        data = art.data or {}
        ok = True
        for key, expected in matches.items():
            if str(data.get(key) or "") != str(expected):
                ok = False
                break
        if ok:
            return ref
    return None


class TSGenPromoteStageBase(BaseStage):
    """Base class for promoting completed TSGen optimization jobs."""

    stage: str = ""
    next_stage: str = ""
    render_stage: str = ""
    default_max_promote_key: str = ""
    default_max_promote: int = 0
    require_one_imag_default: Optional[bool] = None
    policy_accept_all: str = "exploratory_accept_all"
    policy_require_one_imag: str = "one_imaginary_frequency_gate"
    accept_reason: str = "exploratory_accept"

    def __init__(self, *, name: str):
        super().__init__(name=name)

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = _run_dir(context)
        collect_ref = context.get("collect_batch_ref") or _find_latest_artifact_ref(run_dir, "CollectBatch")
        render_ref = context.get("render_batch_ref") or _latest_artifact_ref_with_data(
            run_dir, "RenderBatch", tsgen_stage=self.render_stage
        )
        fp: Dict[str, Any] = {
            "collect_batch_id": getattr(collect_ref, "artifact_id", ""),
            "render_batch_id": getattr(render_ref, "artifact_id", ""),
            "artifact_kind": TSGEN_CANDIDATE_KIND,
            "stage": self.stage,
            "max_promote": int(_cfg(context, "max_promote", _cfg(context, self.default_max_promote_key, self.default_max_promote))),
            "debug": bool(_cfg(context, "debug", False)),
        }
        if self.require_one_imag_default is not None:
            fp["require_one_imag"] = bool(_cfg(context, "require_one_imag", self.require_one_imag_default))
        return fp

    def _require_one_imag(self, context: Dict[str, Any]) -> Optional[bool]:
        if self.require_one_imag_default is None:
            return None
        return bool(_cfg(context, "require_one_imag", self.require_one_imag_default))

    def _policy(self, require_one_imag: Optional[bool]) -> str:
        if require_one_imag is True:
            return self.policy_require_one_imag
        return self.policy_accept_all

    def _accept_reason(self, require_one_imag: Optional[bool]) -> str:
        if require_one_imag is True:
            return "ts_like_frequency_pattern"
        return self.accept_reason

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        run_dir = _run_dir(context)
        collect_ref = context.get("collect_batch_ref") or _find_latest_artifact_ref(run_dir, "CollectBatch")
        render_ref = context.get("render_batch_ref") or _latest_artifact_ref_with_data(
            run_dir, "RenderBatch", tsgen_stage=self.render_stage
        )
        if render_ref is None:
            raise RuntimeError(f"No {self.render_stage} RenderBatch artifact available for TSGen {self.stage} promotion")
        if collect_ref is None:
            raise RuntimeError(f"No CollectBatch artifact available for TSGen {self.stage} promotion")

        debug = bool(_cfg(context, "debug", False))
        max_promote = int(_cfg(context, "max_promote", _cfg(context, self.default_max_promote_key, self.default_max_promote)))
        require_one_imag = self._require_one_imag(context)

        render_art = Artifact.load(run_dir, render_ref)
        collect_art = Artifact.load(run_dir, collect_ref)
        jobs_outdir = Path(str(render_art.data.get("jobs_outdir") or collect_art.data.get("jobs_outdir"))).expanduser().resolve()
        job_dirs: List[str] = list(render_art.data.get("job_dirs") or collect_art.data.get("job_dirs") or [])
        collected_job_dirs = _collected_job_dirs(run_dir, collect_ref) or set()

        candidates, rejected = _scan_stage_jobs_for_candidates(
            jobs_outdir=jobs_outdir,
            job_dirs=job_dirs,
            collected_job_dirs=collected_job_dirs,
            source_stage=self.stage,
            debug=debug,
            require_one_imag=require_one_imag,
            accept_reason=self._accept_reason(require_one_imag),
        )
        promoted = _promote_ranked_candidates(candidates, max_promote)
        _mark_promoted(candidates, promoted)

        ref = _write_candidate_batch(
            context=context,
            run_dir=run_dir,
            stage=self.stage,
            next_stage=self.next_stage,
            policy=self._policy(require_one_imag),
            max_promote=max_promote,
            candidates=candidates,
            promoted=promoted,
            rejected=rejected,
            parents=[render_ref, collect_ref],
        )

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": f"tsgen_{self.stage.lower()}_candidate_batch_complete",
                    "stage": self.name,
                    "n_candidates": len(candidates),
                    "n_promoted": len(promoted),
                    "candidate_batch_ref": _artifact_ref_dict(ref),
                }
            )

        return [ref]


class TSGenL1PromoteStage(TSGenPromoteStageBase):
    """Promote completed L1 OptTS jobs into a TSGenCandidateBatch."""

    stage = "L1"
    next_stage = "L2"
    render_stage = "L1"
    default_max_promote_key = "l1_max_promote"
    default_max_promote = 0
    require_one_imag_default = None
    policy_accept_all = "exploratory_accept_all"
    accept_reason = "exploratory_accept"

    def __init__(self):
        super().__init__(name="tsgen_l1_promote")


class TSGenL2PromoteStage(TSGenPromoteStageBase):
    """Promote completed L2 OptTS/Freq jobs into a TSGenCandidateBatch."""

    stage = "L2"
    next_stage = "VERIFY"
    render_stage = "L2"
    default_max_promote_key = "l2_max_promote"
    default_max_promote = 0
    require_one_imag_default = True
    policy_accept_all = "collected_l2_accept_all"
    policy_require_one_imag = "one_imaginary_frequency_gate"
    accept_reason = "collected_l2_accept"

    def __init__(self):
        super().__init__(name="tsgen_l2_promote")


class TSGenVerifyStage(BaseStage):
    """Exploratory-only TSGen verification for promoted L2 candidates.

    Confirmatory/fingerprint verification is intentionally deferred until the
    TSGen2 confirmatory contract is redesigned. For now, this pipeline-native
    stage classifies promoted L2 candidates using output/geometry presence and
    TS-like frequency criteria.
    """

    def __init__(self):
        super().__init__(name="tsgen_verify")

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = _run_dir(context)
        candidate_ref = _candidate_ref_for_stage(
            run_dir,
            context.get("tsgen_candidate_batch_ref"),
            stage="L2",
        )
        mode = str(_cfg(context, "mode", "exploratory")).strip().lower()
        return {
            "candidate_batch_id": getattr(candidate_ref, "artifact_id", ""),
            "mode": mode,
            "require_one_imag": bool(_cfg(context, "require_one_imag", True)),
            "min_abs_imag_cm1": _cfg(context, "min_abs_imag_cm1", None),
            "max_abs_imag_cm1": _cfg(context, "max_abs_imag_cm1", None),
        }

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        run_dir = _run_dir(context)
        mode = str(_cfg(context, "mode", "exploratory")).strip().lower()
        if mode != "exploratory":
            raise NotImplementedError(
                "Confirmatory TSGen verification is intentionally deferred in the "
                "pipeline-native implementation. Use mode='exploratory' for now."
            )

        candidate_ref = _candidate_ref_for_stage(
            run_dir,
            context.get("tsgen_candidate_batch_ref"),
            stage="L2",
        )
        if candidate_ref is None:
            raise RuntimeError("No L2 TSGenCandidateBatch artifact available for TSGen verification")

        candidate_payload = _load_json_artifact_payload(
            run_dir,
            candidate_ref,
            "candidate_batch_json",
            default_relpath="candidate_batch.json",
        )
        if str(candidate_payload.get("tsgen_stage") or "") != "L2":
            raise RuntimeError("TSGenVerifyStage requires a TSGenCandidateBatch with tsgen_stage='L2'")

        require_one_imag = bool(_cfg(context, "require_one_imag", True))
        min_abs_imag = _cfg(context, "min_abs_imag_cm1", None)
        max_abs_imag = _cfg(context, "max_abs_imag_cm1", None)
        min_abs_imag_f = float(min_abs_imag) if min_abs_imag not in (None, "") else None
        max_abs_imag_f = float(max_abs_imag) if max_abs_imag not in (None, "") else None

        promoted = list(candidate_payload.get("promoted") or [])
        verified: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for i, cand in enumerate(promoted):
            if not isinstance(cand, dict):
                rejected.append({"candidate_index": i, "reason": "malformed_candidate"})
                continue

            row = dict(cand)
            row.setdefault("candidate_index", i)
            xyz_path = Path(str(row.get("xyz") or "")).expanduser()
            out_path = Path(str(row.get("out") or "")).expanduser()

            if not xyz_path.is_file():
                row["verified"] = False
                row["verify_reason"] = "missing_xyz"
                rejected.append(row)
                continue
            if not out_path.is_file():
                row["verified"] = False
                row["verify_reason"] = "missing_out"
                rejected.append(row)
                continue

            try:
                freqs, _ = parse_frequencies_and_modes(out_path)
            except Exception as e:
                row["verified"] = False
                row["verify_reason"] = "freq_parse_error"
                row["message"] = str(e)
                rejected.append(row)
                continue

            imags = [float(f) for f in freqs if float(f) < 0.0]
            row["n_imag"] = len(imags)
            row["imag_cm1"] = min(imags) if imags else None
            row["abs_imag_cm1"] = abs(float(row["imag_cm1"])) if row["imag_cm1"] is not None else None

            if require_one_imag and len(imags) != 1:
                row["verified"] = False
                row["verify_reason"] = "bad_imag_count"
                rejected.append(row)
                continue

            if row["abs_imag_cm1"] is not None:
                if min_abs_imag_f is not None and float(row["abs_imag_cm1"]) < min_abs_imag_f:
                    row["verified"] = False
                    row["verify_reason"] = "imaginary_frequency_below_min"
                    rejected.append(row)
                    continue
                if max_abs_imag_f is not None and float(row["abs_imag_cm1"]) > max_abs_imag_f:
                    row["verified"] = False
                    row["verify_reason"] = "imaginary_frequency_above_max"
                    rejected.append(row)
                    continue

            row["verified"] = True
            row["status"] = "verified"
            row["verify_reason"] = "exploratory_ts_like_frequency_pattern"
            verified.append(row)

        verify_json = _workspace_dir(context) / "tsgen_verified_batch.json"
        payload = {
            "artifact_schema": "TSGenVerifiedBatch/v1",
            "mode": "exploratory",
            "tsgen_stage": "VERIFY",
            "source_stage": "L2",
            "policy": "exploratory_one_imaginary_frequency_gate" if require_one_imag else "exploratory_output_geometry_gate",
            "require_one_imag": require_one_imag,
            "min_abs_imag_cm1": min_abs_imag_f,
            "max_abs_imag_cm1": max_abs_imag_f,
            "n_input_promoted": len(promoted),
            "n_verified": len(verified),
            "n_rejected": len(rejected),
            "verified": verified,
            "rejected": rejected,
        }
        verify_json.parent.mkdir(parents=True, exist_ok=True)
        verify_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        art = Artifact(
            kind="TSGenVerifiedBatch",
            data={
                **payload,
                "verified_batch_json": "verified_batch.json",
            },
            parents=[candidate_ref],
        )
        ref = art.write(run_dir=run_dir, copy_files=[(verify_json, "verified_batch.json")])
        context["tsgen_verified_batch_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": "tsgen_verify_complete",
                    "stage": self.name,
                    "mode": "exploratory",
                    "n_verified": len(verified),
                    "n_rejected": len(rejected),
                    "verified_batch_ref": _artifact_ref_dict(ref),
                }
            )

        return [ref]
