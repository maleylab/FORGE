from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import Artifact, ArtifactRef, sha256_file
from .stages import BaseStage
from labtools.plans.types import PlanEntry


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _cfg(context: Dict[str, Any], key: str, default: Any = None, *, required: bool = False) -> Any:
    """Get a config value without assuming a single context shape.

    Preference order:
      1. direct stage context (ctx[key])
      2. PipelineRun.config (ctx['run'].config[key])
    """
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


def _sbatch_template_name(value: str) -> str:
    """Normalize an sbatch-template argument to the filename expected by dispatch()."""
    return Path(str(value)).name


def _job_output_signature(job_dir: Path) -> Dict[str, Any]:
    """Lightweight signature of files in a job directory for collect fingerprints."""
    sig: List[Dict[str, Any]] = []
    if not job_dir.is_dir():
        return {"files": sig}
    for p in sorted(job_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".out", ".gbw", ".engrad", ".hess", ".xyz", ".inp", ".log"}:
            continue
        try:
            st = p.stat()
            sig.append({"name": p.name, "bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
        except Exception:
            sig.append({"name": p.name, "bytes": -1, "mtime_ns": -1})
    return {"files": sig}


def _terminal_no_output_state(scheduler_state: str) -> str:
    state = str(scheduler_state or "UNKNOWN").upper()
    if state in {"CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "COMPLETED"}:
        return f"NO_OUTPUT_{state}"
    return "NO_OUTPUT"


class PlanRenderStage(BaseStage):
    def __init__(self):
        super().__init__(name="plan_render")

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        plan_value = _cfg(context, "plan", None)
        if plan_value is None:
            plan_value = _cfg(context, "plan_path", required=True)
        plan = Path(str(plan_value)).expanduser().resolve()
        fp: Dict[str, Any] = {
            "plan_path": str(plan),
            "plan_sha256": sha256_file(plan) if plan.is_file() else "",
        }
        # include template tree hash only if we can find the repo root from the installed module
        try:
            import labtools.submit as _submit_mod

            repo_root = Path(_submit_mod.__file__).resolve().parents[3]
            template_root = repo_root / "templates"
            parts: List[str] = []
            if template_root.exists():
                for p in sorted(template_root.rglob("*.j2")):
                    parts.append(str(p.relative_to(template_root)))
                    parts.append(sha256_file(p))
            fp["templates_sha256"] = _sha256_bytes("\n".join(parts).encode("utf-8")) if parts else ""
        except Exception:
            fp["templates_sha256"] = ""
        return fp

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        from labtools.plans.load import load_planentries_jsonl
        from labtools.plans.render import render_planentries
        from labtools.plans.orca_render import render_single_job_orca

        plan_value = _cfg(context, "plan", None)
        if plan_value is None:
            plan_value = _cfg(context, "plan_path", required=True)
        plan_path = Path(str(plan_value)).expanduser().resolve()

        outdir_value = _cfg(context, "outdir", None)
        jobs_outdir = Path(str(outdir_value)).expanduser().resolve() if outdir_value else (_workspace_dir(context) / "jobs")
        jobs_outdir.mkdir(parents=True, exist_ok=True)

        entries = load_planentries_jsonl(plan_path)
        render_planentries(
            entries,
            outdir=jobs_outdir,
            render_func=render_single_job_orca,
            write_plan_entry_json=True,
        )

        job_dirs = sorted([p.name for p in jobs_outdir.iterdir() if p.is_dir()])

        art = Artifact(
            kind="RenderBatch",
            data={
                "plan": str(plan_path),
                "jobs_outdir": str(jobs_outdir),
                "job_dirs": job_dirs,
                "n_jobs": len(job_dirs),
            },
        )
        ref = art.write(run_dir=_run_dir(context))
        context["render_batch_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": "render_complete",
                    "stage": self.name,
                    "n_jobs": len(job_dirs),
                    "jobs_outdir": str(jobs_outdir),
                }
            )

        return [ref]


class OrcaSubmitStage(BaseStage):
    def __init__(self):
        super().__init__(name="orca_submit")

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = _run_dir(context)
        render_ref = context.get("render_batch_ref") or _find_latest_artifact_ref(run_dir, "RenderBatch")
        render_data: Dict[str, Any] = {}
        if render_ref is not None:
            try:
                render_data = Artifact.load(run_dir, render_ref).data
            except Exception:
                render_data = {}
        return {
            "render_batch_id": getattr(render_ref, "artifact_id", ""),
            "profile": _cfg(context, "profile", "medium"),
            "sbatch_template": _sbatch_template_name(str(_cfg(context, "sbatch_template", "single_orca_job.sbatch.j2"))),
            "validate_only": bool(_cfg(context, "validate_only", False)),
            "dry_run": bool(_cfg(context, "dry_run", False)),
            "jobs_outdir": render_data.get("jobs_outdir", ""),
            "job_dirs": render_data.get("job_dirs", []),
        }

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        from labtools.submit import dispatch

        run_dir = _run_dir(context)
        render_ref = context.get("render_batch_ref") or _find_latest_artifact_ref(run_dir, "RenderBatch")
        if render_ref is None:
            raise RuntimeError("No RenderBatch artifact available for submission")

        render_art = Artifact.load(run_dir, render_ref)
        jobs_outdir = Path(str(render_art.data["jobs_outdir"])).expanduser().resolve()
        job_dirs: List[str] = list(render_art.data["job_dirs"])

        profile = str(_cfg(context, "profile", "medium"))
        dry_run = bool(_cfg(context, "dry_run", False))
        validate_only = bool(_cfg(context, "validate_only", False))
        sbatch_template = _sbatch_template_name(str(_cfg(context, "sbatch_template", "single_orca_job.sbatch.j2")))

        job_ids: Dict[str, str] = {}
        sbatch_scripts: Dict[str, str] = {}

        for jd in job_dirs:
            job_dir = jobs_outdir / jd
            inp_files = sorted(job_dir.glob("*.inp"))
            if not inp_files:
                raise RuntimeError(f"No .inp file found in {job_dir}")
            inp_file = inp_files[0]

            job_id = dispatch(
                inp_file,
                mode="job",
                profile=profile,
                job_name=jd,
                sbatch_template=sbatch_template,
                dry_run=dry_run,
                submit_cwd=job_dir,
                sbatch_chdir=job_dir,
                validate_only=validate_only,
            )
            job_ids[jd] = str(job_id or "")
            sbatch_scripts[jd] = str((job_dir / "job.sbatch").resolve())

        art = Artifact(
            kind="SubmitBatch",
            data={
                "render_batch_ref": {
                    "artifact_id": render_ref.artifact_id,
                    "kind": render_ref.kind,
                    "relpath": render_ref.relpath,
                },
                "jobs_outdir": str(jobs_outdir),
                "job_dirs": job_dirs,
                "job_ids": job_ids,
                "sbatch_scripts": sbatch_scripts,
                "backend": "slurm",
                "profile": profile,
                "dry_run": dry_run,
                "validate_only": validate_only,
                "sbatch_template": sbatch_template,
                "n_jobs": len(job_dirs),
            },
            parents=[render_ref],
        )
        ref = art.write(run_dir=run_dir)
        context["submit_batch_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": "submit_complete",
                    "stage": self.name,
                    "n_jobs": len(job_dirs),
                    "dry_run": dry_run,
                    "validate_only": validate_only,
                }
            )

        return [ref]


class OrcaCollectStage(BaseStage):
    def __init__(self):
        super().__init__(name="orca_collect")

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from labtools.pipeline.slurm_status import gather_submit_status

        run_dir = _run_dir(context)
        submit_ref = context.get("submit_batch_ref") or _find_latest_artifact_ref(run_dir, "SubmitBatch")
        submit_data: Dict[str, Any] = {}
        if submit_ref is not None:
            try:
                submit_data = Artifact.load(run_dir, submit_ref).data
            except Exception:
                submit_data = {}

        status = gather_submit_status(run_dir)
        per_job: List[Dict[str, Any]] = []
        jobs_outdir = (
            Path(str(submit_data.get("jobs_outdir") or "")).expanduser().resolve()
            if submit_data.get("jobs_outdir")
            else None
        )

        for rec in (status.get("per_job") or []):
            jd = str(rec.get("job_dir") or "")
            if not jd:
                continue
            job_dir = (jobs_outdir / jd) if jobs_outdir is not None else Path(jd)
            per_job.append(
                {
                    "job_dir": jd,
                    "job_id": str(rec.get("job_id") or ""),
                    "state": str(rec.get("state") or ""),
                    "output_sig": _job_output_signature(job_dir),
                }
            )

        per_job = sorted(per_job, key=lambda x: x["job_dir"])
        return {
            "submit_batch_id": getattr(submit_ref, "artifact_id", ""),
            "per_job": per_job,
        }

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        from labtools.pipeline.slurm_status import gather_submit_status
        from labtools.orca.collect import collect_job_record
        from labtools.orca.queues import classify_record

        run_dir = _run_dir(context)
        workspace_dir = _workspace_dir(context)
        submit_ref = context.get("submit_batch_ref") or _find_latest_artifact_ref(run_dir, "SubmitBatch")
        if submit_ref is None:
            raise RuntimeError("No SubmitBatch artifact available for collection")

        submit_art = Artifact.load(run_dir, submit_ref)
        jobs_outdir = Path(str(submit_art.data["jobs_outdir"])).expanduser().resolve()
        job_dirs: List[str] = list(submit_art.data.get("job_dirs") or [])
        submit_status = gather_submit_status(run_dir)
        status_by_dir: Dict[str, Dict[str, Any]] = {
            str(r.get("job_dir")): r for r in (submit_status.get("per_job") or [])
        }

        counts: Dict[str, int] = {}
        classified_counts: Dict[str, int] = {}
        scheduler_terminal_counts: Dict[str, int] = {}

        tmp_dir = workspace_dir / "collect_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        records_path = tmp_dir / f"collect_{submit_ref.artifact_id}.jsonl"

        def _bump(bucket: Dict[str, int], key: str) -> None:
            bucket[key] = bucket.get(key, 0) + 1

        with records_path.open("w", encoding="utf-8") as f:
            for jd in job_dirs:
                job_dir = jobs_outdir / jd
                sched = status_by_dir.get(jd, {})
                scheduler_state = str(sched.get("state") or "UNKNOWN")

                row: Dict[str, Any] = {
                    "job_dir": jd,
                    "job_path": str(job_dir),
                    "job_id": str(sched.get("job_id") or ""),
                    "scheduler_state": scheduler_state,
                    "scheduler_source": str(sched.get("source") or ""),
                    "scheduler_reason": str(sched.get("reason") or ""),
                    "scheduler_exit_code": str(sched.get("exit_code") or ""),
                }

                if scheduler_state in {"PENDING", "RUNNING", "DRY_RUN", "VALIDATED", "UNSUBMITTED"}:
                    collection_state = scheduler_state
                else:
                    try:
                        rec = collect_job_record(job_dir)
                    except Exception as e:
                        row["collect_error"] = str(e)
                        rec = None

                    if rec is None:
                        collection_state = _terminal_no_output_state(scheduler_state)
                        _bump(scheduler_terminal_counts, scheduler_state)
                    else:
                        row["record"] = rec
                        try:
                            classification = classify_record(rec)
                        except Exception as e:
                            classification = {"status": "CLASSIFY_ERROR", "message": str(e)}

                        row["classification"] = classification
                        cstatus = str((classification or {}).get("status") or "UNKNOWN")
                        _bump(classified_counts, cstatus)
                        collection_state = "COLLECTED"

                row["collection_state"] = collection_state
                _bump(counts, collection_state)
                f.write(json.dumps(row) + "\n")

        art = Artifact(
            kind="CollectBatch",
            data={
                "submit_batch_ref": {
                    "artifact_id": submit_ref.artifact_id,
                    "kind": submit_ref.kind,
                    "relpath": submit_ref.relpath,
                },
                "jobs_outdir": str(jobs_outdir),
                "job_dirs": job_dirs,
                "n_jobs": len(job_dirs),
                "counts": counts,
                "classified_counts": classified_counts,
                "scheduler_terminal_counts": scheduler_terminal_counts,
                "records_jsonl": "records.jsonl",
            },
            parents=[submit_ref],
        )
        ref = art.write(run_dir=run_dir, copy_files=[(records_path, "records.jsonl")])
        context["collect_batch_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit(
                {
                    "event": "collect_complete",
                    "stage": self.name,
                    "n_jobs": len(job_dirs),
                    "counts": counts,
                    "classified_counts": classified_counts,
                    "scheduler_terminal_counts": scheduler_terminal_counts,
                }
            )

        return [ref]



def _planentry_from_json(path: Path) -> PlanEntry:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return PlanEntry(
        id=obj["id"],
        schema_name=obj["schema"]["name"],
        schema_version=int(obj["schema"]["version"]),
        task=obj["intent"]["task"],
        system=obj["intent"]["system"],
        parameters=dict(obj.get("parameters") or {}),
        tags=list((obj.get("metadata") or {}).get("tags") or []),
        notes=(obj.get("metadata") or {}).get("notes"),
    )


def _downstream_allowed_statuses(context: Dict[str, Any]) -> List[str]:
    raw = _cfg(context, "downstream_on_status", ["OK_MIN", "OK_TS", "OK_NOFREQ", "OK_SP"])
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return [str(x) for x in (raw or []) if str(x).strip()]


class MaterializeDownstreamStage(BaseStage):
    def __init__(self):
        super().__init__(name="materialize_downstream")

    def fingerprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = _run_dir(context)
        collect_ref = context.get("collect_batch_ref") or _find_latest_artifact_ref(run_dir, "CollectBatch")
        child_task = str(_cfg(context, "downstream_task", ""))
        child_method = _cfg(context, "downstream_method", None)
        child_basis = _cfg(context, "downstream_basis", None)
        raw_suffix = _cfg(context, "downstream_id_suffix", None)
        child_suffix = str(raw_suffix).strip() if raw_suffix not in (None, "") else (child_task or "child")
        allowed = _downstream_allowed_statuses(context)
        ready: List[str] = []
        if collect_ref is not None:
            art = Artifact.load(run_dir, collect_ref)
            rec_rel = str(art.data.get("records_jsonl") or "")
            rec_path = (run_dir / "artifacts" / collect_ref.relpath / rec_rel).resolve() if rec_rel else None
            if rec_path is not None and rec_path.is_file():
                for line in rec_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if str(row.get("collection_state") or "") != "COLLECTED":
                        continue
                    cls = row.get("classification") or {}
                    if str(cls.get("status") or "") in set(allowed):
                        ready.append(str(row.get("job_dir") or ""))
        return {
            "collect_batch_id": getattr(collect_ref, "artifact_id", ""),
            "downstream_task": child_task,
            "downstream_method": child_method,
            "downstream_basis": child_basis,
            "downstream_id_suffix": child_suffix,
            "downstream_on_status": allowed,
            "ready_jobs": sorted([x for x in ready if x]),
        }

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        from labtools.csvpipe.emit import emit_planentries_jsonl
        from labtools.plans.render import render_planentries
        from labtools.plans.orca_render import render_single_job_orca

        run_dir = _run_dir(context)
        workspace_dir = _workspace_dir(context)
        collect_ref = context.get("collect_batch_ref") or _find_latest_artifact_ref(run_dir, "CollectBatch")
        if collect_ref is None:
            raise RuntimeError("No CollectBatch artifact available for downstream materialization")

        child_task = str(_cfg(context, "downstream_task", "")).strip()
        if not child_task:
            raise RuntimeError("No downstream_task configured")

        child_method = _cfg(context, "downstream_method", None)
        child_basis = _cfg(context, "downstream_basis", None)
        raw_suffix = _cfg(context, "downstream_id_suffix", None)
        child_suffix = str(raw_suffix).strip() if raw_suffix not in (None, "") else child_task
        allowed = set(_downstream_allowed_statuses(context))

        collect_art = Artifact.load(run_dir, collect_ref)
        rec_rel = str(collect_art.data.get("records_jsonl") or "")
        if not rec_rel:
            raise RuntimeError("CollectBatch does not provide records_jsonl")
        rec_path = (run_dir / "artifacts" / collect_ref.relpath / rec_rel).resolve()
        if not rec_path.is_file():
            raise RuntimeError(f"Collect records not found: {rec_path}")

        child_root = workspace_dir / "downstream" / f"{collect_ref.artifact_id}__{child_task}"
        child_jobs_outdir = child_root / "jobs"
        child_structures_dir = child_root / "structures"
        child_jobs_outdir.mkdir(parents=True, exist_ok=True)
        child_structures_dir.mkdir(parents=True, exist_ok=True)
        child_plan_path = child_root / "planentries.jsonl"

        entries: List[PlanEntry] = []
        materialized_from: List[str] = []
        existing_ids: set[str] = set()

        if child_plan_path.is_file():
            try:
                for line in child_plan_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    cid = str(obj.get("id") or "").strip()
                    if cid:
                        existing_ids.add(cid)
            except Exception:
                pass

        if child_jobs_outdir.is_dir():
            for p in child_jobs_outdir.iterdir():
                if p.is_dir():
                    existing_ids.add(p.name)

        for line in rec_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("collection_state") or "") != "COLLECTED":
                continue
            cls = row.get("classification") or {}
            if str(cls.get("status") or "") not in allowed:
                continue

            parent_job_dir = Path(str(row.get("job_path") or "")).expanduser().resolve()
            plan_json = parent_job_dir / "plan_entry.json"
            if not plan_json.is_file():
                continue

            parent = _planentry_from_json(plan_json)
            child_id_base = f"{parent.id}__{child_suffix}"
            if child_id_base in existing_ids:
                # Already materialized. Do not create child_id__2 on repeated advance.
                continue
            child_id = child_id_base
            existing_ids.add(child_id)

            system = parent.system
            final_geom = (((row.get("record") or {}).get("parsed") or {}).get("final_geometry") or None)

            # Canonical child-local structure layout:
            #   structures/<child_id>/structure.xyz
            child_struct_dir = child_structures_dir / child_id
            child_struct_dir.mkdir(parents=True, exist_ok=True)
            xyz_path = child_struct_dir / "structure.xyz"

            if isinstance(final_geom, list) and final_geom:
                xyz_text = (
                    str(len(final_geom))
                    + "\n"
                    + f"materialized from {parent.id}\n"
                    + "\n".join(str(x) for x in final_geom)
                    + "\n"
                )
                xyz_path.write_text(xyz_text, encoding="utf-8")
            else:
                src_xyz = None

                parent_xyz = parent_job_dir / "structure.xyz"
                if parent_xyz.is_file():
                    src_xyz = parent_xyz

                if src_xyz is None and isinstance(system, dict):
                    s = system.get("structure") or system.get("xyz")
                    if s:
                        p = Path(str(s)).expanduser()
                        p = (parent_job_dir / p).resolve() if not p.is_absolute() else p.resolve()
                        if p.is_file():
                            src_xyz = p

                if src_xyz is None and isinstance(system, str):
                    p = Path(system).expanduser()
                    p = (parent_job_dir / p).resolve() if not p.is_absolute() else p.resolve()
                    if p.is_file():
                        src_xyz = p

                if src_xyz is None:
                    raise RuntimeError(f"Structure file not found: {parent.system}")

                xyz_path.write_text(src_xyz.read_text(encoding="utf-8"), encoding="utf-8")

            # PlanEntry.system must be a concrete path string for the renderer.
            system = str(xyz_path)

            params = dict(parent.parameters or {})
            params["job_type"] = child_task
            if child_method is not None:
                params["method"] = child_method
            if child_basis is not None:
                params["basis"] = child_basis

            tags = list(parent.tags or []) + [f"downstream:{child_task}", f"from:{parent.task}"]
            notes = f"Materialized from collect of {parent.id}" if not parent.notes else f"{parent.notes}; materialized to {child_task}"
            entries.append(
                PlanEntry(
                    id=child_id,
                    schema_name=parent.schema_name,
                    schema_version=parent.schema_version,
                    task=child_task,
                    system=system,
                    parameters=params,
                    tags=tags,
                    notes=notes,
                )
            )
            materialized_from.append(parent.id)

        if not entries:
            existing_ref = _find_latest_artifact_ref(run_dir, "RenderBatch")
            if existing_ref is not None:
                existing_art = Artifact.load(run_dir, existing_ref)
                src_collect = (((existing_art.data or {}).get("source_collect_ref") or {}).get("artifact_id") or "")
                child_task_existing = str((existing_art.data or {}).get("child_task") or "")
                if src_collect == collect_ref.artifact_id and child_task_existing == child_task:
                    context["render_batch_ref"] = existing_ref
                    return [existing_ref]
            raise RuntimeError("No eligible collected jobs available for downstream materialization")

        emit_planentries_jsonl(entries, child_plan_path, dry_run=False)
        render_planentries(entries, outdir=child_jobs_outdir, render_func=render_single_job_orca, write_plan_entry_json=True)
        job_dirs = sorted([p.name for p in child_jobs_outdir.iterdir() if p.is_dir()])

        art = Artifact(
            kind="RenderBatch",
            data={
                "plan": str(child_plan_path),
                "jobs_outdir": str(child_jobs_outdir),
                "job_dirs": job_dirs,
                "n_jobs": len(job_dirs),
                "source_collect_ref": {
                    "artifact_id": collect_ref.artifact_id,
                    "kind": collect_ref.kind,
                    "relpath": collect_ref.relpath,
                },
                "child_task": child_task,
                "materialized_from": materialized_from,
            },
            parents=[collect_ref],
        )
        ref = art.write(run_dir=run_dir, copy_files=[(child_plan_path, "planentries.jsonl")])
        context["render_batch_ref"] = ref

        run = context.get("run")
        if run is not None and hasattr(run, "audit"):
            run.audit({
                "event": "materialize_complete",
                "stage": self.name,
                "n_jobs": len(job_dirs),
                "child_task": child_task,
                "source_collect_ref": collect_ref.artifact_id,
            })

        return [ref]


