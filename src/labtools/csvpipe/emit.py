from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import hashlib
import json
import time

try:
    import yaml
except Exception:
    yaml = None

try:
    from jinja2 import Template
except Exception:
    Template = None


# ============================================================================
# Legacy CSV → job YAML emission (UNCHANGED)
# ============================================================================

def emit_job_yaml_files(
    jobs: List[Dict[str, Any]],
    outdir: Path,
    id_field: str = "id",
    provenance: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> None | List[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    for job in jobs:
        job_id = job.get(id_field)
        if not job_id:
            raise ValueError(f"Missing id field {id_field} in job: {job}")
        fname = f"{job_id}.yaml"
        path = outdir / fname
        if not dry_run:
            with path.open("w") as f:
                yaml.safe_dump(job, f, sort_keys=False)
        out_paths.append(str(path))

    # Write provenance.json once
    if provenance and not dry_run:
        prov_path = outdir / "provenance.json"
        with prov_path.open("w") as f:
            json.dump(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_jobs": len(jobs),
                    "job_ids": [job.get(id_field) for job in jobs],
                    "provenance": provenance,
                },
                f,
                indent=2,
            )

    if dry_run:
        return [yaml.safe_dump(job, sort_keys=False) for job in jobs]

    return None


# ============================================================================
# PlanEntry helpers (ADD-ON; no impact on legacy paths)
# ============================================================================

def _stable_planentry_id(
    *,
    schema_name: str,
    schema_version: int,
    task: str,
    system: str,
    parameters: Dict[str, Any],
    index: Optional[int] = None,
) -> str:
    """Create a deterministic identifier for a PlanEntry."""
    payload = {
        "schema": {"name": schema_name, "version": int(schema_version)},
        "intent": {"task": task, "system": system},
        "parameters": parameters,
        "index": index,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    prefix = f"{system}_{task}".replace(" ", "_")
    return f"{prefix}_{digest[:12]}"


# ============================================================================
# CSVPIPE → PlanEntry ADAPTER (NEW, MINIMAL, CANONICAL)
# ============================================================================

def jobs_to_planentries(
    jobs: List[Dict[str, Any]],
    *,
    schema_name: str,
    schema_version: int = 1,
) -> List[Any]:
    """
    Convert expanded csvpipe job dicts into PlanEntry objects.

    Expected job dict shape (current invariant):
      - id
      - task
      - system
      - remaining keys are parameters
    """
    from labtools.plans.types import PlanEntry

    entries: List[PlanEntry] = []

    for i, job in enumerate(jobs):
        if "id" not in job:
            raise KeyError("csvpipe job missing 'id'")
        if "task" not in job:
            raise KeyError("csvpipe job missing 'task'")
        if "system" not in job:
            raise KeyError("csvpipe job missing 'system'")

        params = {
            k: v
            for k, v in job.items()
            if k not in ("id", "task", "system")
        }

        entry = PlanEntry(
            id=str(job["id"]),
            schema_name=str(schema_name),
            schema_version=int(schema_version),
            task=str(job["task"]),
            system=str(job["system"]),
            parameters=params,
        )

        entries.append(entry)

    return entries


# ============================================================================
# Legacy adapter (kept for backward compatibility / migration)
# ============================================================================

def job_to_planentry(
    job: Dict[str, Any],
    *,
    schema_name: str,
    schema_version: int,
    task: str,
    system_key: str = "structure",
    parameters_key: str = "parameters",
    index: Optional[int] = None,
):
    """Convert a legacy job dict into a PlanEntry."""
    from labtools.plans.types import PlanEntry

    system = str(job.get(system_key) or job.get("system") or "")
    if not system:
        raise ValueError(
            f"Cannot build PlanEntry: missing system field '{system_key}'"
        )

    params = job.get(parameters_key)
    if params is None:
        params = {
            k: v
            for k, v in job.items()
            if k not in {"id", "type", system_key, "template"}
        }
    if not isinstance(params, dict):
        raise ValueError(
            f"PlanEntry parameters must be a dict; got {type(params).__name__}"
        )

    entry_id = job.get("plan_id") or job.get("id")
    if not entry_id:
        entry_id = _stable_planentry_id(
            schema_name=schema_name,
            schema_version=schema_version,
            task=task,
            system=system,
            parameters=params,
            index=index,
        )

    return PlanEntry(
        id=str(entry_id),
        schema_name=str(schema_name),
        schema_version=int(schema_version),
        task=str(task),
        system=system,
        parameters=params,
        tags=job.get("tags"),
        notes=job.get("notes"),
    )


# ============================================================================
# PlanEntry → JSONL emission (UNCHANGED)
# ============================================================================

def emit_planentries_jsonl(
    entries: List[Any],
    outpath: Path,
    *,
    dry_run: bool = False,
) -> Optional[str]:
    """Write PlanEntry objects to JSONL."""
    outpath = outpath.expanduser().resolve()
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for e in entries:
        if hasattr(e, "__dataclass_fields__") and getattr(e, "id", None) is not None:
            d = {
                "id": e.id,
                "schema": {"name": e.schema_name, "version": e.schema_version},
                "intent": {"task": e.task, "system": e.system},
                "parameters": e.parameters,
                "metadata": {
                    "tags": e.tags,
                    "notes": e.notes,
                },
            }
        elif isinstance(e, dict):
            d = e
        else:
            raise TypeError(
                f"Unsupported PlanEntry payload type: {type(e).__name__}"
            )

        lines.append(json.dumps(d, sort_keys=False))

    text = "\n".join(lines) + ("\n" if lines else "")
    if dry_run:
        return text

    outpath.write_text(text, encoding="utf-8")
    return None
