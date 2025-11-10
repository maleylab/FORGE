from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import hashlib, json, time

try:
    import yaml
except Exception:
    yaml = None

try:
    from jinja2 import Template
except Exception:
    Template = None

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_yaml(obj: Dict[str, Any], path: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write YAML files")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def render_id(id_pattern: Optional[str], ctx: Dict[str, Any], fallback: str) -> str:
    if id_pattern and Template is not None:
        try:
            return Template(id_pattern).render(**ctx)
        except Exception:
            pass
    return fallback

def emit_job_entries(
    job_docs: List[Dict[str, Any]],
    out_dir: Path,
    id_pattern: Optional[str] = None,
    provenance: bool = False,
    inputs_meta: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Write each job doc (already shaped like a single jobs[] entry) as its own YAML file for debugging.

    Returns list of paths written.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for idx, job in enumerate(job_docs):
        ctx = {"job": job}
        job_id = job.get("id") or render_id(id_pattern, ctx, f"job_{idx}")
        path = out_dir / f"{job_id}.yaml"
        write_yaml(job, path)
        written.append(path)

        if provenance:
            prov = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "yaml_sha256": _sha256_file(path),
                "inputs": inputs_meta or {},
                "row_raw": job.get("_row_raw"),
                "expanded_axes": job.get("_expanded_axes"),
            }
            with open(path.with_suffix(".yaml.prov.json"), "w", encoding="utf-8") as f:
                json.dump(prov, f, indent=2)
    return written

def emit_combined_plan(
    job_docs: List[Dict[str, Any]],
    combine_path: Path,
    version: str = "1",
) -> Path:
    """Write the single plan file matching labtools/schemas/plan.schema.json (version + jobs array).

    """
    plan = {"version": version, "jobs": []}
    for job in job_docs:
        # strip internal annotations
        j = {k: v for k, v in job.items() if not k.startswith("_")}
        plan["jobs"].append(j)
    write_yaml(plan, combine_path)
    return combine_path
