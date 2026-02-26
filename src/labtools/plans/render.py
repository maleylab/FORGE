from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from labtools.plans.types import PlanEntry
from labtools.plans.adapters import planentry_to_legacy_job
from labtools.plans.adapters import planentry_to_dict
from labtools.plans.adapters import planentry_to_render_context

from jinja2 import Environment, FileSystemLoader, ChainableUndefined

PLAN_TASK_TEMPLATES = {
    "sp": "orca/orca_sp.inp.j2",
    "opt": "orca/orca_opt.inp.j2",
    "freq": "orca/orca_freq.inp.j2",
    "optfreq": "orca/orca_optfreq.inp.j2",
    "nmr": "orca/orca_nmr.inp.j2",
    "sp-triplet": "orca/orca_sp_triplet.inp.j2",
    "gradient": "orca/orca_grad.inp.j2",
    "tsopt": "orca/ts_opt.inp.j2"
}


def _serialize_planentry(entry: PlanEntry) -> dict:
    return {
        "id": entry.id,
        "schema": {
            "name": entry.schema_name,
            "version": entry.schema_version,
        },
        "intent": {
            "task": entry.task,
            "system": entry.system,
        },
        "parameters": entry.parameters,
        "metadata": {
            "tags": entry.tags,
            "notes": entry.notes,
        },
    }


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_dirname(s: str) -> str:
    """
    Filesystem-safe name:
      - allow A–Z a–z 0–9 . _ -
      - replace everything else with _
    """
    s = (s or "").strip()
    if not s:
        return "job"
    return _SAFE_CHARS_RE.sub("_", s)


def _safe_job_dirname(outdir: Path, base: str) -> str:
    """
    Ensure the job directory name is unique under outdir.
    If a collision exists, append __N.
    """
    base = _sanitize_dirname(base)
    candidate = base
    n = 1
    while (outdir / candidate).exists():
        candidate = f"{base}__{n}"
        n += 1
    return candidate


def render_planentries(
    entries: Iterable[PlanEntry],
    *,
    render_func,
    outdir: Path,
    system_key: str = "system",
):
    """
    Render PlanEntries using an existing legacy renderer.

    Parameters
    ----------
    entries
        Iterable of PlanEntry objects
    render_func
        Existing renderer function (unchanged)
    outdir
        Output directory
    """

    outdir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries):
        legacy_job = planentry_to_render_context(entry)

        # FIX #1: name job directories by entry.id (sanitized), not job_00001
        # Fall back to job_{i:05d} if entry.id is missing/empty for any reason.
        base = entry.id if getattr(entry, "id", None) else f"job_{i:05d}"
        job_dir_name = _safe_job_dirname(outdir, base)

        job_dir = outdir / job_dir_name
        job_dir.mkdir(exist_ok=False)

        render_func(
            legacy_job,
            job_dir=job_dir,
        )

        # Optional: embed PlanEntry for provenance
        (job_dir / "plan_entry.json").write_text(
            json.dumps(planentry_to_dict(entry), indent=2),
            encoding="utf-8",
        )


def get_orca_template_env() -> Environment:
    """
    Canonical ORCA Jinja environment.
    Safe defaults, no StrictUndefined explosions.
    """

    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "templates" / "orca"
        if cand.is_dir():
            return Environment(
                loader=FileSystemLoader(str(cand.parent)),
                undefined=ChainableUndefined,
                autoescape=False,
            )

    raise RuntimeError("Could not locate templates/orca directory")
