# labtools/plans/adapters.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union, List

from labtools.plans.types import PlanEntry


def _read_xyz_atom_lines(xyz_path: str | Path) -> List[str]:
    """Read an .xyz file and return the coordinate body lines ("El x y z").

    Supports:
      - Strict XYZ: first line = atom count, second line = comment, followed by
        exactly N coordinate lines.
      - Loose XYZ: no atom count line; each non-empty line is treated as a
        coordinate line.

    This is intentionally lightweight and preserves the original coordinate
    formatting from the file as much as possible.
    """
    p = Path(xyz_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    lines = p.read_text(encoding="utf-8").splitlines()
    # Trim trailing blanks
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError(f"Empty XYZ file: {p}")

    first = lines[0].strip()
    if first.isdigit():
        n = int(first)
        if len(lines) < 2 + n:
            raise ValueError(f"XYZ too short for {n} atoms: {p}")
        body = lines[2 : 2 + n]
    else:
        body = [ln for ln in lines if ln.strip()]

    # Basic validation: require at least 4 columns per line
    for i, ln in enumerate(body):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line {i} in {p}: {ln!r}")

    return body


def planentry_to_legacy_job(
    entry: Union[PlanEntry, Dict[str, Any]],
    *,
    system_key: str = "system",
) -> Dict[str, Any]:
    """
    Convert a PlanEntry into the legacy job-dict format expected
    by existing renderers.

    This function MUST preserve semantics exactly.
    No execution logic belongs here.

    Accepts either:
      - PlanEntry dataclass
      - PlanEntry-shaped dict (as emitted to JSONL)
    """

    # ---------------------------------------------------------
    # Normalize input
    # ---------------------------------------------------------

    if isinstance(entry, dict):
        try:
            system = entry["intent"]["system"]
            parameters = entry.get("parameters", {})
        except KeyError as e:
            raise KeyError(f"Invalid PlanEntry dict structure, missing key: {e}") from None
    else:
        system = entry.system
        parameters = entry.parameters

    if not isinstance(parameters, dict):
        raise TypeError("PlanEntry parameters must be a dict")

    # ---------------------------------------------------------
    # Build legacy job dict
    # ---------------------------------------------------------

    job: Dict[str, Any] = {}

    # Preserve legacy top-level system identifier
    job[system_key] = system

    # Parameters are passed through verbatim
    for key, value in parameters.items():
        job[key] = value

    return job


def planentry_to_dict(entry: PlanEntry) -> Dict[str, Any]:
    """
    Canonical serialized representation of a PlanEntry.

    This MUST round-trip through:
        planentry_from_dict(planentry_to_dict(entry))
    """
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
            "tags": list(entry.tags) if entry.tags else [],
            "notes": entry.notes,
        },
    }


def planentry_to_render_context(pe) -> dict:
    """
    Canonical, render-safe normalization layer.

    Output contract:
    - Safe to pass directly into Jinja ORCA templates
    - All optional keys present
    - No chemistry logic, only plumbing
    """

    job = dict(pe.parameters)

    # --- core identifiers ---
    job["id"] = pe.id
    job["task"] = pe.task
    job["system"] = pe.system

    # --- ORCA-required normalization ---
    job.setdefault("charge", None)
    job.setdefault("multiplicity", None)
    job.setdefault("mult", job.get("multiplicity"))

    job.setdefault("method", None)
    job.setdefault("basis", None)

    # --- optional ORCA blocks ---
    job.setdefault("flags", [])
    job.setdefault("restart_flags", [])
    job.setdefault("maxcore_mb", None)
    job.setdefault("scf", {})
    job.setdefault("cpcm", {})

    # Geometry lines for ORCA templates.
    # The canonical templates expect `xyz_lines` (list[str]) containing the
    # "El x y z" coordinate lines.
    # For backwards compatibility we also expose `geom_lines`.
    xyz_lines: List[str] = []
    if isinstance(pe.system, dict):
        struct_path = pe.system.get("structure") or pe.system.get("xyz")
    else:
        struct_path = None

    if struct_path:
        try:
            xyz_lines = _read_xyz_atom_lines(struct_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read XYZ structure for PlanEntry id={pe.id!r} at {struct_path!r}: {e}"
            ) from e

    job.setdefault("xyz_lines", xyz_lines)
    job.setdefault("geom_lines", xyz_lines)

    # Many ORCA templates in this repo expect a `geom` dict and a `freq` dict.
    # Populate safe defaults if absent.
    job.setdefault("geom", {})
    job.setdefault("freq", {})

    return job
