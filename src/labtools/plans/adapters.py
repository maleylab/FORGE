# labtools/plans/adapters.py

from __future__ import annotations

from typing import Dict, Any, Union

from labtools.plans.types import PlanEntry


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
    job.setdefault("geom_lines", [])

    return job
