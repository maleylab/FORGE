# labtools/plans/factory.py

from typing import Dict, Any
from labtools.plans.types import PlanEntry
from labtools.plans.schema import validate_planentry_dict

def planentry_from_dict(d: Dict[str, Any]) -> PlanEntry:
    validate_planentry_dict(d)
    return PlanEntry(
        id=d["id"],
        schema_name=d["schema"]["name"],
        schema_version=d["schema"]["version"],
        task=d["intent"]["task"],
        system=d["intent"]["system"],
        parameters=d["parameters"],
        tags=d.get("metadata", {}).get("tags", []),
        notes=d.get("metadata", {}).get("notes"),
    )
