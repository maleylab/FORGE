# labtools/plans/schema.py

from __future__ import annotations

from typing import Any, Dict


def validate_planentry_dict(d: Dict[str, Any]) -> None:
    """
    Structural validation for PlanEntry JSONL payloads.

    This validates the *serialized* representation, not the dataclass.
    """

    if not isinstance(d, dict):
        raise ValueError("PlanEntry payload must be a dict")

    # -------------------------
    # Required top-level keys
    # -------------------------
    required_top = {
        "id": str,
        "schema": dict,
        "intent": dict,
        "metadata": dict,
    }

    for key, typ in required_top.items():
        if key not in d:
            raise ValueError(f"Missing PlanEntry field: '{key}'")
        if not isinstance(d[key], typ):
            raise ValueError(f"Field '{key}' must be of type {typ.__name__}")

    # -------------------------
    # Parameters (explicit check)
    # -------------------------
    if "parameters" not in d:
        raise ValueError("Missing PlanEntry field: 'parameters'")
    if not isinstance(d["parameters"], dict):
        raise ValueError("parameters must be a dict")

    # -------------------------
    # Schema block
    # -------------------------
    schema = d["schema"]
    if "name" not in schema or "version" not in schema:
        raise ValueError("schema must contain 'name' and 'version'")
    if not isinstance(schema["name"], str):
        raise ValueError("schema.name must be a string")
    if not isinstance(schema["version"], int):
        raise ValueError("schema.version must be an int")

    # -------------------------
    # Intent block
    # -------------------------
    intent = d["intent"]
    if "task" not in intent or "system" not in intent:
        raise ValueError("intent must contain 'task' and 'system'")
    if not isinstance(intent["task"], str):
        raise ValueError("intent.task must be a string")
    if not isinstance(intent["system"], str):
        raise ValueError("intent.system must be a string")

    # -------------------------
    # Metadata
    # -------------------------
    meta = d["metadata"]

    if "tags" in meta and meta["tags"] is not None:
        if not isinstance(meta["tags"], list) or not all(isinstance(t, str) for t in meta["tags"]):
            raise ValueError("metadata.tags must be a list of strings")

    if "notes" in meta and meta["notes"] is not None:
        if not isinstance(meta["notes"], str):
            raise ValueError("metadata.notes must be a string or null")
