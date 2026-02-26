from __future__ import annotations

import json
from pathlib import Path
from typing import List

from labtools.plans.types import PlanEntry


def load_planentries_jsonl(path: Path) -> List[PlanEntry]:
    """
    Load PlanEntry objects from a JSONL file written by emit_planentries_jsonl.

    This expects the *nested document schema*:

    {
      "id": "...",
      "schema": {"name": "...", "version": 1},
      "intent": {"task": "...", "system": "..."},
      "parameters": {...},
      "metadata": {"tags": [...], "notes": "..."}
    }
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    entries: List[PlanEntry] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from e

            try:
                entry = PlanEntry(
                    id=d["id"],
                    schema_name=d["schema"]["name"],
                    schema_version=d["schema"]["version"],
                    task=d["intent"]["task"],
                    system=d["intent"]["system"],
                    parameters=d["parameters"],
                    tags=d.get("metadata", {}).get("tags", []),
                    notes=d.get("metadata", {}).get("notes"),
                )
            except KeyError as e:
                raise KeyError(f"{path}:{lineno}: missing key {e}") from e

            entries.append(entry)

    return entries
