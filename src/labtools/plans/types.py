from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class PlanEntry:
    """
    A PlanEntry is a fully specified, execution-agnostic declaration
    of a single computational task.

    Invariants (v1):
    - parameters MUST be fully expanded (no fanout, no axes)
    - no execution details (SLURM, schedulers, paths)
    - safe to serialize to JSONL
    """

    # Required fields (NO defaults)
    id: str
    schema_name: str
    task: str
    system: str
    parameters: Dict[str, Any]

    # Optional / defaulted fields (MUST come last)
    schema_version: int = 1
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
