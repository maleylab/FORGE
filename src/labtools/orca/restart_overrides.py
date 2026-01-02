# src/labtools/orca/restart_overrides.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RestartOverrides:
    """
    Canonical container for restart-time template overrides.

    All fields default to empty containers so workflow templates and
    context building always behave consistently.
    """

    # always a list
    global_flags_add: List[str] = field(default_factory=list)

    # always dicts
    scf: Dict[str, str | int | float] = field(default_factory=dict)
    geom: Dict[str, str | int | float] = field(default_factory=dict)
    freq: Dict[str, str | int | float] = field(default_factory=dict)
    nmr: Dict[str, str | int | float] = field(default_factory=dict)

    def merge(self, other: "RestartOverrides") -> "RestartOverrides":
        """
        Optional: safe merge utility if ever needed.
        """
        return RestartOverrides(
            global_flags_add=[*self.global_flags_add, *other.global_flags_add],
            scf={**self.scf, **other.scf},
            geom={**self.geom, **other.geom},
            freq={**self.freq, **other.freq},
            nmr={**self.nmr, **other.nmr},
        )
