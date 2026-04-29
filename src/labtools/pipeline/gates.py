from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Literal


GateSeverity = Literal["fail", "warn"]


@dataclass(frozen=True)
class GateResult:
    name: str
    severity: GateSeverity
    passed: bool
    message: str = ""
    details: Optional[Dict[str, Any]] = None


class Gate(Protocol):
    """QC gate interface."""

    name: str
    severity: GateSeverity

    def check(self, *, context: Dict[str, Any]) -> GateResult: ...


@dataclass
class SimpleGate:
    """A small helper gate for common use."""

    name: str
    severity: GateSeverity
    fn: Any  # callable(context) -> (passed: bool, message: str, details: dict|None)

    def check(self, *, context: Dict[str, Any]) -> GateResult:
        out = self.fn(context)
        if isinstance(out, GateResult):
            return out
        passed, message, details = out
        return GateResult(
            name=self.name,
            severity=self.severity,
            passed=bool(passed),
            message=str(message or ""),
            details=details if isinstance(details, dict) else None,
        )
