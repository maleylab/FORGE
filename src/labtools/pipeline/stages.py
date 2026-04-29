from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .artifacts import ArtifactRef
from .gates import Gate, GateResult


@dataclass(frozen=True)
class StageResult:
    stage: str
    ok: bool
    outputs: List[ArtifactRef]
    gate_results: List[GateResult]
    message: str = ""
    elapsed_s: float = 0.0


class Stage(Protocol):
    """Stage interface."""

    name: str
    gates: List[Gate]

    def run(self, *, context: Dict[str, Any]) -> List[ArtifactRef]:
        """Execute stage and return output ArtifactRef(s)."""
        ...

    def describe(self) -> Dict[str, Any]: ...


@dataclass
class BaseStage:
    """Convenience base class."""

    name: str
    gates: List[Gate] = dataclasses.field(default_factory=list)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "gates": [getattr(g, "name", "gate") for g in self.gates]}

    def _run_with_gates(self, *, context: Dict[str, Any], fn) -> StageResult:
        t0 = time.time()
        outputs: List[ArtifactRef] = []
        msg = ""
        ok = False
        try:
            outputs = fn()
            ok = True
        except Exception as e:
            ok = False
            msg = str(e)

        gate_results: List[GateResult] = []
        if ok:
            for g in (self.gates or []):
                try:
                    gate_results.append(g.check(context=context))
                except Exception as e:
                    gate_results.append(
                        GateResult(name=getattr(g, "name", "gate"), severity=getattr(g, "severity", "fail"), passed=False, message=f"Gate error: {e}")
                    )

            # If any fail-severity gate fails, stage is not ok.
            for gr in gate_results:
                if (gr.severity == "fail") and (not gr.passed):
                    ok = False
                    msg = msg or f"QC gate failed: {gr.name}"
                    break

        return StageResult(
            stage=self.name,
            ok=ok,
            outputs=outputs,
            gate_results=gate_results,
            message=msg,
            elapsed_s=time.time() - t0,
        )
