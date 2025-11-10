from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Solvent:
    model: str  # 'none' or 'cpcm'
    name: str | None = None
    epsilon: float | None = None
    refrac: float | None = None


_SOLV = {
    "toluene": {"epsilon": 2.3741, "refrac": 1.4969},
    "thf": {"epsilon": 7.4257, "refrac": 1.4070},
    "dcm": {"epsilon": 8.93, "refrac": 1.4244},
    "acetonitrile": {"epsilon": 35.688, "refrac": 1.3444},
    "water": {"epsilon": 78.355, "refrac": 1.3330},
}


def make_solvent(spec: str, *, epsilon: float | None, refrac: float | None) -> Solvent:
    if spec == "none":
        return Solvent("none")
    if spec == "custom":
        if epsilon is None:
            raise SystemExit("--solvent custom requires --solvent.epsilon <ε>")
        return Solvent("cpcm", None, epsilon, refrac)
    key = spec.lower()
    if key not in _SOLV:
        raise SystemExit(f"Unknown solvent '{spec}'. Try one of: {', '.join(sorted(_SOLV))} or 'custom'.")
    d = _SOLV[key]
    return Solvent("cpcm", key, d["epsilon"], d["refrac"])
