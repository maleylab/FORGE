from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# Canonical failure labels expected by RestartEngine.RESTART_RULES
SCF_CONVERGENCE = "scf_convergence"
GEOM_CONVERGENCE = "geom_convergence"
TS_FAILED = "ts_failed"
UNKNOWN_FAILURE = "unknown_failure"


def _lower_errors(rec: Dict[str, Any]) -> List[str]:
    """Normalize any error messages in the parsed record to lowercase."""
    errors = rec.get("errors") or rec.get("error_messages") or []
    if isinstance(errors, str):
        return [errors.lower()]
    return [str(e).lower() for e in errors]


def _job_type(rec: Dict[str, Any]) -> str:
    """Best-effort extraction of a job type string from the record."""
    jt = rec.get("job_type") or rec.get("type") or ""
    return str(jt).lower()


def _load_output_text(out_path: Path) -> str:
    """Read ORCA .out text defensively, returning lowercase text or ''."""
    try:
        text = out_path.read_text(errors="ignore")
    except Exception:
        return ""
    return text.lower()


def classify_failure(rec: Dict[str, Any], out_path: Path) -> str:
    """
    Best-effort classification of an ORCA failure.

    Parameters
    ----------
    rec
        Parsed ORCA record from ``labtools.orca.parse.parse_orca_output``.
        This function is tolerant of missing keys; it only uses what is present.
    out_path
        Path to the ORCA output file. The text is scanned for tell-tale
        strings if the structured record is not decisive.

    Returns
    -------
    str
        One of:
        - ``"scf_convergence"``
        - ``"geom_convergence"``
        - ``"ts_failed"``
        - ``"unknown_failure"``
    """
    errors = _lower_errors(rec)
    jt = _job_type(rec)
    text = _load_output_text(out_path)

    # ------------------------------------------------------------------
    # 1. SCF convergence problems
    # ------------------------------------------------------------------
    # Structured hints
    if rec.get("scf_converged") is False:
        return SCF_CONVERGENCE

    # Error strings
    scf_keywords = [
        "scf failed to converge",
        "scf not converged",
        "scf convergence failure",
        "could not converge scf",
    ]
    if any(any(k in e for k in scf_keywords) for e in errors):
        return SCF_CONVERGENCE

    # Raw text scan fallback
    if any(k in text for k in scf_keywords):
        return SCF_CONVERGENCE

    # ------------------------------------------------------------------
    # 2. Geometry / optimization convergence problems
    # ------------------------------------------------------------------
    # Structured hints
    if jt in {"opt", "geometry_optimization", "optfreq"}:
        if rec.get("opt_converged") is False:
            return GEOM_CONVERGENCE

    # Error strings
    geom_keywords = [
        "geometry optimization failed",
        "geometry convergence failure",
        "exceeded maximum number of optimization cycles",
        "geometry not converged",
    ]
    if any(any(k in e for k in geom_keywords) for e in errors):
        return GEOM_CONVERGENCE

    if any(k in text for k in geom_keywords):
        return GEOM_CONVERGENCE

    # ------------------------------------------------------------------
    # 3. TS-specific failures
    # ------------------------------------------------------------------
    # If job type looks TS-ish and we did not converge, bias to TS_FAILED
    if jt in {"tsopt", "ts", "optts"}:
        # Heuristic flags that might be present in rec:
        # - "ts_converged": bool
        # - "n_imag": int
        ts_conv = rec.get("ts_converged")
        if ts_conv is False:
            return TS_FAILED

        # Wrong number of imaginary frequencies is classic TS fail
        n_imag = rec.get("n_imag")
        if isinstance(n_imag, int) and n_imag != 1:
            return TS_FAILED

        ts_keywords = [
            "transition state search failed",
            "ts optimization failed",
            "failed to locate a transition state",
        ]
        if any(any(k in e for k in ts_keywords) for e in errors):
            return TS_FAILED
        if any(k in text for k in ts_keywords):
            return TS_FAILED

    # ------------------------------------------------------------------
    # 4. Fallback
    # ------------------------------------------------------------------
    return UNKNOWN_FAILURE
