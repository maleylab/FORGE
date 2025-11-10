# src/labtools/tsgen/stages/verify.py
"""
verify.py
FORGE | tsgen luxury workflow

Mode verification utilities:
- Parse an ORCA output, find imaginary modes
- Compare best imaginary mode to stored Fingerprint
- Emit a compact verify.json and return a structured result
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..orca_io import parse_frequencies
from ..fingerprint import Fingerprint, compare_modes


@dataclass
class VerifyResult:
    passed: bool
    n_imag: int
    imag_freqs: list[float]
    cosine: Optional[float]
    localization: Optional[float]
    out_path: Path
    notes: str = ""


def check_from_outfile(
    out_path: Path,
    fingerprint: Optional[Fingerprint],
    outdir: Optional[Path] = None,
) -> VerifyResult:
    """
    Verify TS character using an ORCA output file.
    """
    out_path = Path(out_path)
    freqs, modes = parse_frequencies(out_path)

    imag_freqs = [f for f in freqs if f < 0.0]
    n_imag = len(imag_freqs)

    cosine = localization = None
    passed = False
    notes = ""

    if n_imag == 0:
        notes = "No imaginary frequencies found."
    elif fingerprint is None:
        passed = (n_imag == 1)
        notes = "No fingerprint provided; pass if exactly one imaginary frequency."
    else:
        # compare_modes expects array of mode vectors (mass-weighted)
        # Select only imaginary modes for the comparison
        import numpy as np
        imag_indices = [i for i, f in enumerate(freqs) if f < 0.0]
        imag_modes = np.array([modes[i] for i in imag_indices])
        comp = compare_modes(imag_modes, fingerprint)
        cosine = comp.cosine
        localization = comp.localization
        passed = (n_imag == 1) and comp.passed
        if n_imag != 1:
            notes = f"{n_imag} imaginary modes; fingerprint match={comp.passed}."
        else:
            notes = f"Fingerprint match={comp.passed}."

    vr = VerifyResult(
        passed=passed,
        n_imag=n_imag,
        imag_freqs=imag_freqs,
        cosine=cosine,
        localization=localization,
        out_path=out_path,
        notes=notes,
    )

    # Write verify.json next to the output or to provided outdir
    target_dir = Path(outdir) if outdir else out_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "verify.json").write_text(_to_json(vr), encoding="utf-8")

    return vr


def check(
    job_dir_or_out: Path,
    fingerprint: Optional[Fingerprint],
    outdir: Optional[Path] = None,
) -> VerifyResult:
    """
    Convenience wrapper: accept either a job directory or a direct .out file.
    Resolution order inside a directory prefers common TS filenames.
    """
    p = Path(job_dir_or_out)
    if p.is_file() and p.suffix.lower() == ".out":
        return check_from_outfile(p, fingerprint, outdir=outdir)

    # Try to locate a likely ORCA output inside the directory
    candidates = [
        "optTS_restart.out",
        "optTS.out",
        "cand.out",
        "freq.out",
        "job.out",
    ]
    for name in candidates:
        q = p / name
        if q.exists():
            return check_from_outfile(q, fingerprint, outdir=outdir)

    # Fallback: first *.out in the directory (if any)
    outs = sorted(p.glob("*.out"))
    if outs:
        return check_from_outfile(outs[0], fingerprint, outdir=outdir)

    # Nothing found
    return VerifyResult(
        passed=False,
        n_imag=0,
        imag_freqs=[],
        cosine=None,
        localization=None,
        out_path=p,
        notes="No ORCA output file found.",
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_json(vr: VerifyResult) -> str:
    return json.dumps(
        {
            "passed": vr.passed,
            "n_imag": vr.n_imag,
            "imag_freqs_cm-1": vr.imag_freqs,
            "cosine": vr.cosine,
            "localization": vr.localization,
            "out_path": str(vr.out_path),
            "notes": vr.notes,
        },
        indent=2,
    )
