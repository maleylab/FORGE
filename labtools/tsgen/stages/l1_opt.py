"""
L1: r2SCAN-3c TS optimization and optional verification.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Sequence
from ..orca_io import write_orca_input, parse_frequencies
from ..fingerprint import Fingerprint, compare_modes
from labtools.submit import dispatch


def run_opt(
    seeds: Sequence[Path],
    method: str,
    outdir: Path,
    charge: int,
    mult: int,
    fingerprint: Optional[Fingerprint] = None,
    profile: str = "medium",
    mode: str = "array",
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    verified, reports = [], []

    for i, seed in enumerate(seeds):
        job_dir = outdir / f"TS_L1_{i:02d}"
        job_dir.mkdir(exist_ok=True)

        inp = job_dir / "optTS.inp"
        write_orca_input(
            path=inp, jobtype="OptTS", method=method,
            charge=charge, mult=mult,
            geom_file=seed,
            use_ri=False, add_aux_basis=False,  # 3c methods handle this internally
            provenance={"stage": "L1", "parent": str(seed)},
        )

        dispatch(inp, mode=mode, profile=profile)

        out_path = job_dir / "optTS.out"
        freqs, modes = parse_frequencies(out_path)

        if fingerprint:
            res = compare_modes(modes, fingerprint)
            rep = {
                "seed": str(seed),
                "cosine": res.cosine,
                "localization": res.localization,
                "passed": res.passed,
            }
            (job_dir / "verify.json").write_text(json.dumps(rep, indent=2))
            reports.append(rep)
            if res.passed:
                verified.append(job_dir / "optTS.xyz")
        else:
            verified.append(job_dir / "optTS.xyz")

    (outdir / "L1_report.json").write_text(json.dumps(reports, indent=2))
    return verified
