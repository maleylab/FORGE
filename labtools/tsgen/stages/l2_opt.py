"""
L2: production-level TS optimization + frequency, restart handling, verification.
Default: M06/def2-SVP RIJCOSX def2/J def2/JK
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
    profile: str = "long",
    restart_on_fail: bool = True,
    mode: str = "array",
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    verified, reports = [], []

    for i, seed in enumerate(seeds):
        job_dir = outdir / f"TS_L2_{i:02d}"
        job_dir.mkdir(exist_ok=True)

        # Primary OptTS + Freq
        inp = job_dir / "optTS.inp"
        write_orca_input(
            path=inp, jobtype="OptTS Freq", method=method,
            charge=charge, mult=mult,
            geom_file=seed,
            use_ri=True, add_aux_basis=True,
            extra_blocks="%geom Recalc_Hess 5 Trust 0.30 end",
            provenance={"stage": "L2", "parent": str(seed)},
        )
        dispatch(inp, mode=mode, profile=profile)

        out_path = job_dir / "optTS.out"
        freqs, modes = parse_frequencies(out_path)
        success = _terminated_normally(out_path)

        # Restart path if needed
        if not success and restart_on_fail:
            _generate_restart(job_dir, method, charge, mult)
            dispatch(job_dir / "optTS_restart.inp", mode=mode, profile=profile)
            freqs, modes = parse_frequencies(job_dir / "optTS_restart.out")

        # Verification
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

    (outdir / "L2_report.json").write_text(json.dumps(reports, indent=2))
    return verified


def _terminated_normally(out_path: Path) -> bool:
    text = Path(out_path).read_text(errors="ignore")
    return "ORCA TERMINATED NORMALLY" in text


def _generate_restart(job_dir: Path, method: str, charge: int, mult: int):
    inp_restart = job_dir / "optTS_restart.inp"
    geom_file = job_dir / "optTS.xyz"
    write_orca_input(
        path=inp_restart,
        jobtype="OptTS",
        method=method,
        charge=charge, mult=mult,
        geom_file=geom_file,
        use_ri=True, add_aux_basis=True,
        extra_blocks=(
            "%geom\n"
            "  InHess Read\n"
            "  Hybrid_Hess true\n"
            "  Trust 0.40\n"
            "end"
        ),
        provenance={"stage": "L2_restart", "parent": str(geom_file)},
    )
