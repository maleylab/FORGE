"""
L3: DLPNO-CCSD(T)/def2-TZVPP single-point energies on verified L2 TS geometries.
"""

from __future__ import annotations
import json
import re
import csv
from pathlib import Path
from typing import Sequence
from ..orca_io import write_orca_input
from labtools.submit import dispatch


def run_sp(
    geometries: Sequence[Path],
    method: str,
    outdir: Path,
    charge: int,
    mult: int,
    profile: str = "long",
    mode: str = "array",
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs, results = [], []

    for i, geom in enumerate(geometries):
        job_dir = outdir / f"SP_L3_{i:02d}"
        job_dir.mkdir(exist_ok=True)
        inp = job_dir / "sp.inp"

        write_orca_input(
            path=inp, jobtype="SP", method=method,
            charge=charge, mult=mult,
            geom_file=geom,
            use_ri=True, add_aux_basis=True,
            provenance={"stage": "L3", "parent": str(geom)},
            extra_blocks="%mdci  TightPNO true  TCutPairs 1e-5  TCutPNO 1e-7  end",
        )

        dispatch(inp, mode=mode, profile=profile)
        out_path = job_dir / "sp.out"
        outputs.append(out_path)

        energy = _extract_energy(out_path)
        results.append({"geometry": str(geom), "energy_Ha": energy})

    (outdir / "L3_energies.json").write_text(json.dumps(results, indent=2))
    _write_csv(results, outdir / "L3_energies.csv")
    return outputs


def _extract_energy(out_path: Path) -> float | None:
    if not out_path.exists():
        return None
    text = Path(out_path).read_text(errors="ignore")
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def _write_csv(rows: list[dict], path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["geometry", "energy_Ha"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
