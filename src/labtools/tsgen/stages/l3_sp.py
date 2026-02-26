"""
TSGen 2.1 | L3 Single-Point Stage

Creates SP (single-point) ORCA jobs for verified TS geometries.

This stage:
    • Creates L3/<job_name>/ directories
    • Writes geometry (.xyz)
    • Writes ORCA SP input file
    • Writes READY flag so the worker can pick it up
    • Returns a list of job dicts for TSGenController

NO cluster logic.
NO template assignment (controller overrides).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

from ..tsgen_orca import write_orca_input


def run_l3(plan, xyz_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Parameters
    ----------
    plan : TSGenPlan
    xyz_paths : list of Paths to verified TS XYZ files

    Returns
    -------
    List[Dict[str, Any]] : job descriptors for controller submission
    """

    stage_dir = plan.stage_dir("L3")
    jobs: List[Dict[str, Any]] = []

    # L3 requires a method
    if not plan.l3_method:
        raise RuntimeError("L3 stage requested but plan.l3_method is not set.")

    for xyz_path in xyz_paths:

        # Keep the upstream job name (seed_000, TS_004, etc.)
        job_name = xyz_path.stem.replace(".verified", "")

        jobdir = plan.job_dir("L3", job_name)

        # -----------------------
        # Copy XYZ
        # -----------------------
        xyz_dest = jobdir / f"{job_name}.xyz"
        xyz_dest.write_text(xyz_path.read_text())

        # -----------------------
        # ORCA SP input
        # -----------------------
        inp = jobdir / f"{job_name}.inp"

        write_orca_input(
            inp,
            jobtype="SP",
            method=plan.l3_method,
            charge=plan.charge,
            mult=plan.mult,
            geom_file=xyz_dest,
            maxcore=plan.maxcore,
            nprocs=plan.nprocs,
            use_ri=True,         # allowed
        )

        # Worker-mode READY flag
        (jobdir / "READY").touch()

        # -----------------------
        # Job spec (controller patches template)
        # -----------------------
        jobs.append({
            "stage": "L3",
            "job_name": job_name,
            "jobdir": jobdir,
            "input": inp,
            "sbatch_template": None,      # controller overwrites
            "extra_sbatch_params": {
                "nprocs": plan.nprocs,
            },
        })

    return jobs
