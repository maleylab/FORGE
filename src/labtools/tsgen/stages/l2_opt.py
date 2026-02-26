"""
TSGen 2.1 | L2 High-Level TS Optimization Stage

Consumes XYZ geometries surviving L1.  Produces high-level
TS optimization job directories for worker-mode execution.

No execution or filtering occurs here.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

from ..tsgen_orca import write_orca_input


def run_l2(plan, upstream_xyz: List[Path]) -> List[Dict[str, Any]]:
    """
    Generate L2 high-level TSOpt jobs from upstream geometries.

    Returns a list of job-spec dicts (consumed by TSGenController).
    """

    stage_dir = plan.stage_dir("L2")
    jobs: List[Dict[str, Any]] = []

    for xyz in upstream_xyz:
        job_name = xyz.stem
        jobdir = plan.job_dir("L2", job_name)

        # Write XYZ to jobdir
        xyz_path = jobdir / f"{job_name}.xyz"
        xyz_path.write_text(xyz.read_text())

        # --------------------------------------------------------
        # ORCA input — high-level TS optimization
        # --------------------------------------------------------
        inp_path = jobdir / f"{job_name}.inp"

        extra_blocks=None
        if plan.stage_keywords and "L2" in plan.stage_keywords:
            extra_blocks = "\n".join(plan.stage_keywords["L2"])


        write_orca_input(
            inp_path,
            jobtype="OptTS freq",
            method=plan.l2_method,     # e.g., M06-2X/Def2-SVP
            charge=plan.charge,
            mult=plan.mult,
            geom_file=xyz_path,
            nprocs=plan.nprocs,
            maxcore=plan.maxcore,
            use_ri=True,
            extra_blocks=extra_blocks,
        )

        # Worker flag
        (jobdir / "READY").touch()

        # --------------------------------------------------------
        # Register job
        # --------------------------------------------------------
        jobs.append({
            "stage": "L2",
            "job_name": job_name,
            "jobdir": jobdir,
            "input": inp_path,
            "sbatch_template": None,  # controller overrides
            "extra_sbatch_params": {
                "nprocs": plan.nprocs,
            },
        })

    return jobs
