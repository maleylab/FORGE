"""
TSGen 2.1 | L0 Seed Generator

Generates IDPP-based seed structures, writes ORCA inputs,
and prepares L0 job directories with READY flags for worker scripts.

No ORCA execution occurs here.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from ..tsgen_orca import write_orca_input
from ..idpp import generate_idpp_seeds


def run_l0(plan) -> List[Dict[str, Any]]:
    """
    Create L0 seed jobs in:
        work_dir / L0 / seed_000, seed_001, ...

    Returns a list of job-spec dicts consumed by TSGenController.
    """

    stage_dir = plan.stage_dir("L0")

    # ------------------------------------------------------------
    # Enforce presence of L0 constraints (unless in debug mode)
    # ------------------------------------------------------------
    if not getattr(plan, "debug", False):
        if not plan.l0_constraints:
            raise ValueError(
                "L0 requires explicit geometry constraints "
                "(plan.l0_constraints is empty or None)."
            )

    # Prepare ORCA geometry constraint block, if provided
    extra_blocks = None
    if plan.l0_constraints:
        extra_blocks = "\n".join(plan.l0_constraints)

    # ------------------------------------------------------------
    # Extract atom lists and coordinates
    # ------------------------------------------------------------
    atoms_r, coords_r = plan.reactant_xyz
    atoms_p, coords_p = plan.product_xyz

    # ------------------------------------------------------------
    # Generate IDPP seeds
    # (generate_idpp_seeds expects: R0, R1, n)
    # ------------------------------------------------------------
    n_seeds = int(plan.max_seeds)
    seeds = generate_idpp_seeds(coords_r, coords_p, n_seeds)

    jobs: List[Dict[str, Any]] = []

    for i, coords in enumerate(seeds):
        job_name = f"seed_{i:03d}"
        jobdir = plan.job_dir("L0", job_name)

        # --------------------------------------------------------
        # Write seed XYZ
        # --------------------------------------------------------
        xyz_path = jobdir / f"{job_name}.xyz"
        with xyz_path.open("w") as f:
            f.write(f"{len(atoms_r)}\nL0 seed\n")
            for a, (x, y, z) in zip(atoms_r, coords):
                f.write(f"{a} {x:.8f} {y:.8f} {z:.8f}\n")

        # --------------------------------------------------------
        # ORCA input: L0 = constrained Opt+Freq, PAL=1
        # --------------------------------------------------------
        inp_path = jobdir / f"{job_name}.inp"
        write_orca_input(
            inp_path,
            jobtype="Opt Freq",
            method=plan.l0_method,
            charge=plan.charge,
            mult=plan.mult,
            geom_file=xyz_path,
            nprocs=1,             # L0 = PAL1
            maxcore=plan.maxcore,
            use_ri=False,         # XTB / DFTB -> no RI
            extra_blocks=extra_blocks,
        )

        # READY flag for worker-mode
        (jobdir / "READY").touch()

        # --------------------------------------------------------
        # Register job spec for controller
        # --------------------------------------------------------
        jobs.append({
            "stage": "L0",
            "job_name": job_name,
            "jobdir": jobdir,
            "input": inp_path,
            "sbatch_template": None,   # controller overwrites
            "extra_sbatch_params": {
                "nprocs": 1,
            },
        })

    return jobs
