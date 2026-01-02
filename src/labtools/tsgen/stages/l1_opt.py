"""
TSGen 2.1 | L1 TS Optimization Stage

Consumes XYZ geometries surviving L0, writes ORCA inputs,
creates job directories with READY flags, and returns
worker job-spec dicts for TSGenController.

This stage performs no ORCA execution.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

from ..tsgen_orca import write_orca_input


def _build_l1_geom_block(plan) -> List[str] | None:
    """
    Construct a %geom block for L1 if requested.

    Opt-in behavior:
        • Hybrid_Hess (PHVA) with user-specified atoms
        • Convergence loose
        • Calc_Hess true

    If no Hybrid_Hess atoms are specified, returns None
    and L1 behaves like a normal OptTS.
    """

    atoms = plan.l1_hybrid_hess_atoms
    if not atoms:
        return None

    geom_lines: List[str] = []

    # Loose convergence + PHVA
    geom_lines.append("Convergence loose")
    geom_lines.append("Calc_Hess true")

    atom_str = " ".join(str(i) for i in atoms)
    geom_lines.append("Hybrid_Hess")
    geom_lines.append(f"  {{{atom_str}}}")
    geom_lines.append("end")

    return geom_lines


def _append_post_opt_freq(
    inp_path: Path,
    plan,
):
    """
    Append a forced post-optimization frequency job.

    This is REQUIRED because ORCA may silently skip
    frequencies when PHVA / Hybrid_Hess is used.

    This does NOT spawn a new HPC job.
    """

    method = plan.l1_method
    parts = method.split("/")
    functional = parts[0]
    basis = parts[1] if len(parts) > 1 else None

    bang = f"! Freq {functional}"
    if basis:
        bang += f" {basis}"
    bang += " RIJCOSX TightSCF"

    freq_block = f"""

$new_job
{bang}

%pal
  nprocs {plan.nprocs}
end

* xyzfile {plan.charge} {plan.mult}
"""

    with inp_path.open("a") as f:
        f.write(freq_block)


def run_l1(plan, upstream_xyz: List[Path]) -> List[Dict[str, Any]]:
    """
    Generate L1 TSOpt jobs from upstream XYZ geometries.

    Returns a list of job-spec dictionaries consumed by TSGenController.
    """

    stage_dir = plan.stage_dir("L1")
    jobs: List[Dict[str, Any]] = []

    # Optional %geom block (PHVA / loose convergence)
    geom_block = _build_l1_geom_block(plan)

    # Optional raw keyword blocks
    extra_blocks = None
    if plan.stage_keywords and "L1" in plan.stage_keywords:
        extra_blocks = plan.stage_keywords["L1"]

    for xyz in upstream_xyz:
        job_name = xyz.stem
        jobdir = plan.job_dir("L1", job_name)

        # Copy upstream XYZ
        xyz_path = jobdir / f"{job_name}.xyz"
        xyz_path.write_text(xyz.read_text())

        inp_path = jobdir / f"{job_name}.inp"

        # -------------------------------
        # Primary TS optimization
        # -------------------------------
        write_orca_input(
            inp_path,
            jobtype="OptTS",
            method=plan.l1_method,
            charge=plan.charge,
            mult=plan.mult,
            geom_file=xyz_path,
            nprocs=plan.nprocs,
            maxcore=plan.maxcore,
            use_ri=True,
            geom_block=geom_block,
            extra_blocks=extra_blocks,
        )

        # -------------------------------
        # FORCED post-opt frequency
        # -------------------------------
        _append_post_opt_freq(inp_path, plan)

        # READY flag
        (jobdir / "READY").touch()

        jobs.append({
            "stage": "L1",
            "job_name": job_name,
            "jobdir": jobdir,
            "input": inp_path,
            "sbatch_template": None,
            "extra_sbatch_params": {
                "nprocs": plan.nprocs,
            },
        })

    return jobs
