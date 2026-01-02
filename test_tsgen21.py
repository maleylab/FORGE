#!/usr/bin/env python3
"""
TSGen 2.1 | End-to-End Debug-Mode Test Harness

Runs the TSGen pipeline entirely in DEBUG mode:

    • No SLURM jobs
    • No ORCA
    • Fake XYZ/OUT files created by controller
    • L0/L1/L2/VERIFY/L3 exercised
    • Produces a summary dictionary

This is the harness you run BEFORE plugging in the worker system.
"""

from pathlib import Path
import shutil
import json

from labtools.tsgen.plan import TSGenPlan
from labtools.tsgen.controller import TSGenController


# =====================================================================
# Helpers: write tiny fake reactant + product geometries
# =====================================================================
def write_xyz(path: Path, atoms, coords):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("test geometry\n")
        for a, (x, y, z) in zip(atoms, coords):
            f.write(f"{a} {x:.6f} {y:.6f} {z:.6f}\n")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Clean test directory
    # ------------------------------------------------------------
    work = Path("tsgen_test_run").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    print(f"Running TSGen2.1 test in: {work}")

    # ------------------------------------------------------------
    # Create trivial test geometries (H2 → H2 slightly shifted)
    # ------------------------------------------------------------
    reactant_xyz = work / "reactant.xyz"
    product_xyz  = work / "product.xyz"

    atoms = ["H", "H"]
    rcoords = [(0.0, 0.0, 0.0),
               (0.7, 0.0, 0.0)]

    pcoords = [(0.0, 0.0, 0.05),
               (0.7, 0.0, -0.05)]

    write_xyz(reactant_xyz, atoms, rcoords)
    write_xyz(product_xyz, atoms, pcoords)

    # ------------------------------------------------------------
    # Build TSGen plan (debug mode → no fingerprint required)
    # ------------------------------------------------------------
    plan = TSGenPlan(
        reactant=reactant_xyz,
        product=product_xyz,
        work_dir=work,
        charge=0,
        mult=1,
        l0_method="XTB2",
        l1_method="r2SCAN-3c",
        l2_method="M06-2X/Def2-SVP",
        max_seeds=3,
        profile="test",
        fingerprint_file=None,     # allowed because debug mode
        do_l3=True,
        l3_method="M06-2X/Def2-TZVP"
    )

    # Inject debug flag (VERIFY stage will read plan.debug)
    plan.debug = True

    # ------------------------------------------------------------
    # Run controller
    # ------------------------------------------------------------
    controller = TSGenController(plan, debug=True)
    summary = controller.run()

    # Pretty print
    print("\nTSGEN SUMMARY:")
    print(json.dumps(summary, indent=4))

    print("\nTest completed.")

