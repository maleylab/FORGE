# src/labtools/workflow/utils.py

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Optional


# labtools/workflow/utils.py

import os
import subprocess
from pathlib import Path

def run_orca(input_file: Path, nprocs: int = 1) -> int:
    """
    Run ORCA inside the directory containing the input file.
    Redirect ORCA stdout to output.out inside that stage directory.
    """

    input_file = Path(input_file)
    stage_dir = input_file.parent

    orca_root = os.environ.get("EBROOTORCA")
    if not orca_root:
        raise RuntimeError("EBROOTORCA is not set (module load orca/6.1?).")

    orca_bin = str(Path(orca_root) / "orca")

    # Output file inside stage directory
    out_path = stage_dir / "output.out"

    with open(out_path, "w") as fout:
        proc = subprocess.run(
            [orca_bin, input_file.name],
            cwd=stage_dir,
            env={"OMP_NUM_THREADS": str(nprocs), **os.environ},
            stdout=fout,
            stderr=subprocess.STDOUT,
        )

    return proc.returncode



def check_orca_available() -> bool:
    """
    Non-intrusive helper for diagnostics.
    Does not modify environment or raise errors by default.
    """
    orca_root = os.environ.get("EBROOTORCA")
    if not orca_root:
        return False
    exe = Path(orca_root) / "orca"
    return exe.exists()
