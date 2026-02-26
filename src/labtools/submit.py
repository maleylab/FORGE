"""
TSGen 2.1 | submit.py

Clean, minimal SLURM dispatcher for TSGen worker-mode execution.

TSGen always dispatches *single-job workers*, never arrays.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union

from labtools.slurm.render import render_template

Pathish = Union[str, Path]

# ---------------------------------------------------------------------------
# SLURM PROFILES
# ---------------------------------------------------------------------------
PROFILES: Dict[str, Dict[str, Any]] = {
    "test":   {"time": "00:10:00", "nprocs": 4,  "mem_per_cpu": "2G"},
    "short":  {"time": "02:00:00", "nprocs": 8,  "mem_per_cpu": "2G"},
    "medium": {"time": "24:00:00", "nprocs": 8,  "mem_per_cpu": "4G"},
    "long":   {"time": "72:00:00", "nprocs": 16, "mem_per_cpu": "4G"},
}

# ---------------------------------------------------------------------------
# TEMPLATE LOCATION
# ---------------------------------------------------------------------------
def _template_root() -> Path:
    return Path(__file__).resolve().parents[2] / "templates" / "sbatch"


# ---------------------------------------------------------------------------
# DISPATCH (TSGen uses ONLY mode="job")
# ---------------------------------------------------------------------------
def dispatch(
    input_path: Pathish,
    *,
    mode: str,
    profile: str = "medium",
    job_name: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    sbatch_template: Optional[str] = None,
    dry_run: bool = False,
    submit_cwd: Optional[Pathish] = None,
    validate_only: bool = False,
) -> str:
    """
    Universal job submission entry point.

    TSGen uses only:
        dispatch(path_to_inp, mode="job",
                 sbatch_template="tsgen_L0_worker.sbatch.j2")
    """

    if mode != "job":
        raise ValueError("TSGen dispatcher only supports mode='job'")

    inp = Path(input_path).resolve()
    jobdir = inp.parent
    cwd = Path(submit_cwd).resolve() if submit_cwd else Path.cwd()

    # ------------------------------
    # Profile defaults
    # ------------------------------
    prof = PROFILES.get(profile, PROFILES["medium"])
    params = {
        "job_name": job_name or jobdir.name,
        "jobdir": str(jobdir),
        "time": prof["time"],
        "nprocs": prof["nprocs"],
        "mem_per_cpu": prof["mem_per_cpu"],
    }

    if extra_params:
        params.update(extra_params)

    # ------------------------------
    # Select template (REQUIRED)
    # ------------------------------
    if not sbatch_template:
        raise ValueError("TSGen requires explicit sbatch_template")

    # normalize to string and enforce correct case/path usage
    tpl = _template_root() / str(sbatch_template)
    if not tpl.is_file():
        raise FileNotFoundError(f"Missing sbatch template: {tpl}")

    # ------------------------------
    # Write sbatch script
    # ------------------------------
    sbatch_script = cwd / f"{params['job_name']}.sbatch"

    if dry_run:
        preview = render_template(tpl, None, params, return_text=True)
        print("\n--- SBATCH DRY RUN ---\n")
        print(preview)
        return ""

    render_template(tpl, sbatch_script, params)

    # ------------------------------
    # Submit to SLURM
    # ------------------------------
    cmd = ["sbatch"]
    if validate_only:
        cmd.append("--test-only")
    cmd.append(str(sbatch_script))

    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())

    m = re.search(r"Submitted batch job (\d+)", proc.stdout)
    return m.group(1) if m else ""
