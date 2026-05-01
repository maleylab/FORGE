"""labtools.submit

SLURM submission helpers.

This module is intentionally small and "dumb": it renders an sbatch script
from templates/sbatch and calls `sbatch`.

It is used both by the CLI (`forge submit`, `forge submit-array`) and by the
pipeline submit stage.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from labtools.slurm.render import render_template

Pathish = Union[str, Path]


# ---------------------------------------------------------------------------
# SLURM PROFILES
# ---------------------------------------------------------------------------

PROFILES: Dict[str, Dict[str, Any]] = {
    "test": {"time": "00:10:00", "nprocs": 4, "cpus_per_task": 4, "mem_per_cpu": "2G", "mem": "8G"},
    "short": {"time": "02:00:00", "nprocs": 8, "cpus_per_task": 8, "mem_per_cpu": "2G", "mem": "16G"},
    "medium": {"time": "24:00:00", "nprocs": 8, "cpus_per_task": 8, "mem_per_cpu": "4G", "mem": "32G"},
    "long": {"time": "72:00:00", "nprocs": 16, "cpus_per_task": 16, "mem_per_cpu": "4G", "mem": "64G"},
}


def _repo_root() -> Path:
    # /lab-tools/src/labtools/submit.py -> parents[2] == /lab-tools/src
    # parents[3] == /lab-tools
    return Path(__file__).resolve().parents[2]


def _sbatch_templates_dir() -> Path:
    return _repo_root() / "templates" / "sbatch"


def _resolve_template(name: Optional[str], *, mode: str) -> Path:
    """Return absolute path to an sbatch template."""

    if name:
        cand = _sbatch_templates_dir() / str(name)
        if not cand.is_file():
            raise FileNotFoundError(f"Missing sbatch template: {cand}")
        return cand

    # Defaults
    if mode == "job":
        cand = _sbatch_templates_dir() / "single_orca_job.sbatch.j2"
    elif mode == "array":
        cand = _sbatch_templates_dir() / "array_driver.sbatch.j2"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not cand.is_file():
        raise FileNotFoundError(f"Missing sbatch template: {cand}")
    return cand


def dispatch(
    input_path: Pathish | Sequence[Pathish],
    *,
    mode: str,
    profile: str = "medium",
    job_name: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    sbatch_template: Optional[str] = None,
    dry_run: bool = False,
    submit_cwd: Optional[Pathish] = None,
    sbatch_chdir: Optional[Pathish] = None,
    validate_only: bool = False,
) -> str:
    """Submit ORCA jobs via SLURM.

    Parameters
    ----------
    input_path
        For mode="job": path to a single .inp.
        For mode="array": list of paths to .inp files.
    sbatch_template
        Template filename under templates/sbatch. If None, defaults are used.
    submit_cwd
        Where to write the sbatch script and run `sbatch`.
    sbatch_chdir
        Passed to `sbatch --chdir=...` if provided.
    validate_only
        Uses `sbatch --test-only`.
    """

    mode = str(mode).strip().lower()
    prof = PROFILES.get(profile, PROFILES["medium"])

    tpl = _resolve_template(sbatch_template, mode=mode)

    if mode == "job":
        inp = Path(str(input_path)).expanduser().resolve()  # type: ignore[arg-type]
        if not inp.is_file():
            raise FileNotFoundError(f"Input not found: {inp}")

        jobdir = inp.parent
        cwd = Path(submit_cwd).expanduser().resolve() if submit_cwd else jobdir
        cwd.mkdir(parents=True, exist_ok=True)

        params: Dict[str, Any] = {
            "job_name": job_name or jobdir.name,
            "jobdir": str(jobdir),
            "time": prof["time"],
            "nprocs": prof["nprocs"],
            "cpus_per_task": prof.get("cpus_per_task", prof["nprocs"]),
            "mem_per_cpu": prof["mem_per_cpu"],
            "mem": prof.get("mem", None),

            # Parameters expected by single_orca_job.sbatch.j2
            "SRC_DIR": str(jobdir),
            "inp_basename": inp.name,
            "orca_cmd": '${ORCA_BIN} "${INP_BN}" > "${INP_BN%.inp}.out"',
            "out_basename": f"{inp.stem}.out",
        }
        if extra_params:
            params.update(extra_params)

        sbatch_script = cwd / "job.sbatch"

        if dry_run:
            preview = render_template(tpl, None, params, return_text=True)
            print(preview)
            return ""

        render_template(tpl, sbatch_script, params)

        cmd: List[str] = ["sbatch"]
        if sbatch_chdir:
            cmd.append(f"--chdir={Path(sbatch_chdir).expanduser().resolve()}")
        if validate_only:
            cmd.append("--test-only")
        cmd.append(str(sbatch_script))

        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())

        m = re.search(r"Submitted batch job (\d+)", proc.stdout)
        return m.group(1) if m else ""

    if mode == "array":
        paths = [Path(str(p)).expanduser().resolve() for p in (input_path if isinstance(input_path, (list, tuple)) else [input_path])]  # type: ignore[list-item]
        paths = [p for p in paths if p.is_file()]
        if not paths:
            raise FileNotFoundError("No input files provided for array")

        first_dir = paths[0].parent
        cwd = Path(submit_cwd).expanduser().resolve() if submit_cwd else first_dir
        cwd.mkdir(parents=True, exist_ok=True)

        params = {
            "job_name": job_name or "array",
            "time": prof["time"],
            "cpus_per_task": prof.get("cpus_per_task", prof["nprocs"]),
            "mem": prof.get("mem", "16G"),
            "account": "def-smaley",
            "inputs": [str(p) for p in paths],
        }
        if extra_params:
            params.update(extra_params)

        sbatch_script = cwd / "array.sbatch"
        if dry_run:
            preview = render_template(tpl, None, params, return_text=True)
            print(preview)
            return ""

        render_template(tpl, sbatch_script, params)

        cmd = ["sbatch"]
        if sbatch_chdir:
            cmd.append(f"--chdir={Path(sbatch_chdir).expanduser().resolve()}")
        if validate_only:
            cmd.append("--test-only")
        cmd.append(str(sbatch_script))

        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        m = re.search(r"Submitted batch job (\d+)", proc.stdout)
        return m.group(1) if m else ""

    raise ValueError(f"Unsupported mode: {mode}")

