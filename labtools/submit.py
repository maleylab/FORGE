# src/labtools/submit.py
from __future__ import annotations

import re
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Union

from labtools.slurm.render import render_template

Pathish = Union[str, Path]

# ----------------------------
# Profiles (queue resources)
# ----------------------------
PROFILES: Dict[str, Dict[str, Any]] = {
    "test": {"time": "00:10:00", "nprocs": 8,  "mem": "8G"},   # quick test
    "short":       {"time": "01:00:00", "nprocs": 4,  "mem": "8G"},
    "medium":      {"time": "12:00:00", "nprocs": 8,  "mem": "16G"},
    "long":        {"time": "72:00:00", "nprocs": 16, "mem": "32G"},
}

# Default whitelist of sidecar files to stage alongside the .inp
DEFAULT_RSYNC_GLOBS: List[str] = [
    "*.xyz", "*.gbw", "*.hess", "*.aux", "*.engrad", "*.molden.input", "*.pcmo"
]

# ----------------------------
# Template resolution helpers
# ----------------------------
def _sbatch_templates_root() -> Path:
    """
    ABSOLUTE location of SLURM templates.
    Hard-coded per user instruction.
    """
    return Path("/home/smaley/lab-tools/templates/sbatch")

def _pick_template(mode: str) -> Path:
    root = _sbatch_templates_root()
    if mode == "job":
        return root / "single_orca_job.sbatch.j2"
    if mode == "array":
        return root / "array_driver.sbatch.j2"
    if mode == "drone":
        return root / "drone_worker.sbatch.j2"
    raise ValueError(f"Unknown mode '{mode}'")

# ----------------------------
# ORCA command builder
# ----------------------------
def _build_orca_cmd(single_input: Path, nprocs: int) -> str:
    """
    Use EBROOTORCA explicitly and write output to STEM.out.
    Respect SLURM cpus-per-task for OMP threads.
    """
    inp_bn = single_input.name
    stem   = single_input.stem
    out_bn = f"{stem}.out"
    return (
        f'export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{nprocs}}}"; '
        f'ulimit -s unlimited; '
        f'${{EBROOTORCA}}/orca "{inp_bn}" > "{out_bn}"'
    )

# ----------------------------
# SBatch execution
# ----------------------------
_SBATCH_ID_RE = re.compile(r"Submitted batch job\s+(\d+)|^(\d+)$", re.IGNORECASE | re.MULTILINE)

def _run_sbatch(
    sbatch_path: Path,
    submit_cwd: Path,
    sbatch_chdir: Optional[Path],
    validate_only: bool
) -> str:
    args = ["sbatch"]
    if sbatch_chdir:
        args += ["--chdir", str(sbatch_chdir)]
    if validate_only:
        args += ["--test-only"]
    args.append(str(sbatch_path))

    proc = subprocess.run(
        args, cwd=str(submit_cwd),
        text=True, capture_output=True, check=False
    )
    stdout, stderr, rc = proc.stdout.strip(), proc.stderr.strip(), proc.returncode

    if rc != 0:
        raise RuntimeError(
            f"sbatch failed with exit code {rc}.\n====== sbatch STDOUT ======\n{stdout}\n"
            f"====== sbatch STDERR ======\n{stderr}\n"
        )

    m = _SBATCH_ID_RE.search(stdout) or _SBATCH_ID_RE.search(stderr)
    return (m.group(1) or m.group(2)) if m else ""

# ----------------------------
# Public dispatch API
# ----------------------------
def dispatch(
    inputs: Union[Pathish, Sequence[Pathish]],
    *,
    mode: str,                         # "job" | "array" | "drone"
    profile: str = "medium",
    job_name: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    submit_cwd: Optional[Pathish] = None,   # where to run the sbatch command (host-side)
    sbatch_chdir: Optional[Pathish] = None, # SLURM working directory inside the job allocation
    validate_only: bool = False
) -> str:
    """
    Render an sbatch script for the chosen `mode` using Jinja2 templates (from the hard-coded sbatch dir) and submit it.
    Returns the SLURM job id (empty string if not reported).
    """
    submit_cwd_path = Path(submit_cwd).expanduser().resolve() if submit_cwd else Path.cwd()
    sbatch_chdir_path = Path(sbatch_chdir).expanduser().resolve() if sbatch_chdir else None

    prof = dict(PROFILES.get(profile, PROFILES["medium"]))
    time_str = prof.get("time")
    nprocs   = int(prof.get("nprocs", 8))
    mem      = prof.get("mem", "16G")

    params: Dict[str, Any] = {
        "job_name": job_name or ("array" if mode == "array" else mode),
        "time": time_str,
        "nprocs": nprocs,
        "mem": mem,
    }
    if extra_params:
        params.update(extra_params)

    tpl_path = _pick_template(mode)

    if mode == "job":
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError("mode=job expects exactly one input file")
            single = Path(inputs[0]).expanduser().resolve()
        else:
            single = Path(inputs).expanduser().resolve()
        if not single.exists():
            raise FileNotFoundError(single)

        src_dir = single.parent
        params.setdefault("SRC_DIR", str(src_dir))
        params.setdefault("inp_basename", single.name)
        params.setdefault("RSYNC_GLOBS", list(DEFAULT_RSYNC_GLOBS))
        params.setdefault("ARCHIVE_NAME", f"{params['job_name']}_{int(time.time())}.tar.gz")
        params.setdefault("orca_cmd", _build_orca_cmd(single, nprocs))

    elif mode == "array":
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("mode=array expects a sequence of inputs")
        inps = [Path(p).expanduser().resolve() for p in inputs]
        inps = [p for p in inps if p.exists()]
        if not inps:
            raise ValueError("mode=array got an empty or invalid inputs list")

        params.setdefault("inputs", [str(p) for p in inps])
        params.setdefault("ARCHIVE_NAME", f"{params['job_name']}_{int(time.time())}.tar.gz")

        if sbatch_chdir_path is None:
            parents = {str(p.parent) for p in inps}
            sbatch_chdir_path = Path(os.path.commonpath(list(parents)))

    elif mode == "drone":
        params.setdefault("QUEUE_DIR", "forge_queue")
        params.setdefault("SLEEP_SECS", 60)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    slurm_dir = submit_cwd_path / ".slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    sbatch_file = slurm_dir / f"{params['job_name']}.{mode}.{ts}.sbatch"

    render_template(str(tpl_path), sbatch_file, params)

    job_id = _run_sbatch(
        sbatch_file,
        submit_cwd=submit_cwd_path,
        sbatch_chdir=sbatch_chdir_path,
        validate_only=validate_only
    )
    return job_id

# ----------------------------
# Optional: render-only helper for tests
# ----------------------------
def render_only(
    inputs: Union[Pathish, Sequence[Pathish]],
    *,
    mode: str,
    profile: str = "medium",
    job_name: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    sbatch_chdir: Optional[Pathish] = None,
    out_path: Optional[Pathish] = None,
) -> Path:
    """
    Render an sbatch script without submitting it. Returns the path to the .sbatch.
    Useful for debugging and unit tests.
    """
    submit_cwd_path = Path.cwd()
    sbatch_chdir_path = Path(sbatch_chdir).expanduser().resolve() if sbatch_chdir else None

    prof = dict(PROFILES.get(profile, PROFILES["medium"]))
    params: Dict[str, Any] = {
        "job_name": job_name or ("array" if mode == "array" else mode),
        "time": prof.get("time"),
        "nprocs": int(prof.get("nprocs", 8)),
        "mem": prof.get("mem", "16G"),
    }
    if extra_params:
        params.update(extra_params)

    tpl_path = _pick_template(mode)

    if mode == "job":
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError("mode=job expects exactly one input file")
            single = Path(inputs[0]).expanduser().resolve()
        else:
            single = Path(inputs).expanduser().resolve()
        if not single.exists():
            raise FileNotFoundError(single)

        params.setdefault("SRC_DIR", str(single.parent))
        params.setdefault("inp_basename", single.name)
        params.setdefault("RSYNC_GLOBS", list(DEFAULT_RSYNC_GLOBS))
        params.setdefault("ARCHIVE_NAME", f"{params['job_name']}_{int(time.time())}.tar.gz")
        params.setdefault("orca_cmd", _build_orca_cmd(single, int(params["nprocs"])))

    elif mode == "array":
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("mode=array expects a sequence of inputs")
        inps = [Path(p).expanduser().resolve() for p in inputs]
        inps = [p for p in inps if p.exists()]
        if not inps:
            raise ValueError("mode=array got an empty or invalid inputs list")
        params.setdefault("inputs", [str(p) for p in inps])
        params.setdefault("ARCHIVE_NAME", f"{params['job_name']}_{int(time.time())}.tar.gz")
        if sbatch_chdir_path is None:
            parents = {str(p.parent) for p in inps}
            sbatch_chdir_path = Path(os.path.commonpath(list(parents)))

    elif mode == "drone":
        params.setdefault("QUEUE_DIR", "forge_queue")
        params.setdefault("SLEEP_SECS", 60)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    slurm_dir = Path(out_path).expanduser().resolve().parent if out_path else (submit_cwd_path / ".slurm")
    slurm_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    sbatch_file = Path(out_path).expanduser().resolve() if out_path else slurm_dir / f"{params['job_name']}.{mode}.{ts}.sbatch"
    render_template(str(tpl_path), sbatch_file, params)
    return sbatch_file
