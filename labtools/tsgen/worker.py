# src/labtools/tsgen/worker.py
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .collect_imag import parse_imag_from_orca_output

# ------------------------------------------------------------------------------
# Simple atomic claim helpers
# ------------------------------------------------------------------------------

def try_claim(seed_dir: Path) -> Optional[Path]:
    """
    Attempt to claim a seed by creating an atomic lock directory.
    Returns path to lock dir on success, else None.
    """
    lock = seed_dir / ".tsgen_lock"
    try:
        lock.mkdir(exist_ok=False)
        return lock
    except FileExistsError:
        return None


def release_claim(lock_dir: Path) -> None:
    try:
        lock_dir.rmdir()
    except Exception:
        pass


def seed_needs_work(seed_dir: Path) -> bool:
    """
    A seed is runnable iff it has an input, has not completed/failed,
    and there isn't a running marker (quick visibility).
    """
    if not (seed_dir / "phase_01.inp").exists():
        return False
    st = seed_dir / "status.txt"
    if st.exists():
        val = st.read_text().strip().lower()
        # treat any explicit state as non-runnable; the lock prevents collisions anyway
        if val in ("ok", "fail", "running"):
            return False
    # If an output exists already, consider it done
    if (seed_dir / "phase_01.out").exists():
        return False
    return True


def discover_seeds(seeds_dir: Path) -> List[Path]:
    return sorted([p for p in seeds_dir.glob("seed_*") if p.is_dir()])


def has_runnable_seeds(seeds_dir: Path) -> bool:
    return any(seed_needs_work(sd) for sd in discover_seeds(seeds_dir))


# ------------------------------------------------------------------------------
# Lightweight state writers (human- and machine-readable)
# ------------------------------------------------------------------------------

def write_state(seed_dir: Path, state: str, **meta) -> None:
    """Write a lightweight state marker + rich JSON for debugging."""
    (seed_dir / "status.txt").write_text(state)
    payload = {
        "state": state,
        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "host": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "pid": os.getpid(),
    }
    if meta:
        payload.update(meta)
    (seed_dir / "status.json").write_text(json.dumps(payload, indent=2))


# ------------------------------------------------------------------------------
# Global “current task” tracker for safe stage-back on signals
# ------------------------------------------------------------------------------

_CURRENT = {
    "seed_dir": None,     # Path
    "work_dir": None,     # Path
    "out_name": "phase_01.out",
}

def stageback_partial():
    """Best-effort copy of current workdir to seed dir on termination."""
    sd = _CURRENT["seed_dir"]
    wd = _CURRENT["work_dir"]
    out = _CURRENT["out_name"]
    if not sd or not wd:
        return
    sd = Path(sd)
    wd = Path(wd)
    try:
        # copy .out if present
        p_out = wd / out
        if p_out.exists():
            shutil.copy2(p_out, sd / out)
        # tar everything else for forensics
        import tarfile
        tarpath = sd / "other_artifacts.partial.tar.gz"
        with tarfile.open(tarpath, "w:gz") as tar:
            for p in wd.iterdir():
                if p.name == out or p.suffix in {".out", ".err"}:
                    continue
                tar.add(p, arcname=p.name)
    except Exception:
        pass


def _on_signal(signum, frame):
    # best effort stage-back, then mark fail and exit with 128+signal
    stageback_partial()
    try:
        sd = _CURRENT["seed_dir"]
        if sd:
            write_state(Path(sd), "fail", reason=f"killed_by_signal_{signum}")
    except Exception:
        pass
    sys.exit(128 + signum)


# Register handlers once at import
signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


# ------------------------------------------------------------------------------
# ORCA execution in node-local scratch
# ------------------------------------------------------------------------------

def run_orca_in_scratch(seed_dir: Path, work_root: Path, pal: int, maxcore: int) -> int:
    """
    Copy input to scratch, run ORCA, copy back output + artifacts.
    Returns ORCA's exit code (0 = success).
    """
    inp_name = "phase_01.inp"
    out_name = "phase_01.out"
    work = work_root / seed_dir.name
    work.mkdir(parents=True, exist_ok=True)

    # Stage-in
    shutil.copy2(seed_dir / inp_name, work / inp_name)

    # Resolve ORCA binary
    orca_bin = os.environ.get("LABTOOLS_ORCA_BIN") or shutil.which("orca")
    if not orca_bin:
        raise RuntimeError(
            "Could not locate ORCA binary. Ensure the job loads orca/6.1.0 "
            "and/or export LABTOOLS_ORCA_BIN in the sbatch."
        )

    # Runtime env: OpenMP-only policy; local scratch; bigger stack
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(pal))
    env.setdefault("MKL_NUM_THREADS", str(pal))
    env.setdefault("OPENBLAS_NUM_THREADS", str(pal))
    env.setdefault("OMP_STACKSIZE", "1G")
    env.setdefault("ORCA_TMPDIR", str(work))
    env.setdefault("ORCA_SCRDIR", str(work))
    env.setdefault("TMPDIR", str(work))
    # Force OpenMP backend; avoid MPI slots logic
    env.setdefault("ORCA_USE_MPI", "0")
    env.setdefault("ORCA_USE_OPENMP", "1")
    # Some OpenMPI wrappers still whine about slots: allow oversubscribe
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")

    # Run ORCA in scratch; capture stdout/stderr
    with (work / out_name).open("w") as fh:
        rc = subprocess.call(
            [orca_bin, inp_name],
            cwd=str(work),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )

    # Stage-back: keep main .out; tar all other artifacts
    try:
        if (work / out_name).exists():
            shutil.copy2(work / out_name, seed_dir / out_name)
        import tarfile
        tarpath = seed_dir / "other_artifacts.tar.gz"
        with tarfile.open(tarpath, "w:gz") as tar:
            for p in work.iterdir():
                # keep primary log separate; skip other *.out/*.err logs
                if p.name == out_name or p.suffix in {".out", ".err"}:
                    continue
                tar.add(p, arcname=p.name)
    except Exception:
        # best-effort packaging; don't fail the job just for tar/copy issues
        pass

    return rc


# ------------------------------------------------------------------------------
# One-unit-of-work execution
# ------------------------------------------------------------------------------

def worker_once(seeds_dir: Path, work_root: Path, pal: int, maxcore: int) -> bool:
    """
    Try to process one seed. Returns True if something was done.
    """
    for sd in discover_seeds(seeds_dir):
        if not seed_needs_work(sd):
            continue
        lock = try_claim(sd)
        if lock is None:
            continue  # someone else claimed it

        work = work_root / sd.name
        try:
            # Mark running immediately for visibility
            write_state(sd, "running", workdir=str(work))

            # Track current task for signal-safe stage-back
            _CURRENT["seed_dir"] = sd
            _CURRENT["work_dir"] = work

            rc = run_orca_in_scratch(sd, work_root, pal, maxcore)

            # Decide status
            ok = False
            reason = ""
            outp = sd / "phase_01.out"
            if outp.exists():
                txt = outp.read_text(errors="ignore")
                if "ORCA TERMINATED NORMALLY" in txt:
                    ok = True
                else:
                    m = re.search(r"ERROR[^\n]*", txt, re.IGNORECASE)
                    reason = m.group(0) if m else "termination"
            else:
                reason = "no_output"

            if ok:
                imag = parse_imag_from_orca_output(outp)
                (sd / "imag.json").write_text(json.dumps({"imag_cm-1": imag, "n_imag": len(imag)}))
                write_state(sd, "ok", n_imag=len(imag), exit_code=rc)
            else:
                write_state(sd, "fail", reason=reason, exit_code=rc)

        finally:
            # Clear current and release claim
            _CURRENT["seed_dir"] = None
            _CURRENT["work_dir"] = None
            release_claim(lock)

        return True  # processed one
    return False  # no work found


# ------------------------------------------------------------------------------
# Main worker loop
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("labtools.tsgen.worker")
    ap.add_argument("--seeds-dir", required=True, type=Path)
    ap.add_argument("--work-root", required=True, type=Path)
    ap.add_argument("--pal", type=int, default=8)
    ap.add_argument("--maxcore", type=int, default=3000)
    ap.add_argument("--max-tasks", type=int, default=1000, help="max seeds this worker processes")
    ap.add_argument("--stop-after-min", type=int, default=120, help="stop this many minutes before wall time (if detectable)")
    ap.add_argument("--cooldown-sec", type=int, default=5, help="sleep when idle")
    args = ap.parse_args()

    tasks = 0
    idle_loops = 0
    while tasks < args.max_tasks:
        did = worker_once(args.seeds_dir, args.work_root, args.pal, args.maxcore)
        if did:
            tasks += 1
            idle_loops = 0
            continue

        # No work processed this iteration
        idle_loops += 1

        # If there is definitively no remaining runnable seed, exit cleanly
        if not has_runnable_seeds(args.seeds_dir):
            # require two consecutive idle checks to avoid race with another writer
            if idle_loops >= 2:
                print("[worker] no runnable seeds; exiting")
                break

        # Otherwise backoff a bit and try again
        time.sleep(args.cooldown_sec + random.random())


if __name__ == "__main__":
    main()

