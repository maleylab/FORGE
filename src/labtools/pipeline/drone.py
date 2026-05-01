from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from labtools.slurm.render import render_template
from labtools.submit import PROFILES


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _template_path(template_name: str) -> Path:
    p = Path(str(template_name)).expanduser()
    if p.is_absolute() and p.is_file():
        return p.resolve()
    cand = _repo_root() / "templates" / "sbatch" / p.name
    if not cand.is_file():
        raise FileNotFoundError(f"Missing drone sbatch template: {cand}")
    return cand.resolve()


def _copy_job_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        # Never copy state sentinels from a prior execution.
        if item.name in {"READY", "STARTED", "DONE", "FAIL"}:
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)


def enqueue_render_batch(*, jobs_outdir: Path, job_dirs: List[str], queue_dir: Path) -> Dict[str, Any]:
    """Copy rendered job directories into a drone queue and mark them READY.

    Queue contract matches templates/sbatch/drone_worker.sbatch.j2:
      queue_dir/<job>/READY   -> available
      queue_dir/<job>/STARTED -> claimed/running
      queue_dir/<job>/DONE    -> completed
      queue_dir/<job>/FAIL    -> failed
    """
    jobs_outdir = jobs_outdir.expanduser().resolve()
    queue_dir = queue_dir.expanduser().resolve()
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "logs").mkdir(parents=True, exist_ok=True)

    enqueued: List[str] = []
    skipped_existing: List[str] = []

    for jd in job_dirs:
        src = jobs_outdir / jd
        if not src.is_dir():
            raise FileNotFoundError(f"Rendered job directory not found: {src}")
        dst = queue_dir / jd

        terminal_or_active = any((dst / s).exists() for s in ("STARTED", "DONE", "FAIL"))
        if terminal_or_active:
            skipped_existing.append(jd)
            continue

        _copy_job_dir(src, dst)
        # Reset to a clean READY state.
        for s in ("STARTED", "DONE", "FAIL"):
            try:
                (dst / s).unlink()
            except FileNotFoundError:
                pass
        (dst / "READY").touch()
        enqueued.append(jd)

    return {"queue_dir": str(queue_dir), "enqueued": enqueued, "skipped_existing": skipped_existing}


def submit_drone_workers(
    *,
    queue_dir: Path,
    n_drones: int,
    profile: str = "medium",
    sbatch_template: str = "drone_worker.sbatch.j2",
    dry_run: bool = False,
    validate_only: bool = False,
    job_name: str = "forge-drone",
    sleep_secs: int = 30,
    max_idle_cycles: int = 10,
    safety_margin_secs: int = 900,
    jitter_max_secs: int = 60,
    idle_jitter_max_secs: int = 10,
    heartbeat_secs: int = 60,
    tail_lines: int = 200,
) -> Dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve()
    queue_dir.mkdir(parents=True, exist_ok=True)
    logs = queue_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    prof = PROFILES.get(profile, PROFILES["medium"])
    tpl = _template_path(sbatch_template)
    n_drones = max(1, int(n_drones))

    drone_job_ids: List[str] = []
    sbatch_scripts: List[str] = []

    for i in range(1, n_drones + 1):
        params: Dict[str, Any] = {
            "job_name": f"{job_name}-{i:03d}" if n_drones > 1 else job_name,
            "QUEUE_DIR": str(queue_dir),
            "SLEEP_SECS": int(sleep_secs),
            "MAX_IDLE_CYCLES": int(max_idle_cycles),
            "SAFETY_MARGIN_SECS": int(safety_margin_secs),
            "JITTER_MAX_SECS": int(jitter_max_secs),
            "IDLE_JITTER_MAX_SECS": int(idle_jitter_max_secs),
            "HEARTBEAT_SECS": int(heartbeat_secs),
            "TAIL_LINES": int(tail_lines),
            "time": prof["time"],
            "nprocs": prof["nprocs"],
            "mem_per_cpu": prof["mem_per_cpu"],
            "mem": prof.get("mem"),
        }
        script = logs / f"drone_{i:03d}.sbatch"
        sbatch_scripts.append(str(script))

        if dry_run:
            print(render_template(tpl, None, params, return_text=True))
            drone_job_ids.append("")
            continue

        render_template(tpl, script, params)
        cmd = ["sbatch"]
        if validate_only:
            cmd.append("--test-only")
        cmd.append(str(script))
        proc = subprocess.run(cmd, cwd=str(queue_dir), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        m = re.search(r"Submitted batch job (\d+)", proc.stdout)
        drone_job_ids.append(m.group(1) if m else "")

    return {"drone_job_ids": drone_job_ids, "sbatch_scripts": sbatch_scripts}


def drone_job_state(job_dir: Path) -> str:
    """Return pipeline-normalized state from drone sentinels."""
    job_dir = job_dir.expanduser().resolve()
    if (job_dir / "FAIL").exists():
        return "FAILED"
    if (job_dir / "DONE").exists():
        return "COMPLETED"
    if (job_dir / "STARTED").exists():
        return "RUNNING"
    if (job_dir / "READY").exists():
        return "PENDING"
    return "UNKNOWN"
