from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

from labtools.pipeline.slurm_status import detect_frontier


def _run_advance(run_dir: Path) -> int:
    # Import lazily to avoid CLI blast radius.
    import subprocess
    cp = subprocess.run(
        ["labtools", "pipeline", "advance", "--run-dir", str(run_dir)],
        text=True,
        check=False,
    )
    return int(cp.returncode)


def drive_pipeline(
    run_dir: Path,
    *,
    poll_seconds: int = 60,
    max_cycles: int | None = None,
    stop_on_fail: bool = True,
    verbose: bool = True,
) -> int:
    run_dir = run_dir.expanduser().resolve()
    cycles = 0

    while True:
        frontier: Dict[str, Any] = detect_frontier(run_dir)
        action = str(frontier.get("action") or "none").strip().lower()
        reason = str(frontier.get("reason") or "")

        if verbose:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] frontier={action} :: {reason}")
            sys.stdout.flush()

        if action == "finish":
            return 0

        if action == "none":
            return 1

        if action == "wait":
            time.sleep(max(5, int(poll_seconds)))
        elif action in {"submit", "collect", "materialize"}:
            rc = _run_advance(run_dir)
            if rc != 0 and stop_on_fail:
                return rc
            # small reconciliation delay
            time.sleep(2)
        else:
            # Unknown / unsafe frontier
            if stop_on_fail:
                return 2
            time.sleep(max(5, int(poll_seconds)))

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Hands-off pipeline driver")
    ap.add_argument("--run-dir", required=True, help="Pipeline run directory")
    ap.add_argument("--poll", type=int, default=60, help="Polling interval in seconds")
    ap.add_argument("--max-cycles", type=int, default=None, help="Optional safety bound")
    ap.add_argument("--keep-going", action="store_true", help="Do not stop on unexpected frontier/advance failure")
    args = ap.parse_args()

    return drive_pipeline(
        Path(args.run_dir),
        poll_seconds=int(args.poll),
        max_cycles=args.max_cycles,
        stop_on_fail=not args.keep_going,
    )


if __name__ == "__main__":
    raise SystemExit(main())

