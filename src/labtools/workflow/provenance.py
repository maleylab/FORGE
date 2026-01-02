from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

PROV_DIRNAME = ".forge"
PROV_FILENAME = "provenance.jsonl"


def _prov_dir(jobdir: Path) -> Path:
    p = Path(jobdir) / PROV_DIRNAME
    p.mkdir(exist_ok=True)
    return p


def _prov_file(jobdir: Path) -> Path:
    return _prov_dir(jobdir) / PROV_FILENAME


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _append_jsonl_atomic(path: Path, record: Dict[str, Any]) -> None:
    line = json.dumps(record, separators=(",", ":"))
    tmp = path.with_suffix(".tmp")

    # First write to temp file
    with tmp.open("w") as f:
        f.write(line + "\n")

    # If file does not exist → atomic replace
    if not path.exists():
        tmp.replace(path)
        return

    # Else append
    with path.open("a") as f:
        f.write(line + "\n")

    # Remove temp
    try:
        tmp.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def write_stage_provenance(
    jobdir: Path,
    event: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a stage event:
      {
        "time": "...",
        "event": "opt_start",
        "stage": "opt",
        "extra": {...}
      }
    """
    stage = event.split("_")[0] if "_" in event else event
    record = {
        "time": _timestamp(),
        "event": event,
        "stage": stage,
    }
    if extra:
        record["extra"] = extra

    _append_jsonl_atomic(_prov_file(jobdir), record)


def write_stage_snapshot(
    jobdir: Path,
    stage: str,
    state: Dict[str, Any],
    label: str = "snapshot",
) -> None:
    """
    Store a structured snapshot:
      {
        "time": "...",
        "event": "snapshot",
        "stage": "opt",
        "snapshot": {...}
      }
    """
    record = {
        "time": _timestamp(),
        "event": label,
        "stage": stage,
        "snapshot": state,
    }
    _append_jsonl_atomic(_prov_file(jobdir), record)


def write_restart_event(
    jobdir: Path,
    stage: str,
    attempt: int,
    fail_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record restart metadata:
      {
        "time": "...",
        "event": "restart",
        "stage": "opt",
        "attempt": 2,
        "fail_type": "SCF_CONV",
        "extra": {...}
      }
    """
    record = {
        "time": _timestamp(),
        "event": "restart",
        "stage": stage,
        "attempt": attempt,
        "fail_type": fail_type,
    }
    if extra:
        record["extra"] = extra

    _append_jsonl_atomic(_prov_file(jobdir), record)


def read_provenance(jobdir: Path) -> List[Dict[str, Any]]:
    path = _prov_file(jobdir)
    if not path.exists():
        return []

    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out
