# src/labtools/orca/queues.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Nested dict getter with dotted paths."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        if part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


# Status labels (exported for potential future use)
STATUS_OK_MIN = "OK_MIN"
STATUS_OK_TS = "OK_TS"
STATUS_OK_NOFREQ = "OK_NOFREQ"
STATUS_BAD_TS_MULTI_IMAG = "BAD_TS_MULTI_IMAG"
STATUS_FAIL_SCF = "FAIL_SCF"
STATUS_FAIL_OPT = "FAIL_OPT"
STATUS_UNKNOWN = "UNKNOWN"


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a single ORCA job record (as produced by collect_job_record).

    Returns a shallow dict with:
      - job_dir
      - job_name
      - status
      - scf_converged
      - opt_status
      - n_imag
    """
    parsed: Dict[str, Any] = rec.get("parsed") or {}

    job_dir = rec.get("dir") or rec.get("job_dir")
    job_name = rec.get("job_name") or (Path(job_dir).name if job_dir else None)

    scf_converged = _get(parsed, "scf.converged")
    opt_status = _get(parsed, "opt.status")
    n_imag_raw = _get(parsed, "freq.n_imag")
    n_imag = _coerce_int(n_imag_raw)

    # Default status
    status = STATUS_UNKNOWN

    # Converged geometry branches
    if opt_status == "CONVERGED":
        if n_imag is None:
            status = STATUS_OK_NOFREQ
        elif n_imag == 0:
            status = STATUS_OK_MIN
        elif n_imag == 1:
            status = STATUS_OK_TS
        elif n_imag > 1:
            status = STATUS_BAD_TS_MULTI_IMAG
        else:
            status = STATUS_UNKNOWN
    else:
        # Non-converged / failed branches
        if scf_converged is False:
            status = STATUS_FAIL_SCF
        else:
            # Opt didn't converge, or unknown/aborted
            fail_like = {None, "FAILED", "ABORTED", "MAXCYC_EXCEEDED", "NO_CONVERGENCE", "INTERRUPTED"}
            if opt_status in fail_like:
                status = STATUS_FAIL_OPT
            else:
                status = STATUS_UNKNOWN

    return {
        "job_dir": job_dir,
        "job_name": job_name,
        "status": status,
        "scf_converged": scf_converged,
        "opt_status": opt_status,
        "n_imag": n_imag,
    }


def load_and_classify(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file of job records (from orca-batch-parse) and classify each.
    Returns a list of classification dicts.
    """
    jsonl_path = jsonl_path.expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cls = classify_record(rec)
            if cls.get("job_dir"):
                rows.append(cls)
    return rows


def _write_list(path: Path, job_dirs: List[str]) -> None:
    """Write a text file with one job_dir per line."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in sorted(set(job_dirs)):
            f.write(str(d) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a CSV summary of classification rows."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still write an empty file with headers
        fieldnames = ["job_dir", "job_name", "status", "scf_converged", "opt_status", "n_imag"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_queues(
    jsonl_path: Path,
    imag_list: Optional[Path] = None,
    failed_list: Optional[Path] = None,
    out_csv: Optional[Path] = None,
) -> Tuple[int, int, int]:
    """
    Main entrypoint: classify jobs from JSONL and write queue files.

    Returns (n_rows, n_imag, n_failed).
    """
    rows = load_and_classify(jsonl_path)

    # Imag queue = converged TSs with exactly 1 imag
    imag_dirs: List[str] = [r["job_dir"] for r in rows if r.get("status") == STATUS_OK_TS and r.get("job_dir")]

    # Failed queue = clearly failed jobs or multi-imag TSs
    failed_statuses = {STATUS_FAIL_SCF, STATUS_FAIL_OPT, STATUS_BAD_TS_MULTI_IMAG}
    failed_dirs: List[str] = [
        r["job_dir"] for r in rows if r.get("status") in failed_statuses and r.get("job_dir")
    ]

    if imag_list is not None:
        _write_list(imag_list, imag_dirs)

    if failed_list is not None:
        _write_list(failed_list, failed_dirs)

    if out_csv is not None:
        _write_csv(out_csv, rows)

    return len(rows), len(set(imag_dirs)), len(set(failed_dirs))
