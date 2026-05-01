from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _latest_artifact_manifest(run_dir: Path, kind: str) -> Optional[Path]:
    kind_dir = run_dir / "artifacts" / kind
    if not kind_dir.is_dir():
        return None
    manifests = sorted(kind_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None


def load_artifact_info(run_dir: Path, kind: str) -> Optional[Dict[str, Any]]:
    man = _latest_artifact_manifest(run_dir, kind)
    if man is None:
        return None
    try:
        obj = json.loads(man.read_text(encoding="utf-8"))
    except Exception:
        return None
    data = obj.get("data")
    if not isinstance(data, dict):
        data = {}
    return {
        "artifact_id": str((obj.get("artifact") or {}).get("id") or man.parent.name),
        "kind": kind,
        "manifest": man,
        "artifact_dir": man.parent,
        "data": data,
        "raw": obj,
    }


def load_artifact_data(run_dir: Path, kind: str) -> Optional[Dict[str, Any]]:
    info = load_artifact_info(run_dir, kind)
    return None if info is None else dict(info.get("data") or {})


def _load_collect_records(run_dir: Path, collect_info: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not collect_info:
        return []
    rel = str((collect_info.get("data") or {}).get("records_jsonl") or "")
    if not rel:
        return []
    path = Path(collect_info["artifact_dir"]) / rel
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def summarize_audit(audit_path: Path) -> Dict[str, Dict[str, Any]]:
    stages: Dict[str, Dict[str, Any]] = {}
    if not audit_path.exists():
        return stages

    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            ev = str(obj.get("event") or "")
            s = str(obj.get("stage") or "")
            if not s:
                continue

            st = stages.setdefault(s, {})
            if ev == "stage_start":
                st["start"] = obj.get("ts", "")
            elif ev == "stage_end":
                st["end"] = obj.get("ts", "")
                st["ok"] = obj.get("ok")
                st["message"] = obj.get("message", "")
            elif ev == "stage_skip":
                st["skip"] = True
                st["skip_reason"] = obj.get("reason", "")
                st["skip_ts"] = obj.get("ts", "")

    return stages


def normalize_slurm_state(raw_state: str) -> str:
    s = str(raw_state or "").strip().upper()
    if not s:
        return "UNKNOWN"
    if s.startswith("CANCELLED"):
        return "CANCELLED"
    if s.startswith("TIMEOUT"):
        return "TIMEOUT"
    if s.startswith("NODE_FAIL") or s.startswith("BOOT_FAIL"):
        return "NODE_FAIL"
    if s.startswith("OUT_OF_MEMORY") or s.startswith("OOM"):
        return "OUT_OF_MEMORY"
    if s.startswith("COMPLETED"):
        return "COMPLETED"
    if s.startswith("FAILED"):
        return "FAILED"
    if s.startswith("PENDING") or s.startswith("CONFIGURING"):
        return "PENDING"
    if s.startswith("RUNNING") or s.startswith("COMPLETING") or s.startswith("SUSPENDED"):
        return "RUNNING"
    if s.startswith("PREEMPTED"):
        return "FAILED"
    return s


def _run_cmd(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def _chunked(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def query_squeue(job_ids: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    ids = [str(j) for j in job_ids if str(j).strip()]
    if not ids:
        return out

    for chunk in _chunked(ids, 200):
        cp = _run_cmd(["squeue", "-h", "-o", "%i|%T", "-j", ",".join(chunk)])
        if cp.returncode != 0:
            continue
        for line in cp.stdout.splitlines():
            parts = [p.strip() for p in line.split("|", 1)]
            if len(parts) != 2:
                continue
            jid, state = parts
            if jid:
                out[jid] = normalize_slurm_state(state)
    return out


def query_sacct(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    ids = [str(j) for j in job_ids if str(j).strip()]
    if not ids:
        return out

    for chunk in _chunked(ids, 200):
        cp = _run_cmd(["sacct", "-n", "-P", "-o", "JobIDRaw,State,ExitCode", "-j", ",".join(chunk)])
        if cp.returncode != 0:
            continue
        for line in cp.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            jid_raw, state, exit_code = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not jid_raw:
                continue
            jid = jid_raw.split(".", 1)[0]
            rec = {
                "state": normalize_slurm_state(state),
                "raw_state": state,
                "exit_code": exit_code,
            }
            cur = out.get(jid)
            if cur is None or jid_raw == jid:
                out[jid] = rec
    return out


def gather_submit_status(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    submit = load_artifact_data(run_dir, "SubmitBatch") or {}
    job_ids_map = dict(submit.get("job_ids") or {})
    job_dirs = list(submit.get("job_dirs") or [])
    backend = str(submit.get("backend") or "slurm").strip().lower()

    dry_run = bool(submit.get("dry_run", False))
    validate_only = bool(submit.get("validate_only", False))

    per_job: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    if backend == "drone":
        from labtools.pipeline.drone import drone_job_state

        queue_dir = Path(str(submit.get("queue_dir") or submit.get("jobs_outdir") or "")).expanduser().resolve()
        for jd in job_dirs:
            state = drone_job_state(queue_dir / jd)
            counts[state] = counts.get(state, 0) + 1
            per_job.append(
                {
                    "job_dir": jd,
                    "job_id": "",
                    "state": state,
                    "source": "drone_queue",
                    "reason": "sentinel",
                    "exit_code": "",
                }
            )
        return {"submit": submit, "per_job": per_job, "counts": counts}

    real_job_ids = [str(job_ids_map.get(jd) or "").strip() for jd in job_dirs]
    real_job_ids = [j for j in real_job_ids if j]

    squeue_by_id = query_squeue(real_job_ids)
    missing_for_sacct = [j for j in real_job_ids if j not in squeue_by_id]
    sacct_by_id = query_sacct(missing_for_sacct)

    for jd in job_dirs:
        job_id = str(job_ids_map.get(jd) or "").strip()
        state = "UNKNOWN"
        source = ""
        reason = ""
        exit_code = ""

        if dry_run:
            state = "DRY_RUN"
            source = "local"
        elif validate_only and not job_id:
            state = "VALIDATED"
            source = "local"
        elif not job_id:
            state = "UNSUBMITTED"
            source = "local"
        elif job_id in squeue_by_id:
            state = squeue_by_id[job_id]
            source = "squeue"
        elif job_id in sacct_by_id:
            rec = sacct_by_id[job_id]
            state = str(rec.get("state") or "UNKNOWN")
            reason = str(rec.get("raw_state") or "")
            exit_code = str(rec.get("exit_code") or "")
            source = "sacct"

        counts[state] = counts.get(state, 0) + 1
        per_job.append(
            {
                "job_dir": jd,
                "job_id": job_id,
                "state": state,
                "source": source,
                "reason": reason,
                "exit_code": exit_code,
            }
        )

    return {"submit": submit, "per_job": per_job, "counts": counts}


def _load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    man = run_dir / "manifest.json"
    if not man.is_file():
        return {}
    try:
        obj = json.loads(man.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _downstream_cfg(run_dir: Path) -> Dict[str, Any]:
    man = _load_run_manifest(run_dir)
    cfg = dict(man.get("config") or {})
    return {
        "task": cfg.get("downstream_task"),
        "on_status": list(cfg.get("downstream_on_status") or ["OK_MIN", "OK_TS", "OK_NOFREQ", "OK_SP"]),
    }


def _count_materializable_jobs(collect_info: Optional[Dict[str, Any]], allowed_statuses: List[str]) -> int:
    rows = _load_collect_records(Path(collect_info["artifact_dir"]).parents[2] if collect_info else Path('.'), collect_info) if collect_info else []
    allowed = {str(x) for x in allowed_statuses}
    n = 0
    for row in rows:
        if str(row.get("collection_state") or "") != "COLLECTED":
            continue
        cls = row.get("classification") or {}
        if str(cls.get("status") or "") in allowed:
            n += 1
    return n


def detect_frontier(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    render = load_artifact_info(run_dir, "RenderBatch")
    submit = load_artifact_info(run_dir, "SubmitBatch")
    collect = load_artifact_info(run_dir, "CollectBatch")
    dcfg = _downstream_cfg(run_dir)
    downstream_task = str(dcfg.get("task") or "").strip()
    allowed_statuses = [str(x) for x in (dcfg.get("on_status") or [])]

    if render is None:
        return {
            "action": "none",
            "reason": "No RenderBatch present.",
            "render_present": False,
            "submit_present": False,
            "collect_present": False,
        }

    render_data = dict(render.get("data") or {})
    render_id = str(render.get("artifact_id") or "")
    submit_render_id = ""
    if submit is not None:
        submit_render_id = str((((submit.get("data") or {}).get("render_batch_ref") or {}).get("artifact_id") or ""))

    # New RenderBatch exists that has not been submitted yet.
    if submit is None or submit_render_id != render_id:
        return {
            "action": "submit",
            "reason": "Latest RenderBatch has not been submitted yet.",
            "render_present": True,
            "submit_present": submit is not None,
            "collect_present": collect is not None,
            "n_jobs": int(render_data.get("n_jobs") or 0),
        }

    submit_status = gather_submit_status(run_dir)
    per_job = list(submit_status.get("per_job") or [])
    counts = dict(submit_status.get("counts") or {})
    current_state_by_job = {str(r.get("job_dir") or ""): str(r.get("state") or "UNKNOWN") for r in per_job}

    submit_id = str(submit.get("artifact_id") or "")
    collect_submit_id = ""
    if collect is not None:
        collect_submit_id = str((((collect.get("data") or {}).get("submit_batch_ref") or {}).get("artifact_id") or ""))

    if collect is None or collect_submit_id != submit_id:
        return {
            "action": "collect",
            "reason": "Latest SubmitBatch has not been collected yet.",
            "render_present": True,
            "submit_present": True,
            "collect_present": collect is not None,
            "counts": counts,
        }

    recorded_rows = _load_collect_records(run_dir, collect)
    recorded_by_job = {str(r.get("job_dir") or ""): r for r in recorded_rows if isinstance(r, dict)}
    stale_jobs: List[str] = []
    for jd, cur_state in current_state_by_job.items():
        row = recorded_by_job.get(jd)
        if row is None:
            stale_jobs.append(jd)
            continue
        old_state = str(row.get("scheduler_state") or "UNKNOWN")
        if old_state != cur_state:
            stale_jobs.append(jd)

    if stale_jobs:
        return {
            "action": "collect",
            "reason": "Scheduler state has changed since the last CollectBatch.",
            "render_present": True,
            "submit_present": True,
            "collect_present": True,
            "counts": counts,
            "stale_jobs": stale_jobs,
        }

    active = counts.get("PENDING", 0) + counts.get("RUNNING", 0)
    if active > 0:
        return {
            "action": "wait",
            "reason": "Jobs are still active; no new frontier is ready.",
            "render_present": True,
            "submit_present": True,
            "collect_present": True,
            "counts": counts,
        }

    # If the latest batch is already the configured downstream task, stop here.
    # This prevents accidental recursion such as SP -> freq -> freq -> freq.
    latest_child_task = str(render_data.get("child_task") or "").strip()
    if downstream_task and latest_child_task == downstream_task:
        return {
            "action": "finish",
            "reason": f"Latest RenderBatch is already downstream '{downstream_task}'; no further promotion.",
            "render_present": True,
            "submit_present": True,
            "collect_present": True,
            "counts": counts,
        }

    # No active jobs; check if downstream work can be materialized from latest collect.
    if downstream_task and collect is not None:
        collect_id = str(collect.get("artifact_id") or "")
        render_parent_collect_id = str(((render_data.get("source_collect_ref") or {}).get("artifact_id") or ""))
        n_ready = 0
        allowed = {str(x) for x in allowed_statuses}
        for row in recorded_rows:
            if str(row.get("collection_state") or "") != "COLLECTED":
                continue
            cls = row.get("classification") or {}
            if str(cls.get("status") or "") in allowed:
                n_ready += 1
        if n_ready > 0 and render_parent_collect_id != collect_id:
            return {
                "action": "materialize",
                "reason": f"Latest CollectBatch can unlock {n_ready} downstream '{downstream_task}' jobs.",
                "render_present": True,
                "submit_present": True,
                "collect_present": True,
                "counts": counts,
                "n_ready": n_ready,
                "downstream_task": downstream_task,
            }

    return {
        "action": "finish",
        "reason": "Latest SubmitBatch is already collected and no jobs are active.",
        "render_present": True,
        "submit_present": True,
        "collect_present": True,
        "counts": counts,
    }
