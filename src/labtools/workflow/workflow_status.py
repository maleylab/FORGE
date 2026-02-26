from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import yaml
import json

from labtools.workflow.state import WorkflowState


def workflow_status(jobdir: Path) -> Dict[str, Any]:
    """
    Lightweight status reporter.
    Safe: does not modify state, does not assume running conditions.
    """
    jobdir = Path(jobdir)
    job_yaml = jobdir / "job.yaml"

    if not job_yaml.exists():
        return {"status": "ERROR", "reason": "missing_job_yaml"}

    try:
        job_config = yaml.safe_load(job_yaml.read_text())
    except Exception:
        job_config = {}

    try:
        state = WorkflowState.load(jobdir)
    except FileNotFoundError:
        return {"status": "UNINITIALIZED"}

    stages = state.stages
    idx = state.stage_index
    current = None
    if idx < len(stages):
        current = stages[idx]

    return {
        "job_id": state.job_id,
        "status": state.status,
        "current_stage": current,
        "attempt": state.attempt,
        "completed": state.is_finished,
        "stages": stages,
        "stage_index": idx,
    }
