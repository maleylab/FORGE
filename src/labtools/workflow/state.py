from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

STATE_FILE = ".forge/state.json"
HISTORY_FILE = ".forge/history.jsonl"

VALID_STATUSES = {
    "PENDING",
    "RUNNING",
    "RESTARTING",
    "SUCCESS",
    "FAIL_RETRY",
    "FAIL_PERMANENT",
    "DONE",
    "FAIL",
}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@dataclass
class WorkflowState:
    """
    Persistent workflow state for multi-stage workflows.
    """

    job_id: str
    stages: List[str]
    stage_index: int = 0
    status: str = "PENDING"
    attempt: int = 0
    max_restarts: int = 3

    history: List[Dict[str, Any]] = field(default_factory=list)
    last_geometry: Optional[List[str]] = None

    # Lifecycle timestamps (non-breaking additions)
    stage_started: Optional[str] = None
    stage_finished: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def current_stage(self) -> str:
        return self.stages[self.stage_index]

    @property
    def is_finished(self) -> bool:
        return self.stage_index >= len(self.stages)

    @property
    def attempts_exhausted(self) -> bool:
        return self.attempt >= self.max_restarts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def load(jobdir: Path) -> "WorkflowState":
        state_path = jobdir / STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(f"No workflow state found: {state_path}")

        with state_path.open("r") as f:
            data = json.load(f)
        return WorkflowState(**data)

    def save(self, jobdir: Path) -> None:
        state_path = jobdir / STATE_FILE
        state_path.parent.mkdir(exist_ok=True)
        with state_path.open("w") as f:
            json.dump(self.__dict__, f, indent=2)

    def append_history(self, jobdir: Path, record: Dict[str, Any]) -> None:
        history_path = jobdir / HISTORY_FILE
        history_path.parent.mkdir(exist_ok=True)
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.history.append(record)

    # ------------------------------------------------------------------
    # Lifecycle hook helpers
    # ------------------------------------------------------------------
    def mark_stage_start(self) -> None:
        self.status = "RUNNING"
        self.stage_started = _timestamp()
        if self.attempt == 0:
            self.attempt = 1

    def mark_stage_success(self, final_geometry: Optional[List[str]] = None) -> None:
        self.status = "SUCCESS"
        self.stage_finished = _timestamp()
        if final_geometry:
            self.last_geometry = final_geometry

    def mark_stage_failure(self, fail_type: str) -> None:
        self.status = "FAIL_RETRY"
        self.stage_finished = _timestamp()

    def forbid_restart_fail(self, fail_type: str) -> None:
        self.status = "FAIL_PERMANENT"
        self.stage_finished = _timestamp()

    def increment_attempt(self) -> None:
        self.attempt += 1
        self.status = "RUNNING"

    def reset_attempts(self) -> None:
        self.attempt = 0

    def advance_stage(self) -> None:
        self.stage_index += 1
        self.attempt = 0
        self.stage_started = None
        self.stage_finished = None

        if self.is_finished:
            self.status = "DONE"
        else:
            self.status = "PENDING"

    def mark_fail(self, fail_type: str, reason: str) -> None:
        self.status = "FAIL"
        self.stage_finished = _timestamp()

    def mark_restart_forbidden(self) -> None:
        self.status = "FAIL"
        self.stage_finished = _timestamp()
