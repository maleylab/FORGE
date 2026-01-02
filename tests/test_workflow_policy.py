from __future__ import annotations

import json
from pathlib import Path
import yaml

from labtools.workflow.worker import workflow_step
from labtools.workflow.state import WorkflowState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_job_yaml(jobdir: Path, cfg: dict) -> None:
    with (jobdir / "job.yaml").open("w") as f:
        yaml.safe_dump(cfg, f)


def _read_state(jobdir: Path) -> WorkflowState:
    return WorkflowState.load(jobdir)


def _read_failure_json(jobdir: Path) -> dict:
    p = jobdir / ".forge" / "failure.json"
    assert p.exists()
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Dummy WorkflowEngine (minimal API for worker.py)
# ---------------------------------------------------------------------------

class DummyEngine:
    """
    Replacement for WorkflowEngine during unit tests.
    Defines a single-stage workflow: ['opt'].
    """

    def __init__(self, job_config, jobdir: Path) -> None:
        self.job_config = job_config
        self.jobdir = jobdir
        self._stages = [{"name": "opt"}]

    def initialize_state(self) -> WorkflowState:
        return WorkflowState(
            job_id=self.job_config.get("id", "test-job"),
            stages=["opt"],
            stage_index=0,
            status="PENDING",
            attempt=0,
        )

    def resolve_stages(self):
        return self._stages

    def render_stage_input(self, stage, stage_dir: Path) -> Path:
        inp = stage_dir / "job.inp"
        inp.write_text("! HF 3-21G\n* xyz 0 1\nH 0 0 0\n*\n")
        return inp


# ---------------------------------------------------------------------------
# TEST 1 — restart forbidden → FAIL
# ---------------------------------------------------------------------------

def test_restart_not_allowed_marks_fail(tmp_path, monkeypatch):

    jobdir = tmp_path / "job"
    jobdir.mkdir()

    job_cfg = {
        "id": "job-no-restart",
        "cpus": 1,
        "restart": {
            "max_restarts": 5,
            "allow": {"opt": False, "default": True},
        },
    }
    _write_job_yaml(jobdir, job_cfg)

    import labtools.workflow.worker as worker_mod

    # Patch dependencies
    monkeypatch.setattr(worker_mod, "WorkflowEngine", DummyEngine)
    monkeypatch.setattr(worker_mod, "run_orca", lambda inp, **kw: 0)
    monkeypatch.setattr(worker_mod, "classify_failure", lambda rec, out: "geom_convergence")
    monkeypatch.setattr(worker_mod, "parse_orca_file", lambda out: {"termination": "failure"})
    monkeypatch.setattr(worker_mod, "write_stage_provenance", lambda d, l: None)

    # generate_restart must NOT be called
    import labtools.orca.restart as restart_mod
    def explode(*args, **kw):
        raise AssertionError("generate_restart should NOT be called when restart is disallowed")
    monkeypatch.setattr(restart_mod.RestartEngine, "generate_restart", explode)

    # First step: create stage
    workflow_step(jobdir)
    stage_dir = jobdir / "opt"
    (stage_dir / "output.out").write_text("fake output\n")

    # Second step: fail → restart forbidden → FAIL state
    workflow_step(jobdir)

    state = _read_state(jobdir)
    assert state.status == "FAIL"

    # FAIL sentinel
    fail_path = jobdir / "FAIL"
    assert fail_path.exists()
    assert "failed permanently" in fail_path.read_text()

    # failure.json
    meta = _read_failure_json(jobdir)
    assert meta["reason"] == "restart_not_allowed"
    assert meta["stage"] == "opt"
    assert meta["fail_type"] == "geom_convergence"


# ---------------------------------------------------------------------------
# TEST 2 — max restarts exceeded → FAIL
# ---------------------------------------------------------------------------

def test_max_restarts_exceeded_marks_fail(tmp_path, monkeypatch):

    jobdir = tmp_path / "job"
    jobdir.mkdir()

    job_cfg = {
        "id": "job-max-restarts",
        "cpus": 1,
        "restart": {"max_restarts": 2},
    }
    _write_job_yaml(jobdir, job_cfg)

    import labtools.workflow.worker as worker_mod

    monkeypatch.setattr(worker_mod, "WorkflowEngine", DummyEngine)
    monkeypatch.setattr(worker_mod, "run_orca", lambda inp, **kw: 0)
    monkeypatch.setattr(worker_mod, "classify_failure", lambda rec, out: "scf_convergence")
    monkeypatch.setattr(worker_mod, "parse_orca_file", lambda out: {"termination": "failure"})
    monkeypatch.setattr(worker_mod, "write_stage_provenance", lambda d, l: None)

    # RestartEngine.generate_restart must NOT be called
    import labtools.orca.restart as restart_mod
    def explode(*args, **kw):
        raise AssertionError("generate_restart should NOT be called when max_restarts exceeded")
    monkeypatch.setattr(restart_mod.RestartEngine, "generate_restart", explode)

    # First workflow call — stage created
    workflow_step(jobdir)
    stage_dir = jobdir / "opt"
    (stage_dir / "output.out").write_text("fake\n")

    # Pretend we already used all restarts
    state = _read_state(jobdir)
    state.attempt = 2  # equal to max_restarts
    state.save(jobdir)

    # Second workflow call — must FAIL
    workflow_step(jobdir)

    state = _read_state(jobdir)
    assert state.status == "FAIL"

    meta = _read_failure_json(jobdir)
    assert meta["reason"] == "max_restarts_exceeded"
    assert meta["fail_type"] == "scf_convergence"


# ---------------------------------------------------------------------------
# TEST 3 — restart allowed → restart happens
# ---------------------------------------------------------------------------

def test_restart_allowed_and_under_limit_executes_restart(tmp_path, monkeypatch):

    jobdir = tmp_path / "job"
    jobdir.mkdir()

    job_cfg = {
        "id": "job-restart-allowed",
        "cpus": 1,
        "restart": {
            "max_restarts": 5,
            "allow": {"opt": True, "default": True},
        },
    }
    _write_job_yaml(jobdir, job_cfg)

    import labtools.workflow.worker as worker_mod

    monkeypatch.setattr(worker_mod, "WorkflowEngine", DummyEngine)
    monkeypatch.setattr(worker_mod, "run_orca", lambda inp, **kw: 0)
    monkeypatch.setattr(worker_mod, "classify_failure", lambda rec, out: "scf_convergence")
    monkeypatch.setattr(worker_mod, "parse_orca_file", lambda out: {"termination": "failure"})
    monkeypatch.setattr(worker_mod, "write_stage_provenance", lambda d, l: None)

    # Count calls to generate_restart
    calls = {"n": 0}

    import labtools.orca.restart as restart_mod
    def fake_generate_restart(self, stage_dir, rec, fail_type):
        calls["n"] += 1
        p = stage_dir / "restart.inp"
        p.write_text("! restart input\n")
        return p

    monkeypatch.setattr(restart_mod.RestartEngine, "generate_restart", fake_generate_restart)

    # Step 1 — stage setup
    workflow_step(jobdir)
    stage_dir = jobdir / "opt"
    (stage_dir / "output.out").write_text("fake\n")

    # Step 2 — should trigger restart
    workflow_step(jobdir)

    # RestartEngine must have been called once
    assert calls["n"] == 1

    # Validate state
    state = _read_state(jobdir)
    assert state.status == "RUNNING"
    assert state.attempt == 2  # initial attempt=1 → after restart attempt=2

    # No FAIL artifacts
    assert not (jobdir / "FAIL").exists()
    assert not (jobdir / ".forge" / "failure.json").exists()
