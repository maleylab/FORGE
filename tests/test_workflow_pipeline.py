import json
from pathlib import Path
import pytest

from labtools.workflow.worker import workflow_step
from labtools.workflow.state import WorkflowState
from labtools.workflow.engine import WorkflowEngine

import labtools.workflow.utils as utils
import labtools.orca.collect as collect
import labtools.orca.classify as classify


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def write_job_yaml(tmp, pipeline=None):
    job = {
        "id": "TJOB",
        "method": "HF",
        "basis": "sto-3g",
        "flags": [],
        "charge": 0,
        "mult": 1,
        "xyz": [
            "C 0.0 0.0 0.0",
            "H 0.0 0.0 1.1",
        ],
        "pipeline": pipeline or ["opt", "freq", "sp"],
        "restart": {"max_restarts": 3}
    }
    p = tmp / "job.yaml"
    p.write_text(json.dumps(job, indent=2))
    return p


def fake_success(path):
    return {
        "errors": [],
        "final_geometry": [
            "C 0.0 0.0 0.1",
            "H 0.0 0.0 1.2",
        ]
    }


def fake_failure(path):
    return {
        "errors": ["SCF_FAILED"],
        "final_geometry": None,
    }


# -------------------------------------------------------
# Engine tests
# -------------------------------------------------------

def test_engine_resolves_stages(tmp_path):
    job = write_job_yaml(tmp_path)
    cfg = json.loads(job.read_text())
    eng = WorkflowEngine(cfg, tmp_path)

    stages = eng.resolve_stages()
    names = [s["name"] for s in stages]

    assert names == ["opt", "freq", "sp"]
    assert stages[0]["template"] == "orca_opt.inp.j2"
    assert stages[1]["template"] == "orca_freq.inp.j2"
    assert stages[2]["template"] == "orca_sp.inp.j2"


def test_engine_loads_initial_geometry(tmp_path):
    job = write_job_yaml(tmp_path)
    cfg = json.loads(job.read_text())
    eng = WorkflowEngine(cfg, tmp_path)

    geom = eng.load_geometry("opt")
    assert len(geom) == 2
    assert geom[0].startswith("C")


def test_engine_loads_propagated_geometry(tmp_path):
    job = write_job_yaml(tmp_path)
    cfg = json.loads(job.read_text())
    eng = WorkflowEngine(cfg, tmp_path)

    # simulate previous stage final.xyz
    prev = tmp_path / "opt"
    prev.mkdir()
    (prev / "final.xyz").write_text(
        "2\nheader\nC 1 1 1\nH 0 0 0\n"
    )

    geom = eng.load_geometry("freq")
    assert geom == ["C 1 1 1", "H 0 0 0"]


# -------------------------------------------------------
# Worker behavior tests
# -------------------------------------------------------

def test_worker_first_stage_run(tmp_path, monkeypatch):
    write_job_yaml(tmp_path)

    monkeypatch.setattr(utils, "run_orca", lambda inp, n: 0)

    workflow_step(tmp_path)

    state = WorkflowState.load(tmp_path)
    assert state.stage_index == 0
    assert state.status == "RUNNING"
    assert state.attempt == 1

    opt_dir = tmp_path / "opt"
    assert opt_dir.exists()
    assert (opt_dir / "opt.inp").exists()


def test_worker_orca_still_running(tmp_path, monkeypatch):
    write_job_yaml(tmp_path)
    monkeypatch.setattr(utils, "run_orca", lambda inp, n: 0)

    workflow_step(tmp_path)

    # No output yet
    msg = workflow_step(tmp_path)
    # Should simply return (None)
    assert msg is None


def test_successful_stage_advances(tmp_path, monkeypatch):
    write_job_yaml(tmp_path)

    monkeypatch.setattr(utils, "run_orca", lambda inp, n: 0)
    monkeypatch.setattr(collect, "parse_orca_out_file", fake_success)

    # Start opt
    workflow_step(tmp_path)

    # Provide ORCA output
    opt_dir = tmp_path / "opt"
    (opt_dir / "output.out").write_text("OK")

    workflow_step(tmp_path)

    # Geometry written?
    assert (opt_dir / "final.xyz").exists()

    state = WorkflowState.load(tmp_path)
    assert state.stage_index == 1   # moved to freq
    assert state.attempt == 0       # reset
    assert state.status == "PENDING"


def test_failure_then_restart_then_success(tmp_path, monkeypatch):
    write_job_yaml(tmp_path)
    monkeypatch.setattr(utils, "run_orca", lambda inp, n: 0)

    # First parse → failure; second parse → success
    calls = {"n": 0}

    def fake_parse(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return fake_failure(path)
        return fake_success(path)

    monkeypatch.setattr(collect, "parse_orca_out_file", fake_parse)
    monkeypatch.setattr(classify, "classify_failure", lambda r, p: "SCF_FAIL")

    # Step 1: run opt
    workflow_step(tmp_path)
    opt_dir = tmp_path / "opt"
    (opt_dir / "output.out").write_text("x")

    # Step 2: detect failure → restart
    workflow_step(tmp_path)

    state = WorkflowState.load(tmp_path)
    assert state.attempt == 2
    assert state.status == "RUNNING"

    # Step 3: success parse
    (opt_dir / "output.out").write_text("done")
    workflow_step(tmp_path)

    state = WorkflowState.load(tmp_path)
    assert state.stage_index == 1
    assert state.attempt == 0
    assert state.status == "PENDING"


def test_restart_not_allowed_generates_fail(tmp_path, monkeypatch):
    write_job_yaml(tmp_path)
    monkeypatch.setattr(utils, "run_orca", lambda inp, n: 0)
    monkeypatch.setattr(collect, "parse_orca_out_file", fake_failure)
    monkeypatch.setattr(classify, "classify_failure", lambda r, p: "BAD_INPUT")

    # Start opt
    workflow_step(tmp_path)
    opt_dir = tmp_path / "opt"
    (opt_dir / "output.out").write_text("x")

    # Restart forbidden → FAIL
    workflow_step(tmp_path)

    state = WorkflowState.load(tmp_path)
    assert state.status == "FAIL"

    fail_file = tmp_path / "FAIL"
    assert fail_file.exists()

    j = json.loads((tmp_path / ".forge" / "failure.json").read_text())
    assert j["reason"] == "restart_not_allowed"

