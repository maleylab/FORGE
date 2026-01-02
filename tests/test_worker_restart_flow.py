from __future__ import annotations
import yaml
from pathlib import Path

from labtools.workflow.worker import workflow_step
from labtools.workflow.engine import WorkflowEngine


def test_worker_performs_restart(tmp_path, monkeypatch):
    jobdir = tmp_path / "job"
    jobdir.mkdir()

    # job.yaml with restart enabled
    (jobdir / "job.yaml").write_text(yaml.safe_dump({
        "id": "job-restart",
        "cpus": 1,
        "restart": {
            "allow": {"opt": True},
            "max_restarts": 3,
        },
        "method": "B3LYP",
        "basis": "def2-SVP",
        "flags": ["TightSCF"],
    }))

    # DummyEngine
    import labtools.workflow.engine as engine_mod
    class DummyEngine(engine_mod.WorkflowEngine):
        def resolve_stages(self):
            return [{"name": "opt"}]

        def _read_structure(self, stage):
            return ["H 0 0 0"]

        def _resolve_template(self, stage):
            # Use a dummy inline template
            return "dummy.inp.j2"

    monkeypatch.setattr(engine_mod, "WorkflowEngine", DummyEngine)

    # Create template
    tpldir = jobdir / "templates"
    tpldir.mkdir()
    (tpldir / "dummy.inp.j2").write_text("""
! {{ method }} {{ restart_flags | join(' ') }}

%scf
{% for k, v in scf.items() %}  {{ k }} {{ v }}{% endfor %}
end

%geom
{% if geom.restart %}  Restart true{% endif %}
end

* xyz {{ charge }} {{ mult }}
H 0 0 0
*
""")

    # Load templates
    from labtools.workflow.engine import WorkflowEngine
    WorkflowEngine.jenv.loader.searchpath.insert(0, str(tpldir))

    # ORCA call is a noop
    monkeypatch.setattr("labtools.workflow.utils.run_orca", lambda inp, **kw: 0)

    # First parse: failure
    monkeypatch.setattr(
        "labtools.orca.parse.parse_orca_file",
        lambda _p: {"termination": "failure"}
    )

    # classify failure: SCF issue
    monkeypatch.setattr(
        "labtools.orca.classify.classify_failure",
        lambda rec, out: "scf_convergence"
    )

    # Step 1: generate the stage folder
    workflow_step(jobdir)
    stage_dir = jobdir / "opt"
    (stage_dir / "output.out").write_text("* dummy *")

    # Step 2: failure triggers restart
    workflow_step(jobdir)
    second_inp = (stage_dir / "job.inp").read_text()
    assert "Restart true" in second_inp or "MaxIter" in second_inp
