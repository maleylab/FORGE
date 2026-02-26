from pathlib import Path

from labtools.plans.types import PlanEntry
from labtools.plans.render import render_planentries

def test_planentry_render_adapter(tmp_path: Path):
    rendered = {}

    def fake_renderer(job: dict, job_dir: Path):
        rendered["job"] = job
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(str(job))

    entry = PlanEntry(
        id="test_001",
        schema_name="test_schema",
        task="noop",
        system="dummy_system",
        parameters={"x": 1, "y": 2},
        tags=["unit-test"],
        notes="golden path",
    )

    render_planentries(
        [entry],
        render_func=fake_renderer,
        outdir=tmp_path,
    )

    # One job directory created
    job_dirs = list(tmp_path.iterdir())
    assert len(job_dirs) == 1

    # Renderer payload
    job = rendered["job"]
    assert isinstance(job, dict)

    # System propagated
    assert job["system"] == "dummy_system"

    # Parameters flattened
    assert job["x"] == 1
    assert job["y"] == 2

