from pathlib import Path

from labtools.plans.render import render_planentries
from labtools.plans.types import PlanEntry


def fake_renderer(job, job_dir: Path):
    """
    Minimal legacy-style renderer.
    Writes the received job dict to disk so we can
    assert adapter correctness without ORCA/SLURM.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(repr(job))


def test_planentry_render_adapter(tmp_path: Path):
    entry = PlanEntry(
        id="test_001",
        schema_name="test",
        schema_version=1,
        task="noop",
        system="foo",
        parameters={
            "x": 1,
            "y": 2,
        },
    )

    render_planentries(
        entries=[entry],
        render_func=fake_renderer,
        outdir=tmp_path,
    )

    job_dir = tmp_path / "job_00000"
    job_file = job_dir / "job.json"

    # Structural assertions
    assert job_dir.exists(), "Job directory was not created"
    assert job_file.exists(), "Renderer did not write job.json"

    # Semantic assertion: adapter preserved parameters
    text = job_file.read_text()
    assert "'x': 1" in text
    assert "'y': 2" in text
    assert "'foo'" in text or "foo" in text
