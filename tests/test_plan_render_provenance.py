import json
from pathlib import Path

from labtools.plans.factory import planentry_from_dict
from labtools.plans.adapters import planentry_to_dict
from labtools.plans.render import render_planentries
from labtools.plans.types import PlanEntry


def test_render_embeds_planentry(tmp_path: Path):
    entry = PlanEntry(
        id="test_001",
        schema_name="unit-test",
        schema_version=1,
        task="noop",
        system="dummy",
        parameters={"x": 1},
        tags=["a", "b"],
        notes="hello",
    )

    outdir = tmp_path / "jobs"

    def _noop_renderer(job, job_dir):
        pass

    render_planentries(
        [entry],
        render_func=_noop_renderer,
        outdir=outdir,
    )

    pe_path = outdir / "job_00000" / "plan_entry.json"
    assert pe_path.is_file()

    data = json.loads(pe_path.read_text())
    rebuilt = planentry_from_dict(data)

    assert rebuilt == entry

