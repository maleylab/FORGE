from pathlib import Path
import json

from labtools.csvpipe.emit import emit_planentries_jsonl
from labtools.csvpipe.emit import job_to_planentry
from labtools.plans.render import render_planentries


def test_csvpipe_planentries_render_roundtrip(tmp_path: Path):
    """
    End-to-end sanity test:

    CSV → job dict → PlanEntry → JSONL → render adapter

    This asserts that the CLI wiring path is a thin wrapper
    over already-tested primitives.
    """

    # -------------------------
    # Fake expanded job (what csvpipe produces)
    # -------------------------
    job = {
        "system": "dummy_system",
        "parameters": {
            "x": 1,
            "y": 2,
        },
    }

    entry = job_to_planentry(
        job,
        schema_name="test_schema",
        schema_version=1,
        task="noop",
        index=0,
    )

    plan_path = tmp_path / "planentries.jsonl"

    # -------------------------
    # Emit JSONL
    # -------------------------
    emit_planentries_jsonl([entry], plan_path)

    assert plan_path.is_file()

    # -------------------------
    # Load PlanEntry from disk
    # -------------------------
    from labtools.plans.types import PlanEntry

    entries = []
    for line in plan_path.read_text().splitlines():
        entries.append(json.loads(line))

    assert len(entries) == 1

    # -------------------------
    # Render adapter
    # -------------------------
    rendered = {}

    def fake_renderer(job_dict: dict, job_dir: Path):
        rendered["job"] = job_dict
        job_dir.mkdir(parents=True, exist_ok=True)

    render_planentries(entries, render_func=fake_renderer, outdir=tmp_path)

    # -------------------------
    # Assertions
    # -------------------------
    assert "job" in rendered

    job_payload = rendered["job"]

    assert job_payload["system"] == "dummy_system"
    assert job_payload["x"] == 1
    assert job_payload["y"] == 2

