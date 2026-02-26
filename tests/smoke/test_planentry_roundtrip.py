from pathlib import Path

from labtools.plans.types import PlanEntry
from labtools.csvpipe.emit import emit_planentries_jsonl
from labtools.plans.load import load_planentries_jsonl


def test_planentry_jsonl_roundtrip(tmp_path: Path):
    """
    PlanEntry → JSONL → PlanEntry roundtrip.

    This locks the PlanEntry document schema and loader behavior.
    """

    entry = PlanEntry(
        id="rt_001",
        schema_name="test_schema",
        task="noop",
        system="dummy_system",
        parameters={"x": 1, "y": 2},
        tags=["roundtrip"],
        notes="test",
    )

    out = tmp_path / "planentries.jsonl"

    emit_planentries_jsonl(
        [entry],
        outpath=out,
        dry_run=False,
    )

    loaded = load_planentries_jsonl(out)

    assert len(loaded) == 1

    e = loaded[0]

    assert e.id == entry.id
    assert e.schema_name == entry.schema_name
    assert e.schema_version == entry.schema_version
    assert e.task == entry.task
    assert e.system == entry.system
    assert e.parameters == entry.parameters
    assert e.tags == entry.tags
    assert e.notes == entry.notes

