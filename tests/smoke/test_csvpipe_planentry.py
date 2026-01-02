from labtools.csvpipe.emit import jobs_to_planentries
from labtools.plans.types import PlanEntry


def test_jobs_to_planentries_minimal():
    """
    Contract test for csvpipe → PlanEntry adapter.

    Asserts that:
    - A csvpipe-style job dict is converted into exactly one PlanEntry
    - Semantic fields are preserved
    - Parameters are passed through verbatim
    """

    jobs = [
        {
            "id": "job_001",
            "task": "noop",
            "system": "dummy_system",
            "x": 1,
            "y": 2,
        }
    ]

    entries = jobs_to_planentries(
        jobs,
        schema_name="test_schema",
        schema_version=1,
    )

    assert len(entries) == 1
    entry = entries[0]

    assert isinstance(entry, PlanEntry)

    assert entry.id == "job_001"
    assert entry.schema_name == "test_schema"
    assert entry.schema_version == 1

    assert entry.task == "noop"
    assert entry.system == "dummy_system"

    assert entry.parameters == {
        "x": 1,
        "y": 2,
    }

