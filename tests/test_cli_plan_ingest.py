from pathlib import Path
import json

import pytest
from typer.testing import CliRunner

from labtools.cli import app


runner = CliRunner()


def _write_plan(tmp_path: Path, entries):
    path = tmp_path / "plan.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


def _valid_planentry_dict(**overrides):
    base = {
        "id": "test_001",
        "schema": {"name": "test_schema", "version": 1},
        "intent": {"task": "noop", "system": "dummy"},
        "parameters": {"x": 1, "y": 2},
        "metadata": {"tags": ["a", "b"], "notes": "ok"},
    }
    base.update(overrides)
    return base


# ==========================================================
# STUB RENDERER
# ==========================================================

def _noop_renderer(job, job_dir: Path):
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.inp").write_text("noop")


# ==========================================================
# HAPPY PATH
# ==========================================================


def test_plan_render_happy_path(tmp_path: Path, monkeypatch):
    """
    Valid PlanEntry JSONL → plan render succeeds
    (renderer stubbed)
    """
    plan = _write_plan(tmp_path, [_valid_planentry_dict()])
    outdir = tmp_path / "jobs"

    # Patch the renderer used by the CLI
    monkeypatch.setattr(
        "labtools.cli.render_plan",
        _noop_renderer,
    )

    result = runner.invoke(
        app,
        ["plan", "render", "--plan", str(plan), "--outdir", str(outdir)],
    )

    assert result.exit_code == 0
    assert outdir.exists()

    jobs = list(outdir.iterdir())
    assert len(jobs) == 1
    assert (jobs[0] / "plan_entry.json").exists()


# ==========================================================
# SCHEMA VIOLATIONS
# ==========================================================


@pytest.mark.parametrize(
    "bad_entry, match",
    [
        (
            {"schema": {"name": "x", "version": 1}},
            "Missing PlanEntry field",
        ),
        (
            _valid_planentry_dict(parameters=["not", "a", "dict"]),
            "parameters must be a dict",
        ),
        (
            _valid_planentry_dict(metadata={"tags": "nope"}),
            "metadata.tags must be a list of strings",
        ),
        (
            _valid_planentry_dict(intent={"task": 123, "system": "x"}),
            "intent.task must be a string",
        ),
    ],
)
def test_plan_render_schema_failures(tmp_path: Path, bad_entry, match):
    """
    Invalid PlanEntry payloads fail fast with clear errors
    """
    plan = _write_plan(tmp_path, [bad_entry])

    result = runner.invoke(
        app,
        ["plan", "render", "--plan", str(plan)],
    )

    assert result.exit_code != 0

    combined = (result.stdout or "") + (result.stderr or "")

    # Strip Typer / Rich box-drawing characters
    normalized = (
        combined
        .replace("│", " ")
        .replace("─", " ")
        .replace("╭", " ")
        .replace("╮", " ")
        .replace("╰", " ")
        .replace("╯", " ")
    )

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    assert match in normalized



# ==========================================================
# FAIL-FAST ON MULTILINE
# ==========================================================


def test_plan_render_stops_on_first_invalid_entry(tmp_path: Path, monkeypatch):
    """
    First invalid entry aborts execution (no partial renders)
    """
    entries = [
        _valid_planentry_dict(id="ok"),
        _valid_planentry_dict(parameters="bad"),
        _valid_planentry_dict(id="should_not_be_seen"),
    ]

    plan = _write_plan(tmp_path, entries)
    outdir = tmp_path / "jobs"

    monkeypatch.setattr(
        "labtools.cli.render_plan",
        _noop_renderer,
    )

    result = runner.invoke(
        app,
        ["plan", "render", "--plan", str(plan), "--outdir", str(outdir)],
    )

    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "parameters must be a dict" in combined

    if outdir.exists():
        jobs = list(outdir.iterdir())
        assert len(jobs) <= 1

