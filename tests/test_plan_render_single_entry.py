from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from labtools.cli import app


runner = CliRunner()


def _valid_planentry_dict() -> dict:
    """
    Minimal valid PlanEntry payload (serialized form).
    """
    return {
        "id": "test_001",
        "schema": {
            "name": "debug",
            "version": 1,
        },
        "intent": {
            "task": "orca",
            "system": "generic",
        },
        "parameters": {
            "charge": 0,
            "multiplicity": 1,
            "method": "r2SCAN-3c",
            "job_type": "sp",

        },
        "metadata": {
            "tags": ["test"],
            "notes": "single entry render test",
        },
    }


def test_plan_render_single_entry_creates_one_input(tmp_path: Path):
    """
    Contract test:

    One PlanEntry
      → exactly one job directory
      → exactly one ORCA input file
      → rendered directly (no workflow engine, no SLURM)
    """

    # -------------------------
    # Write planentries.jsonl
    # -------------------------
    plan_path = tmp_path / "planentries.jsonl"
    plan_path.write_text(
        json.dumps(_valid_planentry_dict()) + "\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "jobs"

    # -------------------------
    # Run CLI
    # -------------------------
    result = runner.invoke(
        app,
        ["plan", "render", "--plan", str(plan_path), "--outdir", str(outdir)],
    )

    assert result.exit_code == 0, result.output

    # -------------------------
    # Assertions
    # -------------------------
    assert outdir.is_dir()

    job_dirs = sorted(p for p in outdir.iterdir() if p.is_dir())
    assert len(job_dirs) == 1, "Exactly one job directory must be created"

    job_dir = job_dirs[0]

    inp_files = list(job_dir.glob("*.inp"))
    assert len(inp_files) == 1, "Exactly one ORCA input file must be rendered"

    inp = inp_files[0]
    text = inp.read_text(encoding="utf-8")

    assert text.strip(), "Rendered input file must not be empty"
    assert "charge" in text.lower(), "Input should reflect PlanEntry parameters"

