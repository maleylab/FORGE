import json
from pathlib import Path

from typer.testing import CliRunner

from labtools.cli import app


runner = CliRunner()


def _valid_planentry_dict(task: str):
    """
    Minimal PlanEntry payload exercising non-SP task rendering.
    Mirrors the structure used in existing tests.
    """
    return {
        "id": "test_001",
        "schema": {"name": "test", "version": 1},
        "intent": {
            "task": task,
            "system": "generic",
        },
        "parameters": {
            "charge": 0,
            "multiplicity": 1,
            "method": "r2scan-3c",
        },
        "metadata": {
            "tags": ["test"],
        },
    }


def test_plan_render_opt_creates_one_input(tmp_path: Path):
    """
    Contract test:

    One PlanEntry (task=opt)
      → exactly one job directory
      → exactly one ORCA input file
      → OPT template is used
    """

    plan_path = tmp_path / "planentries.jsonl"
    plan_path.write_text(
        json.dumps(_valid_planentry_dict("opt")) + "\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "jobs"

    result = runner.invoke(
        app,
        ["plan", "render", "--plan", str(plan_path), "--outdir", str(outdir)],
    )

    assert result.exit_code == 0, result.output

    job_dirs = [p for p in outdir.iterdir() if p.is_dir()]
    assert len(job_dirs) == 1

    inp_files = list(job_dirs[0].glob("*.inp"))
    assert len(inp_files) == 1

    text = inp_files[0].read_text(encoding="utf-8")

    # Sanity checks that distinguish OPT from SP
    assert "opt" in text.lower() or "optimize" in text.lower()
    assert "charge" in text.lower()

