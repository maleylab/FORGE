# labtools/workflow/worker_cmd.py

from __future__ import annotations

import time
import traceback
from pathlib import Path

from labtools.workflow.worker import workflow_step
from labtools.workflow.state import WorkflowState
from labtools.workflow.provenance import write_stage_provenance


def workflow_entrypoint(jobdir: Path, poll_seconds: float = 2.0) -> int:
    """
    Drive the entire multi-stage workflow until completion.
    Called from the SLURM workflow.sbatch script.
    """

    jobdir = Path(jobdir)

    write_stage_provenance(jobdir, "workflow_start")

    while True:
        try:
            workflow_step(jobdir)
            state = WorkflowState.load(jobdir)

        except Exception as exc:
            tb = traceback.format_exc()
            write_stage_provenance(
                jobdir,
                "workflow_exception",
                extra={"error": str(exc), "traceback": tb},
            )
            print(tb)
            return 2

        if state.status in ("DONE", "SUCCESS"):
            write_stage_provenance(jobdir, "workflow_done")
            print(f"Workflow completed successfully at stage index {state.stage_index}.")
            return 0

        if state.status in ("FAIL", "FAIL_PERMANENT"):
            write_stage_provenance(
                jobdir,
                "workflow_fail",
                extra={"stage": state.current_stage, "attempt": state.attempt},
            )
            print(f"Workflow failed permanently on stage '{state.current_stage}'.")
            return 1

        time.sleep(poll_seconds)
