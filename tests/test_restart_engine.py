from __future__ import annotations
from pathlib import Path

from labtools.orca.restart import RestartEngine
from labtools.workflow.policy import RestartPolicy
from labtools.orca.restart_overrides import RestartOverrides
from labtools.orca.restart_registry import STRATEGY_REGISTRY


def test_restartengine_disallow_diagnostic(tmp_path):
    print("\n========== DIAGNOSTIC TEST START ==========\n")

    # Explicit policy disallowing restart for 'opt'
    policy = RestartPolicy.from_config({
        "allow": {"opt": False, "default": True},
        "max_restarts": 3,
    })

    print("Policy.allow =", policy.allow)
    print("Policy.allow_restart('opt') =", policy.allow_restart("opt", "scf_convergence"))

    engine = RestartEngine(policy)

    stage_dir = tmp_path / "opt"
    stage_dir.mkdir()

    rec = {"attempt": 0}

    overrides = engine.generate_restart(
        stage_dir=stage_dir,
        rec=rec,
        fail_type="scf_convergence",
    )

    print("Overrides returned:", overrides)
    print("Overrides.global_flags_add =", overrides.global_flags_add)
    print("Overrides.scf =", overrides.scf)

    print("\n========== DIAGNOSTIC TEST END ==========\n")

    # The actual assertion:
    assert overrides.global_flags_add == []
    assert overrides.scf == {}

