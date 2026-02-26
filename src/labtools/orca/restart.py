from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from labtools.workflow.policy import RestartPolicy
from labtools.orca.restart_overrides import RestartOverrides
from labtools.orca.restart_registry import STRATEGY_REGISTRY


class RestartEngine:
    """
    Returns structured RestartOverrides dicts used by templates.
    """

    def __init__(self, policy: Optional[RestartPolicy] = None) -> None:
        self.policy = policy or RestartPolicy()

    def generate_restart(
        self,
        stage_dir: Path,
        rec: Dict[str, Any],
        fail_type: str,
    ) -> RestartOverrides:
        """
        Select the next restart strategy and return overrides.
        """

        stage = stage_dir.name

        # -----------------------------------------------------
        # Restart not allowed → EMPTY overrides
        # -----------------------------------------------------
        if not self.policy.allow_restart(stage, fail_type):
            return RestartOverrides.empty()

        # Strategy list for this failure type
        strategies = STRATEGY_REGISTRY.get(
            fail_type,
            STRATEGY_REGISTRY["unknown_failure"]
        )

        # Choose strategy based on attempt index
        attempt = rec.get("attempt", 0)
        idx = min(attempt, len(strategies) - 1)

        strategy_fn = strategies[idx]

        # Return structured overrides
        return strategy_fn()
