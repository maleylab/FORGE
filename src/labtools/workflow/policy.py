from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Restart profiles (optional presets)
# ---------------------------------------------------------------------------

RESTART_PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {"max_restarts": 1},
    "default": {"max_restarts": 3},
    "aggressive": {"max_restarts": 5},
}


# ---------------------------------------------------------------------------
# RestartPolicy (CLEAN + TEST-CORRECT VERSION)
# ---------------------------------------------------------------------------

@dataclass
class RestartPolicy:
    """
    Defines restart capabilities for workflow stages.

    Fields:
      max_restarts: global limit per stage
      per_stage_max: optional override {stage: max}
      allow: {stage: bool, "default": bool}
    """

    max_restarts: int = 3
    per_stage_max: Dict[str, int] = field(default_factory=dict)
    allow: Dict[str, bool] = field(default_factory=lambda: {"default": True})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any] | None) -> "RestartPolicy":
        """
        Build from job.yaml entry:

        restart:
          profile: default
          max_restarts: 5
          per_stage_max:
            opt: 2
          allow:
            opt: false
            default: true
        """

        if cfg is None:
            return cls()

        cfg = dict(cfg)

        # Step 1 — profile
        profile_name = cfg.pop("profile", None)
        profile_data = RESTART_PROFILES.get(profile_name, {}) if profile_name else {}

        # Step 2 — merge profile → user overrides
        merged = {**profile_data, **cfg}

        max_restarts = int(merged.get("max_restarts", 3))
        per_stage_max = dict(merged.get("per_stage_max", {}))
        allow = dict(merged.get("allow", {}))

        # Always define a "default" rule
        if "default" not in allow:
            allow["default"] = True

        return cls(
            max_restarts=max_restarts,
            per_stage_max=per_stage_max,
            allow=allow,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def max_restarts_for(self, stage: str) -> int:
        """Return per-stage limit with fallback to global."""
        return int(self.per_stage_max.get(stage, self.max_restarts))

    def allow_restart(self, stage: str, fail_type: str | None = None) -> bool:
        """
        Determine whether restart is allowed for this stage.
        """
        if stage in self.allow:
            return bool(self.allow[stage])
        return bool(self.allow.get("default", True))

    # alias used in worker.py
    def is_allowed(self, stage: str, fail_type: str | None = None) -> bool:
        return self.allow_restart(stage, fail_type)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_restarts": self.max_restarts,
            "per_stage_max": self.per_stage_max,
            "allow": self.allow,
        }
