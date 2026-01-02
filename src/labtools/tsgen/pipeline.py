"""
TSGen 2.0 | tsgen_plan.py

Central configuration object for TSGen pipelines.
Everything downstream imports ONLY this plan.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import logging

from .fp.fingerprint import Fingerprint
from .fp.fingerprint import load_fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    return p


# ---------------------------------------------------------------------------
# TSGenPlan
# ---------------------------------------------------------------------------

@dataclass
class TSGenPlan:
    """
    TSGenPlan is the ONLY configuration object used by TSGen 2.0.

    It centralizes:
      - reactant/product paths
      - charge/multiplicity
      - methods for each stage (L0/L1/L2)
      - execution mode + SLURM profile
      - fingerprint object (optional)
      - working directory
      - logger

    All stages receive an instance of TSGenPlan.
    """

    # Required system inputs
    reactant: Path
    product: Path
    work_dir: Path

    # Electronic structure inputs
    charge: int
    mult: int

    # Methods (full ORCA-level strings)
    l0_method: str = "XTB2"
    l1_method: str = "M06/Def2-SVP"
    l2_method: str = "M06/Def2-SVP"

    # Execution
    execution_mode: str = "array"   # array | single | job
    profile: str = "medium"         # SLURM profile

    # Fingerprint (optional)
    fingerprint_file: Optional[Path] = None
    fingerprint: Optional[Fingerprint] = field(default=None, init=False)

    # TSGen internal config
    tsgen: Dict[str, Any] = field(default_factory=lambda: {
        "n_images": 7,
        "max_seeds": 3,
    })

    # Logger
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("TSGen"))

    # -----------------------------------------------------------------------
    # Post-init
    # -----------------------------------------------------------------------
    def __post_init__(self):

        # Normalize paths
        self.reactant = _resolve(self.reactant)
        self.product  = _resolve(self.product)
        self.work_dir = _resolve(self.work_dir)

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

        # Load fingerprint if provided
        if self.fingerprint_file is not None:
            fp_path = _resolve(self.fingerprint_file)
            if fp_path and fp_path.is_file():
                self.fingerprint = load_fingerprint(fp_path)
            else:
                raise FileNotFoundError(f"Fingerprint file not found: {fp_path}")

        # Configure logger if not configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = "[TSGen] %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a dictionary summary for debugging/logging."""
        return {
            "reactant": str(self.reactant),
            "product": str(self.product),
            "work_dir": str(self.work_dir),
            "charge": self.charge,
            "mult": self.mult,
            "l0_method": self.l0_method,
            "l1_method": self.l1_method,
            "l2_method": self.l2_method,
            "execution_mode": self.execution_mode,
            "profile": self.profile,
            "fingerprint_loaded": self.fingerprint is not None,
            "tsgen_settings": self.tsgen,
        }
