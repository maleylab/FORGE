"""
TSGen2 | plan.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
import yaml
import numpy as np

# Correct import path for fingerprint resolver
from .fp.resolver import resolve_fingerprint_path


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _resolve(p: str | Path | None) -> Optional[Path]:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


def _read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    txt = path.read_text().strip().splitlines()

    # Remove XYZ header if present
    if len(txt) >= 2 and txt[0].strip().isdigit():
        txt = txt[2:]

    atoms = []
    coords = []

    for line in txt:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return atoms, np.array(coords, float)


# ---------------------------------------------------------
# TSGenPlan — single source of truth for TSGen pipeline
# ---------------------------------------------------------
@dataclass
class TSGenPlan:

    # Required paths
    reactant: Path
    product: Path
    work_dir: Path           # must match test harness

    # Required electronic structure info
    charge: int
    mult: int

    # Methods
    l0_method: str = "XTB2"
    l1_method: str = "M062X/Def2-SVP"  
    l2_method: str = "M062X/Def2-SVP"

    l0_constraints: Optional[List[str]] = None

    # Cluster / runtime configuration
    execution_mode: str = "array"
    profile: str = "medium"

    # TSGen semantic mode
    mode: str = "exploratory"   # exploratory | confirmatory

    # Fingerprint system
    fingerprint_file: Optional[Path] = None
    force_pass_fingerprint: bool = False

    # Stage parameters
    max_seeds: int = 5
    nprocs: int = 8
    maxcore: int = 4000

    # Optional L3
    do_l3: bool = False
    l3_method: Optional[str] = None

    # -----------------------------------------------------
    # Stage-specific behavior (schema only)
    # -----------------------------------------------------

    # Optional raw ORCA keyword blocks by stage
    stage_keywords: Optional[Dict[str, List[str]]] = None

    # Explicit atom list for L1 Hybrid_Hess / PHVA
    l1_hybrid_hess_atoms: Optional[List[int]] = None

    # -----------------------------------------------------
    # Runtime state
    # -----------------------------------------------------
    reactant_xyz: Tuple[List[str], np.ndarray] = field(init=False)
    product_xyz: Tuple[List[str], np.ndarray] = field(init=False)

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("TSGen"))
    stages: Dict[str, Any] = field(init=False)

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------
    def __post_init__(self):
        # Validate mode
        if self.mode not in {"exploratory", "confirmatory"}:
            raise ValueError(
                f"Invalid TSGen mode '{self.mode}'. "
                "Expected 'exploratory' or 'confirmatory'."
            )

        # Resolve absolute paths
        self.reactant = _resolve(self.reactant)
        self.product  = _resolve(self.product)
        self.work_dir = _resolve(self.work_dir)

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

        # Load reactant/product XYZ
        self.reactant_xyz = _read_xyz(self.reactant)
        self.product_xyz  = _read_xyz(self.product)

        # Fingerprint resolution
        if self.fingerprint_file:
            fp_path = resolve_fingerprint_path(
                self.fingerprint_file,
                base=self.work_dir
            )
            self.fingerprint_file = fp_path

        # Logger formatting
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[TSGen] %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        # Stage registry
        from .stages.l0_seed import run_l0
        from .stages.l1_opt  import run_l1
        from .stages.l2_opt  import run_l2
        from .stages.verify  import run_verify
        from .stages.l3_sp   import run_l3

        self.stages = {
            "L0": run_l0,
            "L1": run_l1,
            "L2": run_l2,
            "VERIFY": run_verify,
            "L3": run_l3,
        }

    # -----------------------------------------------------
    # Directory helpers
    # -----------------------------------------------------
    def stage_dir(self, stage: str) -> Path:
        d = self.work_dir / stage
        d.mkdir(exist_ok=True)
        return d

    def job_dir(self, stage: str, job_name: str) -> Path:
        d = self.stage_dir(stage) / job_name
        d.mkdir(exist_ok=True)
        return d

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        return {
            "reactant": str(self.reactant),
            "product": str(self.product),
            "work_dir": str(self.work_dir),
            "charge": self.charge,
            "mult": self.mult,
            "l0_method": self.l0_method,
            "l1_method": self.l1_method,
            "l2_method": self.l2_method,
            "l0_constraints": self.l0_constraints,
            "max_seeds": self.max_seeds,
            "nprocs": self.nprocs,
            "maxcore": self.maxcore,
            "do_l3": self.do_l3,
            "l3_method": self.l3_method,
            "execution_mode": self.execution_mode,
            "profile": self.profile,
            "mode": self.mode,
            "fingerprint_file": (
                str(self.fingerprint_file)
                if self.fingerprint_file else None
            ),
            "stage_keywords": self.stage_keywords,
            "l1_hybrid_hess_atoms": self.l1_hybrid_hess_atoms,
        }

    # -----------------------------------------------------
    @staticmethod
    def from_yaml(path: Path) -> "TSGenPlan":
        data = yaml.safe_load(Path(path).read_text())
        return TSGenPlan(**data)
