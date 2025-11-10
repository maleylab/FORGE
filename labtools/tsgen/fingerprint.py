"""
fingerprint.py
FORGE | tsgen

Stores and verifies transition-state fingerprints.
A fingerprint encodes the expected imaginary mode vector (mass-weighted)
and the subset of atoms participating in the reaction coordinate.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# -------------------------------------------------------------------------
# Data container
# -------------------------------------------------------------------------
@dataclass
class Fingerprint:
    """Transition-state fingerprint."""
    atom_indices: list[int]        # 0-based indices of key atoms
    ref_mode: np.ndarray           # reference displacement vector (mass-weighted)
    mode_label: Optional[str] = None
    threshold_cosine: float = 0.80
    threshold_localization: float = 0.70

    @classmethod
    def load(cls, path: Path) -> "Fingerprint":
        data = json.loads(Path(path).read_text())
        ref_mode = np.array(data["ref_mode"])
        return cls(
            atom_indices=data["atom_indices"],
            ref_mode=ref_mode,
            mode_label=data.get("mode_label"),
            threshold_cosine=data.get("threshold_cosine", 0.80),
            threshold_localization=data.get("threshold_localization", 0.70),
        )

    def save(self, path: Path):
        data = {
            "atom_indices": self.atom_indices,
            "ref_mode": self.ref_mode.tolist(),
            "mode_label": self.mode_label,
            "threshold_cosine": self.threshold_cosine,
            "threshold_localization": self.threshold_localization,
        }
        Path(path).write_text(json.dumps(data, indent=2))


# -------------------------------------------------------------------------
# Mode comparison
# -------------------------------------------------------------------------
@dataclass
class ComparisonResult:
    cosine: float
    localization: float
    passed: bool


def compare_modes(candidate_modes: np.ndarray, fingerprint: Fingerprint) -> ComparisonResult:
    """
    Compare a set of candidate normal modes to the stored fingerprint.

    Parameters
    ----------
    candidate_modes : np.ndarray
        Mode vectors (shape: n_modes x n_atoms x 3), assumed mass-weighted.
    fingerprint : Fingerprint
        Reference fingerprint object.

    Returns
    -------
    ComparisonResult
    """
    # Flatten reference mode
    ref = fingerprint.ref_mode.flatten()
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-8:
        raise ValueError("Reference fingerprint mode is zero-length")

    # Compute cosine overlap vs each imaginary mode
    cosines = []
    for mode in candidate_modes:
        c = np.dot(ref, mode.flatten()) / (ref_norm * np.linalg.norm(mode))
        cosines.append(c)
    best_idx = int(np.argmax(np.abs(cosines)))
    best_cos = float(cosines[best_idx])

    # Localization: fraction of displacement on key atoms
    mode = candidate_modes[best_idx]
    disp_sq = np.sum(mode**2, axis=1)
    total = np.sum(disp_sq)
    local = np.sum(disp_sq[fingerprint.atom_indices]) / total if total > 1e-12 else 0.0

    passed = (abs(best_cos) >= fingerprint.threshold_cosine) and (local >= fingerprint.threshold_localization)
    return ComparisonResult(cosine=best_cos, localization=local, passed=passed)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def load_fingerprint(path: Path | None) -> Optional[Fingerprint]:
    if path is None:
        return None
    return Fingerprint.load(path)


def save_fingerprint(atom_indices: list[int], ref_mode: np.ndarray,
                     path: Path, mode_label: str | None = None,
                     threshold_cosine: float = 0.8,
                     threshold_localization: float = 0.7):
    fp = Fingerprint(
        atom_indices=atom_indices,
        ref_mode=ref_mode,
        mode_label=mode_label,
        threshold_cosine=threshold_cosine,
        threshold_localization=threshold_localization,
    )
    fp.save(path)
    return fp
