from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Optional, Dict, Any
import yaml
import numpy as np

from .orca_helper import (
    parse_geometry_xyz,
    select_atoms_mode6,
    has_mode6,
)

class VerifyResult:
    """
    Backward-compatible result object.
    Accepts legacy positional forms like:
      VerifyResult(ok)
      VerifyResult(ok, rmsd, cos)
      VerifyResult(ok, rmsd, cos, details)
      VerifyResult(ok, rmsd, cos, ref_meta, cand_meta, details)
    and any variants that stuff extra dicts before/after details.
    """
    def __init__(self,
                 ok: bool,
                 rmsd: Optional[float] = None,
                 cos_sim: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None,
                 *extras,
                 **kw):
        self.ok = bool(ok)
        self.rmsd = rmsd
        self.cos_sim = cos_sim

        # Start with provided details or an empty dict
        merged: Dict[str, Any] = {}
        if isinstance(details, dict):
            merged.update(details)

        # Fold in any extra positional dict-like payloads (legacy calls)
        # Common legacy pattern: (ok, rmsd, cos, ref_meta, cand_meta, details)
        # We'll tuck dict-looking extras under names if we can guess; otherwise into 'extras'
        named_slots = ["ref", "cand"]
        slot_i = 0
        loose_extras = []
        for ex in extras:
            if isinstance(ex, dict):
                if slot_i < len(named_slots) and not any(k in merged for k in (named_slots[slot_i],)):
                    merged[named_slots[slot_i]] = ex
                    slot_i += 1
                else:
                    # merge if looks like details, else append
                    merged.update(ex)
            else:
                loose_extras.append(ex)

        if kw:
            merged.update(kw)
        if loose_extras:
            merged.setdefault("extras", []).extend(loose_extras)

        self.details = merged

    # Back-compat for code expecting `.passed`
    @property
    def passed(self) -> bool:
        return bool(self.ok)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "passed": self.ok,
            "rmsd": self.rmsd,
            "cos_sim": self.cos_sim,
            "details": self.details or {},
        }

def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # P, Q: (K,3) centered. Returns rotation R (3x3) s.t. Q@R ≈ P
    C = Q.T @ P
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def verify_against_reference(
    reference_yaml: Path | str,
    new_out: Path | str,
    threshold: float = 0.90,
) -> VerifyResult:
    """
    Compare Mode-6 vectors on selected atoms between reference fingerprint YAML
    and a new ORCA .out. Geometry is rigid-aligned (Kabsch) before comparing vectors.
    """
    ref_path = Path(reference_yaml)
    new_out = Path(new_out)

    # Preflight: ensure Mode 6 exists
    if not has_mode6(new_out):
        return VerifyResult(False, 0.0, 0, threshold,
                            "Mode-6 column not found in output (no NORMAL MODES page with '6 7 8 9 10 11').")

    # Load ref fingerprint
    data = yaml.safe_load(ref_path.read_text())
    atom_idx: Sequence[int] = list(map(int, data["meta"]["atom_indices"]))
    ref_vecs = np.asarray(data["selected_mode6"], dtype=float)      # (K,3)
    ref_geom_sel = np.asarray(data["selected_geometry"], dtype=float)  # (K,3)

    # New geometry + selected Mode-6
    new_geom_all, _ = parse_geometry_xyz(new_out)
    new_vecs_sel = select_atoms_mode6(new_out, atom_idx)  # (K,3)
    new_geom_sel = new_geom_all[np.asarray(atom_idx, dtype=int), :]

    # Center both geoms and compute best-fit rotation to align NEW onto REF
    P = ref_geom_sel - ref_geom_sel.mean(axis=0, keepdims=True)
    Q = new_geom_sel - new_geom_sel.mean(axis=0, keepdims=True)
    R = _kabsch(P, Q)  # rotate NEW → REF frame

    # Rotate the NEW displacement vectors by the same R
    new_vecs_rot = (new_vecs_sel @ R)

    # Compare per-atom cosine similarity; score = mean cosine
    cosines = [ _cosine(ref_vecs[i], new_vecs_rot[i]) for i in range(len(atom_idx)) ]
    score = float(np.mean(cosines))
    ok = bool(score >= threshold)

    msg = f"mean cosine={score:.3f} over K={len(atom_idx)} atoms; threshold={threshold:.2f}"
    return VerifyResult(ok, score, len(atom_idx), threshold, msg)
