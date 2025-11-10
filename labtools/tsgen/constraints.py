"""
constraints.py
FORGE | tsgen

Generates ORCA-compatible constraint blocks for geometry optimizations.

Used primarily in L0 (constrained XTB2 preopts) but general enough for
higher levels. Supports both Cartesian and internal (distance/angle)
constraints, with optional weighting for "soft" enforcement.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence


def make_cartesian_constraints(
    atoms: Sequence[str],
    R_xyz: np.ndarray,
    P_xyz: np.ndarray,
    active_atoms: Optional[Sequence[int]] = None,
    soft: bool = False,
    weight: float = 0.5,
) -> str:
    """
    Build a %geom Constraints block for ORCA.

    Parameters
    ----------
    atoms : list[str]
        Atom symbols (not used directly but may be helpful later)
    R_xyz, P_xyz : np.ndarray
        Reactant and product coordinates (Å)
    active_atoms : list[int], optional
        Indices of atoms to constrain (0-based). If None, choose all atoms
        with |Δr| > 0.25 Å between R and P.
    soft : bool, optional
        Whether to apply "soft" weighting (restrained instead of fixed)
    weight : float, optional
        Force constant for soft constraints (arbitrary units)

    Returns
    -------
    str
        Text block to include in ORCA input under %geom Constraints
    """
    if active_atoms is None:
        disp = np.linalg.norm(P_xyz - R_xyz, axis=1)
        active_atoms = [i for i, d in enumerate(disp) if d > 0.25]

    if not active_atoms:
        return ""  # no constraints needed

    lines = ["%geom", "  Constraints"]
    if soft:
        for idx in active_atoms:
            lines.append(f"    {{ C {idx+1} C C C {weight:.3f} }}  # soft Cartesian constraint")
    else:
        for idx in active_atoms:
            lines.append(f"    {{ C {idx+1} C C C }}  # fix atom {idx+1}")
    lines += ["  end", "end"]
    return "\n".join(lines)


# -------------------------------------------------------------------------
# Distance constraint builder (for future use)
# -------------------------------------------------------------------------
def make_distance_constraints(
    pairs: list[tuple[int, int]],
    R_xyz: np.ndarray,
    P_xyz: np.ndarray,
    weight: Optional[float] = None,
) -> str:
    """
    Build simple distance constraints for selected atom pairs.
    Useful for following a reaction coordinate or restraining a bond.

    Returns
    -------
    str : ORCA-compatible Constraints block
    """
    lines = ["%geom", "  Constraints"]
    for (i, j) in pairs:
        dR = np.linalg.norm(R_xyz[i] - R_xyz[j])
        dP = np.linalg.norm(P_xyz[i] - P_xyz[j])
        d_avg = 0.5 * (dR + dP)
        if weight is not None:
            lines.append(f"    {{ B {i+1} {j+1} C {d_avg:.4f} {weight:.3f} }}")
        else:
            lines.append(f"    {{ B {i+1} {j+1} C {d_avg:.4f} }}")
    lines += ["  end", "end"]
    return "\n".join(lines)
