"""
TSGen 2.1 | Fingerprint utilities (index-robust)
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import yaml

from ..tsgen_orca import (
    parse_frequencies_and_modes,
    read_final_xyz,
)

# ======================================================================
# Dataclass
# ======================================================================

@dataclass
class Fingerprint:
    ref_mode: np.ndarray                 # (Nref,3) normalized
    atom_indices: list                   # indices in fingerprint space
    threshold_cosine: float
    threshold_localization: float
    atom_map: dict | None = None         # fp_index → system_index


# ======================================================================
# Load fingerprint
# ======================================================================

def load_fingerprint(path: Path) -> Fingerprint:
    data = yaml.safe_load(Path(path).read_text())

    if "ref_mode" in data:
        ref = np.array(data["ref_mode"], float)
    elif "vector" in data:
        vec = np.array(data["vector"], float)
        if vec.size % 3 != 0:
            raise ValueError("Fingerprint vector length not divisible by 3")
        ref = vec.reshape(vec.size // 3, 3)
    else:
        raise KeyError("Fingerprint must define ref_mode or vector")

    n = np.linalg.norm(ref)
    if n == 0:
        raise ValueError("Reference mode is zero")
    ref /= n

    thresholds = data.get("thresholds", {})

    atom_map = data.get("atom_map", None)
    if atom_map is not None:
        atom_map = {int(k): int(v) for k, v in atom_map.items()}

    atom_indices = list(data.get("atom_indices", []))
    if len(atom_indices) != ref.shape[0]:
        raise ValueError(
            "Fingerprint atom_indices length does not match ref_mode"
        )

    return Fingerprint(
        ref_mode=ref,
        atom_indices=atom_indices,
        threshold_cosine=float(
            thresholds.get("min_cosine", data.get("threshold_cosine", 0.5))
        ),
        threshold_localization=float(
            thresholds.get("min_localization", data.get("threshold_localization", 0.3))
        ),
        atom_map=atom_map,
    )


# ======================================================================
# Extract TS mode with canonical / mapped contracts
# ======================================================================

def extract_ts_mode_from_orca(out_path: Path, fingerprint: Fingerprint | None = None):
    out_path = Path(out_path).expanduser().resolve()

    geo = read_final_xyz(out_path)
    if geo is None:
        raise ValueError(f"Could not extract geometry from {out_path}")

    atoms, coords = geo
    n_atoms = len(atoms)

    freqs, modes = parse_frequencies_and_modes(out_path)
    if freqs is None or modes is None:
        raise ValueError(f"No vibrational modes found in {out_path}")

    # Locate imaginary mode
    if isinstance(freqs, dict):
        imag = [k for k, v in freqs.items() if v < 0]
        if not imag:
            raise ValueError(f"No imaginary modes found in {out_path}")
        imag_idx = min(imag, key=lambda k: abs(freqs[k]))
        mode = np.array(modes[imag_idx], float)
    else:
        imag = [i for i, f in enumerate(freqs) if f < 0]
        if not imag:
            raise ValueError(f"No imaginary modes found in {out_path}")
        imag_idx = min(imag, key=lambda i: abs(freqs[i]))
        mode = np.array(modes[imag_idx], float)

    if mode.shape != (n_atoms, 3):
        raise ValueError("Mode/geometry size mismatch")

    # Normalize full mode
    norm = np.linalg.norm(mode)
    if norm == 0:
        raise ValueError("Imaginary mode has zero norm")
    mode /= norm

    # --------------------------------------------------
    # Apply fingerprint (if present)
    # --------------------------------------------------
    if fingerprint is not None:
        # Determine system indices
        if fingerprint.atom_map is None:
            # Contract A: canonical ordering
            indices = fingerprint.atom_indices
        else:
            # Contract B: explicit map
            indices = []
            for fp_idx in fingerprint.atom_indices:
                if fp_idx not in fingerprint.atom_map:
                    raise KeyError(
                        f"Fingerprint atom {fp_idx} missing from atom_map"
                    )
                sys_idx = fingerprint.atom_map[fp_idx]
                if sys_idx < 0 or sys_idx >= n_atoms:
                    raise IndexError(
                        f"Mapped system index {sys_idx} out of bounds"
                    )
                indices.append(sys_idx)

        # Slice and renormalize
        mode = mode[indices, :]
        coords = coords[indices, :]

        sub_norm = np.linalg.norm(mode)
        if sub_norm == 0:
            raise ValueError(
                "Mapped TS mode has zero norm; fingerprint atoms "
                "do not describe this imaginary mode"
            )
        mode /= sub_norm

        # Align sign to reference
        if np.dot(fingerprint.ref_mode.reshape(-1), mode.reshape(-1)) < 0:
            mode = -mode

    return coords, mode


# ======================================================================
# Metrics
# ======================================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    a = v1.reshape(-1)
    b = v2.reshape(-1)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def mode_localization_fraction(mode: np.ndarray, fingerprint: Fingerprint) -> float:
    if fingerprint is None:
        return 0.0

    total = np.sum(mode * mode)
    if total == 0:
        return 0.0

    # After extract_ts_mode_from_orca, mode is already restricted
    # to fingerprint atoms
    return float(total)
