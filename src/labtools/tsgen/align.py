from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np

def read_mapping_csv(path: Path) -> List[Tuple[int, int]]:
    pairs = []
    for ln in path.read_text().strip().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        a, b = ln.replace(",", " ").split()[:2]
        pairs.append((int(a), int(b)))
    return pairs

def reorder_by_map(elemsA, coordsA, elemsB, coordsB, pairs: List[Tuple[int, int]]):
    n = len(elemsA)
    B2 = np.zeros_like(coordsB)
    elemsB2 = [None] * n
    for a, b in pairs:
        B2[a] = coordsB[b]
        elemsB2[a] = elemsB[b]
    if any(e is None for e in elemsB2):
        raise ValueError("Mapping CSV does not cover all atoms")
    return elemsA, coordsA, elemsB2, B2

def kabsch_align(B: np.ndarray, A: np.ndarray):
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    H = Bc.T @ Ac
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = A.mean(axis=0) - (B.mean(axis=0) @ R)
    return R, t

def apply_rot_trans(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return X @ R + t

def choose_anchor_triplet(reactant_indices: List[int], coords: np.ndarray) -> Tuple[int, int, int]:
    import itertools

    best = None
    best_area = -1.0
    for i, j, k in itertools.combinations(reactant_indices, 3):
        A = coords[i]
        B = coords[j]
        C = coords[k]
        area = float(np.linalg.norm(np.cross(B - A, C - A)) / 2.0)
        if area > best_area:
            best_area = area
            best = (i, j, k)
    if best is None:
        r = reactant_indices + reactant_indices[:3]
        best = (r[0], r[1], r[2])
    return best

def rotate_eigenvector(vec: np.ndarray, R: np.ndarray) -> np.ndarray:
    if vec.ndim != 2 or vec.shape[1] != 3:
        raise ValueError("Eigenvector must have shape (n_atoms, 3)")
    return vec @ R

def align_structures(
    ref_xyz: np.ndarray,
    new_xyz: np.ndarray,
    atom_subset: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    if ref_xyz.shape != new_xyz.shape:
        raise ValueError("ref_xyz and new_xyz must have same shape")

    if len(atom_subset) < 3:
        raise ValueError("atom_subset requires >= 3 atoms for stable alignment")

    i, j, k = choose_anchor_triplet(atom_subset, ref_xyz)

    A = ref_xyz[[i, j, k]]
    B = new_xyz[[i, j, k]]

    R, t = kabsch_align(B, A)
    return R, t
