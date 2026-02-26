from __future__ import annotations
import numpy as np
from typing import List

def normalize_vector(v: np.ndarray) -> np.ndarray:
    v2 = v.reshape(-1)
    norm = np.linalg.norm(v2)
    if norm < 1e-15:
        return v
    return (v2 / norm).reshape(v.shape)

def cosine_similarity(ref: np.ndarray, cand: np.ndarray) -> float:
    a = ref.reshape(-1)
    b = cand.reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return abs(cos)

def localization_score(mode: np.ndarray, subset: List[int]) -> float:
    if mode.ndim != 2 or mode.shape[1] != 3:
        raise ValueError("mode must have shape (n_atoms, 3)")
    disp_sq = np.sum(mode * mode, axis=1)
    total = float(np.sum(disp_sq))
    if total < 1e-15:
        return 0.0
    return float(np.sum(disp_sq[subset]) / total)
