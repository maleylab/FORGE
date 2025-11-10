from __future__ import annotations
import numpy as np
from typing import List

# vdW radii (minimal)
_VDW = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "Fe": 2.00,
    "Ru": 2.10,
    "Ni": 1.97,
    "Co": 2.00,
    "Mn": 2.00,
    "Cu": 1.96,
}


def min_vdw_scale(elems, coords: np.ndarray) -> float:
    n = len(elems)
    m = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            r = _VDW.get(elems[i], 1.8) + _VDW.get(elems[j], 1.8)
            m = min(m, d / r)
    return m


def local_linear_frames(A: np.ndarray, B: np.ndarray, mask: np.ndarray, n: int, mass_weighted: bool = False, masses=None) -> List[np.ndarray]:
    delta = (B - A) * mask[:, None]
    if mass_weighted:
        if masses is None:
            raise ValueError("mass_weighted=True requires masses array")
        w = np.sqrt(masses)[:, None]
        delta = delta * w / (w + 1e-12)
    return [A + t * delta for t in np.linspace(0.0, 1.0, n)]


def idpp_smooth_frames(frames: List[np.ndarray], mask: np.ndarray, n_iter: int = 200, step: float = 0.05) -> List[np.ndarray]:
    if len(frames) <= 2:
        return frames
    A, B = frames[0], frames[-1]
    mov = np.where(mask.astype(bool))[0]

    def pair_dists(X):
        Xm = X[mov]
        diff = Xm[:, None, :] - Xm[None, :, :]
        return np.linalg.norm(diff, axis=2)

    D_A = pair_dists(A)
    D_B = pair_dists(B)
    out = [A]
    for k in range(1, len(frames) - 1):
        X = frames[k].copy()
        t = k / (len(frames) - 1)
        D_t = (1 - t) * D_A + t * D_B
        for _ in range(n_iter):
            D_X = pair_dists(X)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_D = np.where(D_X > 1e-6, 1.0 / D_X, 0.0)
                inv_Dt = np.where(D_t > 1e-6, 1.0 / D_t, 0.0)
            F = (inv_D - inv_Dt) * (inv_D ** 2)
            Xm = X[mov]
            diff = Xm[:, None, :] - Xm[None, :, :]
            unit = diff / (np.linalg.norm(diff, axis=2, keepdims=True) + 1e-12)
            fm = (F[..., None] * unit).sum(axis=1) - (F[..., None] * unit).sum(axis=0)
            X[mov] -= step * fm
        out.append(X)
    out.append(B)
    return out


