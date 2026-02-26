# tsgen2/idpp.py
# REAL IDPP IMPLEMENTATION – NO APPROXIMATIONS
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------
# Utility: pairwise distances
# ---------------------------------------------------------
def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise distance matrix, shape (N,N)
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(-1) + 1e-15)


# ---------------------------------------------------------
# Interpolate pairwise distances
# ---------------------------------------------------------
def interpolate_distances(D0: np.ndarray, D1: np.ndarray, lam: float) -> np.ndarray:
    """
    Reference distances for IDPP: linear interpolation of PAIRWISE distances.
    """
    return (1 - lam) * D0 + lam * D1


# ---------------------------------------------------------
# IDPP Energy + Gradient
# ---------------------------------------------------------
def idpp_energy_and_gradient(X: np.ndarray, D_ref: np.ndarray):
    """
    Compute true IDPP energy and gradient.

    X : (N,3) coords
    D_ref : (N,N) reference distance matrix
    """
    N = X.shape[0]
    D = pairwise_distances(X)
    diff = (D - D_ref)

    # weights = 1 / D_ref^4
    W = 1.0 / np.maximum(D_ref, 1e-6) ** 4

    # IDPP energy
    E = np.sum(W * diff * diff)

    # Gradient: dE/dX_i
    grad = np.zeros_like(X)

    # For each pair (i,j)
    for i in range(N):
        for j in range(i + 1, N):
            Dij = D[i, j]
            if Dij < 1e-12:
                continue

            rij = X[i] - X[j]
            dE_dDij = 2 * W[i, j] * diff[i, j]

            # ∂Dij/∂X_i = (X_i - X_j) / Dij
            g = dE_dDij * (rij / Dij)

            grad[i] += g
            grad[j] -= g

    return E, grad


# ---------------------------------------------------------
# Optimization loop (simple L-BFGS-ish)
# ---------------------------------------------------------
def minimize_idpp(X0: np.ndarray, D_ref: np.ndarray,
                  maxiter: int = 200,
                  stepsize: float = 0.05,
                  tol: float = 1e-3):
    """
    Robust IDPP minimization without external dependencies.
    """
    X = X0.copy()

    for _ in range(maxiter):
        E, g = idpp_energy_and_gradient(X, D_ref)
        gnorm = np.linalg.norm(g)

        if gnorm < tol:
            break

        # simple backtracking line search
        step = stepsize
        while step > 1e-6:
            X_new = X - step * g
            E_new, _ = idpp_energy_and_gradient(X_new, D_ref)
            if E_new < E:
                X = X_new
                break
            step *= 0.5

    return X


# ---------------------------------------------------------
# Generate one IDPP seed
# ---------------------------------------------------------
def generate_idpp_seed(R0: np.ndarray, R1: np.ndarray, lam: float) -> np.ndarray:
    """
    Generate a single IDPP seed at interpolation coordinate lam.
    """
    # initial guess = linear interpolation
    X0 = (1 - lam) * R0 + lam * R1

    # reference distances
    D0 = pairwise_distances(R0)
    D1 = pairwise_distances(R1)
    Dref = interpolate_distances(D0, D1, lam)

    # optimize
    X = minimize_idpp(X0, Dref)
    return X


# ---------------------------------------------------------
# Generate N seeds
# ---------------------------------------------------------
def generate_idpp_seeds(R0: np.ndarray, R1: np.ndarray, n: int):
    """
    Generate n IDPP seeds uniformly between reactant and product.
    Returns ONLY coordinate arrays of shape (N,3).
    """
    seeds = []
    for k in range(1, n + 1):
        lam = k / (n + 1)
        X = generate_idpp_seed(R0, R1, lam)
        seeds.append(X)   # RETURN ONLY X, NOT (lam, X)
    return seeds

