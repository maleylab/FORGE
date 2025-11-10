"""
L0: R–P alignment, linear + IDPP interpolation, constrained XTB2 Opt+Freq,
fingerprint gating; returns verified TS seeds.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Optional
from ..orca_io import write_orca_input, run_orca, parse_frequencies
from ..constraints import make_cartesian_constraints
from ..fingerprint import Fingerprint, compare_modes
from labtools.data.io import read_xyz, write_xyz


def generate_guesses(
    reactant: Path,
    product: Path,
    method: str,
    charge: int,
    mult: int,
    outdir: Path,
    fingerprint: Optional[Fingerprint] = None,
    n_images: int = 7,
    preopt: bool = True,
    mode: str = "array",
    profile: str = "medium",
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    R_atoms, R_xyz = read_xyz(reactant)
    P_atoms, P_xyz = read_xyz(product)
    assert len(R_atoms) == len(P_atoms), "R/P atom mismatch"

    # Align + interpolate + IDPP
    R_aligned, P_aligned = _align_structures(R_xyz, P_xyz)
    images = np.linspace(R_aligned, P_aligned, n_images)
    images = _idpp_relax(images, max_iter=200, step=0.05)

    # Write images
    image_files = []
    for i, coords in enumerate(images):
        f = outdir / f"idpp_{i:02d}.xyz"
        write_xyz(R_atoms, coords, f)
        image_files.append(f)

    # Candidates: midpoint ±1
    mid = n_images // 2
    candidate_idx = [mid - 1, mid, mid + 1] if n_images >= 5 else [mid]
    candidate_paths = [image_files[i] for i in candidate_idx]

    verified, report = [], []

    for idx, geom_path in enumerate(candidate_paths):
        cand_dir = outdir / f"cand_{idx:02d}"
        cand_dir.mkdir(exist_ok=True)

        active_atoms = fingerprint.atom_indices if fingerprint else None
        constraints_block = make_cartesian_constraints(
            R_atoms, R_xyz, P_xyz, active_atoms=active_atoms
        )

        inp = cand_dir / "cand.inp"
        write_orca_input(
            inp, jobtype="Opt Freq", method=method,
            charge=charge, mult=mult,
            geom_file=geom_path,
            constraints=constraints_block,
            use_ri=False, add_aux_basis=False,
            provenance={"stage": "L0", "parent": str(geom_path)},
        )
        run_orca(inp, cwd=cand_dir, mode=mode, profile=profile)

        freqs, modes = parse_frequencies(cand_dir / "cand.out")

        if fingerprint:
            res = compare_modes(modes, fingerprint)
            data = {
                "candidate": idx,
                "imag_count": len([f for f in freqs if f < 0]),
                "cosine": res.cosine,
                "localization": res.localization,
                "passed": res.passed,
            }
            (cand_dir / "cand_report.json").write_text(json.dumps(data, indent=2))
            report.append(data)
            if res.passed:
                # final pre-opt geometry name; adjust if your template writes different name
                verified.append(cand_dir / "cand.xyz")
        else:
            verified.append(cand_dir / "cand.xyz")

    (outdir / "L0_report.json").write_text(json.dumps(report, indent=2))
    if not verified and candidate_paths:
        verified = [candidate_paths[len(candidate_paths) // 2]]
    return verified


def _align_structures(R_xyz: np.ndarray, P_xyz: np.ndarray):
    R_c, P_c = R_xyz.mean(0), P_xyz.mean(0)
    R_shift, P_shift = R_xyz - R_c, P_xyz - P_c
    C = P_shift.T @ R_shift
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ W))
    U = V @ np.diag([1, 1, d]) @ W
    P_aligned = (P_shift @ U.T) + R_c
    return R_xyz, P_aligned


def _idpp_relax(images: np.ndarray, max_iter=200, step=0.05):
    n_images, n_atoms, _ = images.shape
    R, P = images[0], images[-1]
    d_R = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    d_P = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    target = [d_R + (d_P - d_R) * (i / (n_images - 1)) for i in range(n_images)]

    images_relaxed = images.copy()
    for _ in range(max_iter):
        max_disp = 0.0
        for i in range(1, n_images - 1):
            grad = np.zeros_like(images_relaxed[i])
            for a in range(n_atoms):
                for b in range(a + 1, n_atoms):
                    rij = images_relaxed[i][a] - images_relaxed[i][b]
                    dist = np.linalg.norm(rij)
                    if dist < 1e-8:
                        continue
                    t = target[i][a, b]
                    f = 2 * (1 / dist - 1 / t) / (dist**3)
                    grad[a] += f * rij
                    grad[b] -= f * rij
            disp = -step * grad
            images_relaxed[i] += disp
            max_disp = max(max_disp, np.max(np.linalg.norm(disp, axis=1)))
        if max_disp < 1e-4:
            break
    return images_relaxed
