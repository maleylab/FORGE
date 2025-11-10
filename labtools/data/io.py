# src/labtools/data/io.py
from __future__ import annotations

"""
Lightweight I/O utilities used across FORGE.

Included:
- jsonl_append(path, rec)
- jsonl_to_parquet(jsonl_path, parquet_path)
- read_xyz(path) -> (atoms: list[str], coords: np.ndarray[n,3])
- write_xyz(atoms, coords, path, comment="")
- write_xyz_multi(frames, path, comment_per_frame=None)
- guess_xyz_atom_count(path) -> int
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSONL utilities (existing)
# ---------------------------------------------------------------------------

def jsonl_append(path: str | Path, rec: Dict[str, Any]) -> None:
    """
    Append a JSON object to a .jsonl (JSON Lines) file, creating parent dirs.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def jsonl_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> None:
    """
    Convert a .jsonl file to Apache Parquet.
    """
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(str(jsonl_path))
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
# XYZ geometry I/O
# ---------------------------------------------------------------------------

def read_xyz(path: str | Path) -> Tuple[List[str], np.ndarray]:
    """
    Read a single-structure XYZ file.

    Supports:
      - Strict XYZ (first line = integer atom count; second line = comment)
      - Loose XYZ (no count line; each non-empty line is 'El x y z')

    Returns
    -------
    atoms : list[str]
    coords : (n_atoms, 3) np.ndarray (Å)
    """
    p = Path(path)
    lines = p.read_text().splitlines()

    # Strip trailing blanks
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        raise ValueError(f"Empty XYZ file: {p}")

    first = lines[0].strip()
    strict = first.isdigit()

    if strict:
        n_atoms = int(first)
        if len(lines) < 2 + n_atoms:
            raise ValueError(f"XYZ too short for {n_atoms} atoms: {p}")
        body = lines[2:2 + n_atoms]
        # Tolerate only blank lines after one frame
        trailing = lines[2 + n_atoms:]
        if any(s.strip() for s in trailing):
            raise ValueError(f"Multiple XYZ frames detected in {p}; expected one.")
    else:
        # Treat all non-empty lines as coordinates
        body = [ln for ln in lines if ln.strip()]
        # Guard against accidental strict files that weren't parsed as such
        if body and body[0].strip().isdigit():
            raise ValueError(f"Ambiguous XYZ format for {p}.")

    atoms: List[str] = []
    coords: List[List[float]] = []
    for i, ln in enumerate(body):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line {i} in {p}: {ln!r}")
        atom = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except Exception as e:
            raise ValueError(f"Non-numeric coordinate on line {i} in {p}: {ln!r}") from e
        atoms.append(atom)
        coords.append([x, y, z])

    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"XYZ coords must be (n,3); got {arr.shape} for {p}")

    return atoms, arr


def write_xyz(
    atoms: Iterable[str],
    coords: np.ndarray,
    path: str | Path,
    comment: str = "",
) -> None:
    """
    Write a strict single-frame XYZ file.

    Parameters
    ----------
    atoms : iterable[str]
    coords : (n,3) array-like
    path   : output file path
    comment: optional comment line
    """
    p = Path(path)
    atoms_list = list(atoms)
    arr = np.asarray(coords, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"`coords` must be (n,3); got {arr.shape}")
    if len(atoms_list) != arr.shape[0]:
        raise ValueError(f"{len(atoms_list)} atoms but coords.shape={arr.shape}")

    p.parent.mkdir(parents=True, exist_ok=True)
    comment_line = (comment or "").replace("\n", " ").strip()

    with p.open("w", encoding="utf-8") as f:
        f.write(f"{len(atoms_list)}\n")
        f.write(f"{comment_line}\n")
        for a, (x, y, z) in zip(atoms_list, arr):
            f.write(f"{a:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def write_xyz_multi(
    frames: Iterable[tuple[Iterable[str], np.ndarray]],
    path: str | Path,
    comment_per_frame: Iterable[str] | None = None,
) -> None:
    """
    Write a multi-frame XYZ by concatenating strict XYZ blocks.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    comments = list(comment_per_frame) if comment_per_frame is not None else None
    with p.open("w", encoding="utf-8") as f:
        for i, (atoms, coords) in enumerate(frames):
            cmt = comments[i] if comments and i < len(comments) else ""
            atoms_list = list(atoms)
            arr = np.asarray(coords, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"`coords` must be (n,3); got {arr.shape} (frame {i})")
            if len(atoms_list) != arr.shape[0]:
                raise ValueError(f"{len(atoms_list)} atoms vs coords {arr.shape} (frame {i})")
            f.write(f"{len(atoms_list)}\n{cmt.replace(chr(10),' ').strip()}\n")
            for a, (x, y, z) in zip(atoms_list, arr):
                f.write(f"{a:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def guess_xyz_atom_count(path: str | Path) -> int:
    """
    If file is strict XYZ, return the atom count; else -1.
    """
    p = Path(path)
    try:
        first = p.read_text().splitlines()[0].strip()
    except Exception:
        return -1
    return int(first) if first.isdigit() else -1
