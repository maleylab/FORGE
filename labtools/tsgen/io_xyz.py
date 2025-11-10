from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np


def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    with open(path, "r") as fh:
        lines = [l.rstrip() for l in fh]
    try:
        n = int(lines[0].split()[0])
    except Exception as e:
        raise ValueError(f"{path} is not a valid XYZ: first line must have atom count") from e
    body = lines[2 : 2 + n]
    elems, coords = [], []
    for i, ln in enumerate(body):
        toks = ln.split()
        if len(toks) < 4:
            raise ValueError(f"Malformed XYZ line {i+3} in {path}")
        elems.append(toks[0])
        coords.append([float(toks[1]), float(toks[2]), float(toks[3])])
    return elems, np.array(coords, dtype=float)


def write_xyz(path: Path, elems: List[str], coords: np.ndarray, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{len(elems)}\n")
        fh.write(comment + "\n")
        for s, (x, y, z) in zip(elems, coords):
            fh.write(f"{s:2s} {x: .8f} {y: .8f} {z: .8f}\n")
