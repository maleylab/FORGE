from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Sequence
import re
import time
import numpy as np
import yaml

# --------------------------
# ORCA block regexes
# --------------------------
_COORDS_HEAD_RE = re.compile(r"^\s*CARTESIAN COORDINATES\s*\(ANGSTROEM\)\s*$")
_COORDS_SEP_RE = re.compile(r"^-{5,}\s*$")

# We only need the page whose header shows columns "... 6 7 8 9 10 11"
# Example:
#                   6          7          8          9         10         11
_MODES_PAGE_6TO11_RE = re.compile(r"\b6\s+7\s+8\s+9\s+10\s+11\b")

# A numeric data row under that header looks like:
#       0       0.308846   0.024439   0.080534  -0.101673  -0.112280   0.180447
# i.e., <row_index> <six floats>
_ROW_SPLIT_RE = re.compile(r"\s+")


# --------------------------
# File reading
# --------------------------
def _read_lines(path: Path) -> List[str]:
    return Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()


# --------------------------
# Geometry parsing
# --------------------------
def parse_geometry_xyz(out_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Parse 'CARTESIAN COORDINATES (ANGSTROEM)' → (N,3) array and list of symbols.
    """
    lines = _read_lines(out_path)
    n = len(lines)
    i = 0
    while i < n:
        if _COORDS_HEAD_RE.match(lines[i]):
            # skip header line and the dashed separator line
            i += 1
            while i < n and not _COORDS_SEP_RE.match(lines[i]):
                i += 1
            if i >= n:
                break
            i += 1  # first data line after dashes
            coords: List[List[float]] = []
            symbols: List[str] = []
            while i < n:
                s = lines[i].strip()
                if not s or _COORDS_SEP_RE.match(s):
                    break
                parts = s.split()
                if len(parts) < 4:
                    break
                sym = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    break
                symbols.append(sym)
                coords.append([x, y, z])
                i += 1
            if coords:
                return np.asarray(coords, dtype=float), symbols
        i += 1
    raise ValueError("Failed to locate 'CARTESIAN COORDINATES (ANGSTROEM)' block.")


def extract_geometry_xyz(out_path: str | Path) -> Tuple[List[str], List[List[float]]]:
    """
    Return (symbols, coords) from ORCA 'CARTESIAN COORDINATES (ANGSTROEM)'.
    coords is a list of [x, y, z] in Å.
    """
    arr, syms = parse_geometry_xyz(Path(out_path))
    return syms, arr.tolist()


# --------------------------
# Mode-6 extraction
# --------------------------
def _locate_mode6_header(lines: List[str]) -> Optional[int]:
    """
    Find line index of the header line that contains '6 7 8 9 10 11'.
    Returns None if not found.
    """
    for i, ln in enumerate(lines):
        if _MODES_PAGE_6TO11_RE.search(ln):
            return i
    return None


def parse_mode6_column(out_path: str | Path) -> List[float]:
    """
    Parse ONLY the Mode-6 displacement column from the 'NORMAL MODES' table.
    Returns a flat list of length 3N (x,y,z per atom).
    Assumes the page that contains headers '6 7 8 9 10 11'.
    """
    p = Path(out_path)
    lines = _read_lines(p)

    # We need N (atoms) to know how many rows to collect (3N).
    _, coords = extract_geometry_xyz(p)
    N = len(coords)
    required_rows = 3 * N

    header_idx = _locate_mode6_header(lines)
    if header_idx is None:
        raise ValueError("Mode-6 header page not found")

    out_vals: List[float] = []
    i = header_idx + 1
    while i < len(lines) and len(out_vals) < required_rows:
        s = lines[i].strip()
        if not s:
            # blank line → most likely end of page
            break
        parts = _ROW_SPLIT_RE.split(s)
        # expect: index + 6 numeric columns
        if len(parts) < 2:
            i += 1
            continue
        # parts[0] is the row index; parts[1] is the value for column "6"
        try:
            val = float(parts[1])
        except Exception:
            # hit a non-numeric or next header → stop
            break
        out_vals.append(val)
        i += 1

    if len(out_vals) != required_rows:
        raise ValueError(f"Mode-6 column length mismatch: got {len(out_vals)}, expected {required_rows}")
    return out_vals


def has_mode6(out_path: str | Path) -> bool:
    """
    Quick preflight: does this output contain a usable Mode-6 column?
    """
    try:
        _ = parse_mode6_column(out_path)
        return True
    except Exception:
        return False


# --------------------------
# Fingerprint conveniences
# --------------------------
def select_atoms_mode6(out_path: str | Path, atom_indices: Sequence[int]) -> np.ndarray:
    """
    Return the Mode-6 displacement vectors for selected atoms as (K,3),
    picking rows [3*i:3*i+3] per atom i.
    """
    flat = parse_mode6_column(out_path)  # (3N,)
    syms, coords = extract_geometry_xyz(out_path)
    N = len(coords)
    if len(flat) != 3 * N:
        raise ValueError("Internal error: 3N mismatch when reshaping Mode-6 column.")
    mat = np.asarray(flat, dtype=float).reshape(N, 3)
    idx = np.asarray(list(atom_indices), dtype=int)
    if (idx < 0).any() or (idx >= N).any():
        raise ValueError(f"Atom index out of range (N={N}): {idx.tolist()}")
    return mat[idx, :].copy()


def write_fingerprint_yaml(
    out_path: Path | str,
    atom_indices: Sequence[int],
    dst_path: Path | str,
    extra_meta: Dict[str, Any] | None = None,
) -> Path:
    """
    Build fingerprint (geometry + selected Mode-6 vectors) and write YAML.
    """
    out_path = Path(out_path)
    geom_arr, symbols = parse_geometry_xyz(out_path)
    sel_vecs = select_atoms_mode6(out_path, atom_indices)

    meta: Dict[str, Any] = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_out": str(out_path),
        "mode": 6,
        "atom_indices": [int(i) for i in atom_indices],
    }
    if extra_meta:
        meta.update(extra_meta)

    doc: Dict[str, Any] = {
        "meta": meta,
        "geometry": geom_arr.tolist(),                 # full geometry
        "symbols": symbols,
        "selected_geometry": geom_arr[np.asarray(list(atom_indices), dtype=int), :].tolist(),
        "selected_mode6": sel_vecs.tolist(),           # (K,3)
    }

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return dst_path
