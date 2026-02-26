"""
TSGen 2.1 | tsgen_orca.py

Minimal ORCA I/O utilities required by TSGen:
    - write_orca_input
    - parse_frequencies_and_modes
    - read_final_xyz

No SLURM logic. No scratch logic. Pure I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Union
import json
import re

import numpy as np


# ======================================================================
#  WRITE ORCA INPUT
# ======================================================================

_StrOrStrList = Union[str, List[str]]


def _as_blocks(x: Optional[_StrOrStrList]) -> List[str]:
    """
    Normalize user-supplied blocks to a list of strings.

    Accepts:
        - None
        - str (treated as a single verbatim block)
        - list[str] (each entry treated as a single block; joined later with blank lines)
    """
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    # fallback: coerce
    s = str(x).strip()
    return [s] if s else []


def _as_geom_lines(x: Optional[_StrOrStrList]) -> List[str]:
    """
    Normalize geom_block lines.

    Accepts:
        - None
        - str  (splitlines)
        - list[str]
    Returns:
        list of lines WITHOUT trailing newlines.
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [ln.rstrip("\n") for ln in x.splitlines() if ln.strip()]
    if isinstance(x, list):
        return [str(ln).rstrip("\n") for ln in x if str(ln).strip()]
    return [str(x).rstrip("\n")]


def write_orca_input(
    path: Path,
    jobtype: str,
    method: str,
    charge: int,
    mult: int,
    geom_file: Path,
    *,
    use_ri: bool = True,
    add_aux_basis: bool = True,
    maxcore: int = 4000,
    nprocs: int = 8,
    geom_block: Optional[_StrOrStrList] = None,
    extra_blocks: Optional[_StrOrStrList] = None,
    provenance: Optional[dict] = None,
) -> None:
    """
    Minimal, stable ORCA input writer for TSGen.

    Notes on blocks
    --------------
    - geom_block:
        Intended to be the *contents* of a %geom block (lines only),
        e.g. ["Convergence loose", "MaxIter 200"].
        If a string/list includes a line starting with "%geom", it is treated as a
        complete raw block and is appended verbatim (no wrapping).

    - extra_blocks:
        Appended verbatim as one or more raw blocks. This is where L0 constraints
        should go, since they already include "%geom ... end".
    """

    path = Path(path)

    # --------------------------------------------------
    # Load XYZ (drop header if present)
    # --------------------------------------------------
    raw = Path(geom_file).read_text().strip().splitlines()
    if len(raw) >= 2 and raw[0].strip().isdigit():
        raw = raw[2:]  # skip XYZ header

    # --------------------------------------------------
    # Method / basis
    # --------------------------------------------------
    parts = method.split("/")
    functional = parts[0].strip()
    basis = parts[1].strip() if len(parts) > 1 else None

    # --------------------------------------------------
    # Bang (!) line
    # --------------------------------------------------
    bang = f"! {jobtype} {functional}"
    if basis:
        bang += f" {basis}"

    if use_ri:
        bang += " RIJCOSX"
        if add_aux_basis and basis:
            b = basis.lower()
            if b.startswith("def2-"):
                bang += " def2/J def2/JK"
            elif b.startswith("ma-def2-"):
                bang += " ma-def2/J ma-def2/JK"
            elif "3c" in functional.lower():
                # composite methods bundle appropriate auxiliaries
                pass
            else:
                bang += " def2/J"

    bang += " TightSCF"

    # --------------------------------------------------
    # Core blocks
    # --------------------------------------------------
    blocks: List[str] = [
        f"%maxcore {int(maxcore)}",
        f"%pal\n  nprocs {int(nprocs)}\nend",
    ]

    # --------------------------------------------------
    # %geom block (default or user-specified)
    # --------------------------------------------------
    geom_lines = _as_geom_lines(geom_block)

    if geom_lines:
        # If the user supplied a full %geom block, do not wrap
        if any(ln.strip().lower().startswith("%geom") for ln in geom_lines):
            blocks.append("\n".join(geom_lines).strip())
        else:
            inner = "\n".join(f"  {ln}".rstrip() for ln in geom_lines)
            blocks.append(f"%geom\n{inner}\nend")
    else:
        pass

    # --------------------------------------------------
    # Extra raw blocks (verbatim)
    # --------------------------------------------------
    for blk in _as_blocks(extra_blocks):
        blocks.append(blk)

    # --------------------------------------------------
    # Geometry section
    # --------------------------------------------------
    geom_section = "\n".join([f"* xyz {charge} {mult}", *raw, "*"])

    # --------------------------------------------------
    # Provenance header
    # --------------------------------------------------
    header: List[str] = []
    if provenance:
        header.append(f"# TSGen provenance: {json.dumps(provenance)}")

    # --------------------------------------------------
    # Final content
    # --------------------------------------------------
    content = "\n\n".join([*header, bang, "\n\n".join(blocks), geom_section]) + "\n"
    path.write_text(content)


# ======================================================================
#  PARSE FREQUENCIES + NORMAL MODES
# ======================================================================

def parse_frequencies_and_modes(path: Path) -> Tuple[List[float], np.ndarray]:
    """
    Robust ORCA frequency + normal mode parser.

    Returns:
        freqs : list[float]                  # indexed by ORCA mode index (0-based)
        modes : (n_modes, n_atoms, 3)        # modes[mode_index] corresponds to ORCA mode_index
    Notes:
        - Safe for constrained / projected calculations
        - Ignores non-matching lines in the frequency section (e.g., "Scaling factor ...")
    """
    lines = Path(path).read_text(errors="ignore").splitlines()
    n = len(lines)

    # --------------------------------------------------
    # Locate “VIBRATIONAL FREQUENCIES”
    # --------------------------------------------------
    freq_start = None
    for i, line in enumerate(lines):
        if "VIBRATIONAL FREQUENCIES" in line.upper():
            freq_start = i
            break
    if freq_start is None:
        return [], np.zeros((0, 0, 3))

    # ORCA prints e.g.:  6:    -414.91 cm**-1  ***imaginary mode***
    freq_re = re.compile(r"^\s*(\d+)\s*:\s*([+-]?\d+(?:\.\d*)?)")
    freqs_dict: dict[int, float] = {}

    # Scan until NORMAL MODES / IR SPECTRUM / THERMOCHEMISTRY
    for line in lines[freq_start + 1 :]:
        s = line.strip().upper()
        if "NORMAL MODES" in s or "IR SPECTRUM" in s or "THERMOCHEMISTRY" in s:
            break
        if not s or s.startswith("-"):
            continue
        m = freq_re.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        val = float(m.group(2))
        freqs_dict[idx] = val

    if not freqs_dict:
        return [], np.zeros((0, 0, 3))

    max_idx = max(freqs_dict.keys())
    n_modes = max_idx + 1

    # Build freqs list aligned to ORCA indices
    freqs: List[float] = [float("nan")] * n_modes
    for k, v in freqs_dict.items():
        freqs[k] = v

    # --------------------------------------------------
    # Locate “NORMAL MODES”
    # --------------------------------------------------
    modes_start = None
    for i, line in enumerate(lines):
        if "NORMAL MODES" in line.upper():
            modes_start = i
            break
    if modes_start is None:
        return freqs, np.zeros((n_modes, 0, 3))

    # Header lines look like: "                  0          1          2 ..."
    header_re = re.compile(r"^\s*(\d+(?:\s+\d+)+)\s*$")

    idx = modes_start + 1
    while idx < n and not header_re.match(lines[idx]):
        idx += 1
    if idx >= n:
        return freqs, np.zeros((n_modes, 0, 3))

    collected: dict[int, list[np.ndarray]] = {}
    block_modes: list[int] = []
    block_rows: list[list[float]] = []

    def flush_block() -> None:
        if not block_modes or not block_rows:
            return
        arr = np.array(block_rows, float)  # rows × cols
        for col, midx in enumerate(block_modes):
            vec = arr[:, col]
            collected.setdefault(midx, []).append(vec)

    # --------------------------------------------------
    # Main parse loop
    # --------------------------------------------------
    while idx < n:
        line = lines[idx]

        mh = header_re.match(line)
        if mh:
            flush_block()
            # ORCA headers are already 0-based mode indices
            block_modes = [int(x) for x in line.split()]
            block_rows = []
            idx += 1
            continue

        parts = line.split()
        if parts and parts[0].lstrip("+-").isdigit():
            floats: List[float] = []
            for t in parts[1:]:
                try:
                    floats.append(float(t))
                except ValueError:
                    pass

            # pad/truncate to block width
            if len(floats) < len(block_modes):
                floats += [0.0] * (len(block_modes) - len(floats))
            elif len(floats) > len(block_modes):
                floats = floats[: len(block_modes)]

            block_rows.append(floats)
            idx += 1
            continue

        idx += 1

    flush_block()

    # --------------------------------------------------
    # Infer n_atoms from any collected vector length
    # --------------------------------------------------
    vec_lengths: List[int] = []
    for blocks in collected.values():
        if blocks:
            vec_lengths.append(len(blocks[-1]))
    if not vec_lengths:
        return freqs, np.zeros((n_modes, 0, 3))

    longest = max(vec_lengths)
    if longest % 3 != 0:
        return freqs, np.zeros((n_modes, 0, 3))

    n_atoms = longest // 3
    modes = np.zeros((n_modes, n_atoms, 3), float)

    for midx, blocks in collected.items():
        if not blocks:
            continue
        vec = blocks[-1]
        if len(vec) != 3 * n_atoms:
            continue
        if 0 <= midx < n_modes:
            modes[midx] = np.array(vec, float).reshape(n_atoms, 3)

    return freqs, modes


# ======================================================================
#  READ FINAL GEOMETRY FROM OUT FILE
# ======================================================================

def read_final_xyz(out_path: Path):
    """
    Extract final geometry from ORCA .out file.

    Returns:
        (atoms, coords) or None
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return None

    lines = out_path.read_text(errors="ignore").splitlines()
    n = len(lines)

    # Locate last "CARTESIAN COORDINATES (ANGSTROEM)" block
    starts = [
        i for i, l in enumerate(lines)
        if "CARTESIAN COORDINATES (ANGSTROEM)" in l.upper()
    ]
    if not starts:
        return None

    start = starts[-1] + 2
    atoms: List[str] = []
    coords: List[List[float]] = []

    for i in range(start, n):
        s = lines[i].strip()
        if not s or s.startswith("-"):
            break

        parts = s.split()
        if len(parts) < 4:
            break

        atom = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            break

        atoms.append(atom)
        coords.append([x, y, z])

    if not atoms:
        return None

    return atoms, np.array(coords, float)
