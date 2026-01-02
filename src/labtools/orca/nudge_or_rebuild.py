#!/usr/bin/env python3
"""
nudge_or_rebuild.py

Two behaviors, now exposed via functions:

1) Imaginary-frequency jobs (nudge_ifreq_jobs):
   - Parse ORCA .out
   - Take stationary-point geometry (FINAL ENERGY...; fallback to last coordinates)
   - Extract lowest imaginary normal mode
   - Displace geometry a small distance along +direction of that mode
   - Overwrite .inp geometry (backing up original to .bak.inp)
   - Delete all other files in the directory
   - Create READY

2) Failed jobs (rebuild_failed_jobs):
   - Parse ORCA .out
   - Take stationary-point geometry if present; otherwise last coordinates
   - Overwrite .inp geometry (backing up original to .bak.inp)
   - Delete all other files in the directory
   - Create READY

Priority in the original standalone script if a job appears in both lists: IFREQ > FAILED.
Here we expose separate functions so the caller (CLI) decides which to run.
"""

import math
from pathlib import Path
import re
from typing import List, Tuple, Optional, Set


# ---------- Utilities ----------

def read_job_dirs(list_path: Path) -> List[Path]:
    """Read a text file with one job directory path per line (comments/# allowed)."""
    text = list_path.read_text()
    dirs: List[Path] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        dirs.append(Path(s))
    return dirs


def choose_primary_out(jobdir: Path) -> Optional[Path]:
    """
    Choose a primary ORCA .out file in a job directory:
      - ignore slurm logs and .N.out chunks
      - prefer one whose stem matches directory name
      - otherwise, the largest file
    """
    outs = [
        p
        for p in jobdir.glob("*.out")
        if not re.match(r"slurm[-_].*|.*\.\d+\.out|.*\.\d+\.err", p.name, re.I)
    ]
    if not outs:
        return None
    for p in outs:
        if p.stem == jobdir.name:
            return p
    outs.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return outs[0]


def choose_primary_inp(jobdir: Path) -> Optional[Path]:
    """
    Choose a primary ORCA .inp file in a job directory:
      - prefer name matching directory
      - otherwise, largest/most recent
    """
    inps = list(jobdir.glob("*.inp"))
    if not inps:
        return None
    for p in inps:
        if p.stem == jobdir.name:
            return p
    inps.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return inps[0]


# ---------- Geometry parsing ----------

def _parse_cartesian_block_from(text: str, start_idx: int) -> List[Tuple[str, float, float, float]]:
    """Given a starting index into the file, find the first
    'CARTESIAN COORDINATES (ANGSTROEM)' block and parse it."""
    sub = text[start_idx:]
    m2 = re.search(r"CARTESIAN COORDINATES\s*\(ANGSTROEM\)", sub)
    if not m2:
        raise ValueError("CARTESIAN COORDINATES (ANGSTROEM) not found")
    geo_start = start_idx + m2.end()
    rest = text[geo_start:].splitlines()
    coords: List[Tuple[str, float, float, float]] = []
    for line in rest:
        s = line.strip()
        if not s:
            if coords:
                break
            else:
                continue
        parts = s.split()
        if len(parts) < 4:
            if coords:
                break
            else:
                continue
        label = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            if coords:
                break
            else:
                continue
        coords.append((label, x, y, z))
    if not coords:
        raise ValueError("Failed to parse any coordinates after CARTESIAN COORDINATES block")
    return coords


def parse_final_geometry(text: str) -> List[Tuple[str, float, float, float]]:
    """
    Prefer geometry from the 'FINAL ENERGY EVALUATION AT THE STATIONARY POINT'
    section, CARTESIAN COORDINATES (ANGSTROEM).
    """
    m = re.search(r"FINAL ENERGY EVALUATION AT THE STATIONARY POINT", text)
    if not m:
        raise ValueError("Stationary-point FINAL ENERGY section not found")
    return _parse_cartesian_block_from(text, m.start())


def parse_last_cartesian_geometry(text: str) -> List[Tuple[str, float, float, float]]:
    """
    Find the *last* 'CARTESIAN COORDINATES (ANGSTROEM)' block in the file.
    Useful for failed jobs where stationary-point section is missing.
    """
    matches = list(re.finditer(r"CARTESIAN COORDINATES\s*\(ANGSTROEM\)", text))
    if not matches:
        raise ValueError("No CARTESIAN COORDINATES (ANGSTROEM) blocks found")
    last = matches[-1]
    return _parse_cartesian_block_from(text, last.start())


def parse_geometry_with_fallback(text: str) -> List[Tuple[str, float, float, float]]:
    """
    Try to parse stationary-point geometry; if that fails, fall back to
    last coordinates block in the file.
    """
    try:
        return parse_final_geometry(text)
    except Exception:
        return parse_last_cartesian_geometry(text)


# ---------- Vibrational mode parsing ----------

def parse_imag_mode_index(text: str) -> Optional[int]:
    """
    From the VIBRATIONAL FREQUENCIES section, find the most negative
    frequency and return its mode index. Returns None if no imag.
    """
    m = re.search(r"VIBRATIONAL FREQUENCIES", text)
    if not m:
        return None
    sub = text[m.end():]
    lines = sub.splitlines()
    freqs = []
    for line in lines:
        if "NORMAL MODES" in line:
            break
        mline = re.match(
            r"\s*(\d+):\s*([-+]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*cm", line
        )
        if mline:
            idx = int(mline.group(1))
            val = float(mline.group(2))
            freqs.append((idx, val))
    neg = [f for f in freqs if f[1] < 0.0]
    if not neg:
        return None
    neg.sort(key=lambda x: x[1])  # most negative first
    return neg[0][0]


def parse_normal_mode_displacements(text: str, mode_index: int, natoms: int) -> List[float]:
    """
    Parse the NORMAL MODES matrix and extract the column for mode_index
    as a flattened 3N displacement vector (mass-weighted as printed).
    """
    m = re.search(r"NORMAL MODES", text)
    if not m:
        raise ValueError("NORMAL MODES section not found")
    lines = text[m.end():].splitlines()
    disp = {}
    current_modes: Optional[List[int]] = None

    for line in lines:
        if not line.strip():
            continue
        # Header line: digits but no '.' (mode indices 0 1 2 3 4 5 ...)
        if re.search(r"\d", line) and "." not in line:
            modes = [int(x) for x in re.findall(r"\d+", line)]
            current_modes = modes
            continue
        # Row line: has '.' and starts with row index
        if "." in line:
            mrow = re.match(r"\s*(\d+)\s+(.+)", line)
            if not mrow or current_modes is None:
                continue
            row_idx = int(mrow.group(1))
            float_str = mrow.group(2)
            try:
                vals = [float(v) for v in float_str.split() if v]
            except ValueError:
                continue
            if mode_index in current_modes:
                col = current_modes.index(mode_index)
                if col < len(vals):
                    disp[row_idx] = vals[col]
            if len(disp) >= 3 * natoms:
                break

    if not disp:
        raise ValueError(f"No displacements found for mode {mode_index}")

    size = 3 * natoms
    vec = [0.0] * size
    for i, v in disp.items():
        if i < size:
            vec[i] = v
    return vec


# ---------- Geometry modification / input editing ----------

def displace_geometry(coords, vec, step: float):
    """
    Normalize vec and displace coordinates by 'step' Å along +direction.
    """
    nat = len(coords)
    if len(vec) < 3 * nat:
        raise ValueError("Displacement vector shorter than 3N")
    norm = math.sqrt(sum(v * v for v in vec[:3 * nat]))
    if norm == 0.0:
        raise ValueError("Zero-norm displacement vector")
    scale = step / norm
    new = []
    for i, (label, x, y, z) in enumerate(coords):
        dx = vec[3 * i] * scale
        dy = vec[3 * i + 1] * scale
        dz = vec[3 * i + 2] * scale
        new.append((label, x + dx, y + dy, z + dz))
    return new


def replace_geometry_in_inp(inp_text: str, new_coords) -> str:
    """
    Replace the * xyz ... * block in an ORCA input with new_coords.
    """
    lines = inp_text.splitlines()
    start = None
    end = None

    # Find "* xyz" header
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("* xyz"):
            start = idx
            break
    if start is None:
        raise ValueError("No '* xyz' line found in input")

    # Find terminating "*" of geometry block
    for idx in range(start + 1, len(lines)):
        if lines[idx].strip().startswith("*"):
            end = idx
            break
    if end is None:
        raise ValueError("No terminating '*' found for geometry block")

    geom_lines = []
    for label, x, y, z in new_coords:
        geom_lines.append(f"  {label:<2}  {x: .8f}  {y: .8f}  {z: .8f}")

    new_lines = lines[: start + 1] + geom_lines + lines[end:]
    return "\n".join(new_lines) + ("\n" if inp_text.endswith("\n") else "")


def backup_and_write_inp(inp_path: Path, new_text: str) -> None:
    """
    Back up original input as <stem>.inp.bak (if not exists),
    then overwrite the .inp with new_text.
    """
    inp_text = inp_path.read_text()
    backup = inp_path.with_suffix(".inp.bak")
    if not backup.exists():
        backup.write_text(inp_text)
    inp_path.write_text(new_text)


def cleanup_jobdir(jobdir: Path, inp_path: Path) -> None:
    """
    In JOBDIR, keep only:
      - the updated .inp
      - its backup .bak.inp
    Remove all other files.
    Then create an empty READY file.
    """
    backup_name = inp_path.with_name(inp_path.stem + ".bak.inp").name
    for p in jobdir.iterdir():
        if not p.is_file():
            continue
        if p == inp_path:
            continue
        if p.name == backup_name:
            continue
        p.unlink()
    (jobdir / "READY").write_text("")


# ---------- Per-job operations ----------

def process_ifreq_job(jobdir: Path, step: float) -> None:
    out_path = choose_primary_out(jobdir)
    if not out_path:
        print(f"[IFREQ][SKIP] {jobdir}: no .out file found")
        return
    inp_path = choose_primary_inp(jobdir)
    if not inp_path:
        print(f"[IFREQ][SKIP] {jobdir}: no .inp file found")
        return

    text = out_path.read_text(errors="ignore")

    try:
        coords = parse_geometry_with_fallback(text)
    except Exception as e:
        print(f"[IFREQ][SKIP] {jobdir}: failed to parse geometry: {e}")
        return

    natoms = len(coords)
    mode_idx = parse_imag_mode_index(text)
    if mode_idx is None:
        print(f"[IFREQ][SKIP] {jobdir}: no imaginary frequency found")
        return

    try:
        vec = parse_normal_mode_displacements(text, mode_idx, natoms)
    except Exception as e:
        print(f"[IFREQ][SKIP] {jobdir}: failed to parse normal mode {mode_idx}: {e}")
        return

    try:
        new_coords = displace_geometry(coords, vec, step=step)
    except Exception as e:
        print(f"[IFREQ][SKIP] {jobdir}: failed to displace geometry: {e}")
        return

    inp_text = inp_path.read_text()
    try:
        new_inp = replace_geometry_in_inp(inp_text, new_coords)
    except Exception as e:
        print(f"[IFREQ][SKIP] {jobdir}: failed to update input file: {e}")
        return

    backup_and_write_inp(inp_path, new_inp)
    cleanup_jobdir(jobdir, inp_path)

    print(f"[IFREQ][OK] {jobdir}: nudged along mode {mode_idx} by step {step} Å and marked READY")


def process_failed_job(jobdir: Path) -> None:
    out_path = choose_primary_out(jobdir)
    if not out_path:
        print(f"[FAIL][SKIP] {jobdir}: no .out file found")
        return
    inp_path = choose_primary_inp(jobdir)
    if not inp_path:
        print(f"[FAIL][SKIP] {jobdir}: no .inp file found")
        return

    text = out_path.read_text(errors="ignore")

    try:
        coords = parse_geometry_with_fallback(text)
    except Exception as e:
        print(f"[FAIL][SKIP] {jobdir}: failed to parse geometry: {e}")
        return

    # Just overwrite geometry, no displacement
    inp_text = inp_path.read_text()
    try:
        new_inp = replace_geometry_in_inp(inp_text, coords)
    except Exception as e:
        print(f"[FAIL][SKIP] {jobdir}: failed to update input file: {e}")
        return

    backup_and_write_inp(inp_path, new_inp)
    cleanup_jobdir(jobdir, inp_path)

    print(f"[FAIL][OK] {jobdir}: rebuilt from last geometry and marked READY")


# ---------- Public batch functions for CLI ----------

def nudge_ifreq_jobs(list_path: Path, step: float = 0.1) -> None:
    """
    Nudge all jobs listed in list_path (one jobdir per line) along the
    lowest imaginary mode by 'step' Å and mark them READY.
    """
    ifreq_dirs: Set[Path] = set(read_job_dirs(list_path))
    for jobdir in sorted(ifreq_dirs):
        if not jobdir.is_dir():
            print(f"[IFREQ][SKIP] {jobdir}: not a directory")
            continue
        process_ifreq_job(jobdir, step=step)


def rebuild_failed_jobs(list_path: Path) -> None:
    """
    Rebuild all failed jobs listed in list_path (one jobdir per line)
    from their last geometry and mark them READY.
    """
    failed_dirs: Set[Path] = set(read_job_dirs(list_path))
    for jobdir in sorted(failed_dirs):
        if not jobdir.is_dir():
            print(f"[FAIL][SKIP] {jobdir}: not a directory")
            continue
        process_failed_job(jobdir)
