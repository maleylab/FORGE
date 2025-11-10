"""
ORCA I/O: input rendering, frequency parsing, and submission wrapper.
"""

from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Optional


def write_orca_input(
    path: Path,
    jobtype: str,
    method: str,
    charge: int,
    mult: int,
    geom_file: Path,
    constraints: Optional[str] = None,
    use_ri: bool = True,
    add_aux_basis: bool = True,
    maxcore: int = 4000,
    nprocs: int = 8,
    extra_blocks: Optional[str] = None,
    provenance: Optional[dict] = None,
) -> None:
    geom_text = Path(geom_file).read_text().strip().splitlines()
    if len(geom_text) > 2 and geom_text[0].strip().isdigit():
        geom_text = geom_text[2:]

    # Bang line assembly
    method_tokens = method.split("/")
    functional = method_tokens[0]
    basis = method_tokens[1] if len(method_tokens) > 1 else None

    bang = f"! {jobtype} {functional}"
    if basis:
        bang += f" {basis}"

    if use_ri:
        bang += " RIJCOSX"
        if add_aux_basis:
            if basis and basis.lower().startswith("def2-"):
                bang += " def2/J def2/JK"
            elif basis and basis.lower().startswith("ma-def2-"):
                bang += " ma-def2/J ma-def2/JK"
            elif "3c" in functional.lower():
                pass
            else:
                bang += " def2/J"

    bang += " TightSCF"

    header = []
    if provenance:
        header.append(f"# FORGE provenance: {json.dumps(provenance)}")

    blocks = [f"%maxcore {maxcore}", f"%pal nprocs {nprocs} end"]
    if constraints:
        blocks.append(constraints)
    if extra_blocks:
        blocks.append(extra_blocks.strip())

    geom_section = "\n".join(["* xyz %d %d" % (charge, mult), *geom_text, "*"])
    content = "\n\n".join([*header, bang, "\n".join(blocks), geom_section])
    Path(path).write_text(content)


def parse_frequencies(out_path: Path):
    import numpy as np
    text = Path(out_path).read_text(errors="ignore")
    freq_lines = re.findall(r"Frequencies --\s+([-\d\.E\s]+)", text)
    freqs = []
    for fl in freq_lines:
        freqs += [float(x) for x in fl.split()]

    mode_pattern = re.compile(r"\s*\d+\s+([-0-9.Ee\s]+)")
    mode_data = mode_pattern.findall(text)
    modes = []
    if mode_data and freqs:
        n_atoms = _guess_atom_count(freqs, len(mode_data))
        vecs = np.array([list(map(float, x.split())) for x in mode_data])
        for i in range(len(freqs)):
            modes.append(vecs[i*n_atoms:(i+1)*n_atoms])
    return freqs, np.array(modes)


def _guess_atom_count(freqs, total_lines):
    if not freqs:
        return 0
    n_modes = len(freqs)
    for n in range(1, 2000):
        if n_modes * n == total_lines:
            return n
    return int(total_lines / n_modes)


def run_orca(inp_path: Path, cwd: Optional[Path] = None, mode: str = "job", profile: str = "medium"):
    """Delegate execution to global dispatcher using templates."""
    from labtools.submit import dispatch
    workdir = Path(cwd or inp_path.parent)
    # ensure relative paths inside workdir
    dispatch(workdir / inp_path.name, mode=mode, profile=profile)
