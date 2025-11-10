from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .constraints import CartFreeze, to_orca_cartesian_constraints
from .solvents import Solvent


def _jinja_env(templates_root: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_root)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_orca_input(
    elems: List[str],
    coords,
    cf: CartFreeze,
    solvent: Solvent,
    *,
    pal_threads: int,
    maxcore_mb: int,
    templates_root: Optional[Path] = None,
    template_name: str = "tsgen_optfreq_cartesian.inp.j2",
) -> str:
    """
    Render the tsgen-specific ORCA input using Jinja.

    Args:
        elems, coords: molecule.
        cf: Cartesian-only freezes for this phase.
        solvent: none|cpcm (with name/epsilon/refrac).
        pal_threads: PALN (e.g., 8).
        maxcore_mb: %maxcore value (per thread, MB).
        templates_root: path to your repo's templates/orca directory.
                        If None, defaults to <repo_root>/templates/orca.
        template_name: j2 file to render (default is tsgen-specific).

    Returns:
        ORCA input text (str).
    """
    method = "B97-3c"
    flags = ["Opt", "Freq", f"PAL{pal_threads}"]
    cpcm = None
    if solvent and solvent.model == "cpcm":
        flags.append("CPCM")
        cpcm = {"epsilon": solvent.epsilon}
        if solvent.refrac is not None:
            cpcm["refrac"] = solvent.refrac

    cart_lines = to_orca_cartesian_constraints(cf)
    xyz_lines = [f"{s:2s} {x: .8f} {y: .8f} {z: .8f}" for s, (x, y, z) in zip(elems, coords)]

    # Locate templates/orca by default if not provided
    if templates_root is None:
        # repo layout: <repo_root>/templates/orca
        # this file is at src/labtools/tsgen/render_orca.py → go up 3, then templates/orca
        templates_root = Path(__file__).resolve().parents[3] / "templates" / "orca"

    env = _jinja_env(templates_root)
    tmpl = env.get_template(template_name)

    ctx = {
        "method": method,
        "flags": flags,
        "maxcore_mb": maxcore_mb,
        "cartesian_constraints": cart_lines,
        "charge": 0,
        "mult": 1,
        "xyz_lines": xyz_lines,
        "cpcm": cpcm,
    }
    return tmpl.render(**ctx)

