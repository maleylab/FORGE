# src/labtools/slurm/render.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any, Tuple
import sys
import yaml
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateNotFound,
    UndefinedError,
)

# ── Paths (define these FIRST) ────────────────────────────────────────────────
# For THIS file at: /.../lab-tools/src/labtools/slurm/render.py
# parents[0]=.../slurm, [1]=.../labtools, [2]=.../src, [3]=.../lab-tools
REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES_ROOT = REPO_ROOT / "templates"
ORCA_DIR = TEMPLATES_ROOT / "orca"
SBATCH_DIR = TEMPLATES_ROOT / "sbatch"

# ── Jinja environment (after constants) ───────────────────────────────────────
_ENV = Environment(
    loader=FileSystemLoader([str(ORCA_DIR), str(SBATCH_DIR), str(TEMPLATES_ROOT), "/"]),
    undefined=StrictUndefined,   # fail loudly on missing vars
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

__all__ = ["render_template", "render_plan_jobs", "render_plan_entrypoint"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return int(default)

def _load_xyz_meta(structure_path: str | Path) -> Tuple[Optional[int], str]:
    """
    Return (natoms, coords_text). If not a standard .xyz, natoms=None and coords_text best-effort.
    For .xyz: strip the first two header lines (count + comment) and return the body as coords_text.
    """
    if not structure_path:
        return None, ""
    p = Path(structure_path)
    if not p.is_file():
        # Maybe inline coordinates were provided instead of a path
        txt = str(structure_path)
        return None, (txt if txt.strip() else "")
    txt = p.read_text(encoding="utf-8").strip("\n")
    lines = [ln.rstrip() for ln in txt.splitlines()]
    if p.suffix.lower() == ".xyz" and len(lines) >= 3:
        try:
            n = int(lines[0].strip())
            coords_text = "\n".join(lines[2:]).strip() + "\n"
            return n, coords_text
        except ValueError:
            pass
    # Not a standard .xyz; return as-is (we can’t trust natoms)
    return None, ("\n".join(lines).strip() + "\n" if lines else "")

def _parse_xyz_coords(coords_text: str):
    """
    Parse an XYZ coordinate block (no header) into a list of atoms:
    [{element, x, y, z, line}, ...]
    """
    atoms = []
    for ln in (coords_text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) >= 4:
            el = parts[0]
            try:
                x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
            except Exception:
                x = y = z = None
            atoms.append({"element": el, "x": x, "y": y, "z": z, "line": ln})
        else:
            atoms.append({"element": parts[0], "x": None, "y": None, "z": None, "line": ln})
    return atoms

def _resolve_template_candidates(name: str) -> List[str]:
    """
    Return candidate names to try in order, supporting legacy forms:
      - "orca/foo.inp.j2"      -> try "orca/foo.inp.j2", "foo.inp.j2", "orca_foo.inp.j2"
      - "sbatch/bar.sbatch.j2" -> try "sbatch/bar.sbatch.j2", "bar.sbatch.j2", "sbatch_bar.sbatch.j2"
      - absolute paths pass through unchanged
    """
    name = str(name).strip()
    if name.startswith("/"):
        return [name]
    base = Path(name).name
    cands = [name]
    if base not in cands:
        cands.append(base)
    if name.startswith("orca/"):
        alt = f"orca_{base}"
        if alt not in cands:
            cands.append(alt)
    if name.startswith("sbatch/"):
        alt = f"sbatch_{base}"
        if alt not in cands:
            cands.append(alt)
    return cands

def _normalize_out_root(
    out: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    out_root: Optional[Path] = None,
    output: Optional[Path] = None,
    outdir: Optional[Path] = None,
    default: Optional[Path] = None,
) -> Path:
    """
    Accept many alias names for output root; fall back to default if none provided.
    """
    choice = out_root or out or out_dir or output or outdir or default
    return Path(choice or "build/inputs").resolve()

# ── Public API ────────────────────────────────────────────────────────────────

def render_template(
    template_name_or_path: str | Path,
    out_path: str | Path,
    params: dict,
    *,
    return_text: bool = False,  # NEW
) -> str | None:
    """
    Render a single Jinja2 template to a file.
    Accepts either a filename resolved via configured search paths, or an absolute path.
    If return_text=True, returns the rendered string instead of writing to disk.
    """
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    import os

    if isinstance(template_name_or_path, Path):
        tpl_path = template_name_or_path
        tpl_name = tpl_path.name
        search_path = str(tpl_path.parent)
    else:
        tpl_name = template_name_or_path
        search_path = os.path.dirname(__file__)

    env = Environment(
        loader=FileSystemLoader(search_path),
        undefined=StrictUndefined,
        lstrip_blocks=True,
        trim_blocks=True,
    )
    template = env.get_template(tpl_name)
    rendered = template.render(params)

    if return_text:
        return rendered
    else:
        out_path = Path(out_path)
        out_path.write_text(rendered)
        return None

# ── Compatibility entrypoint (tolerant to legacy kwargs) ──────────────────────
def render_plan_entrypoint(
    plan: str | Path,
    # output aliases (any is fine; default used if none provided)
    out: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    out_root: Optional[str | Path] = None,
    output: Optional[str | Path] = None,
    outdir: Optional[str | Path] = None,
    # behavior
    only: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    # swallow unknown kwargs so older wrappers don't blow up
    **kwargs: Any,
) -> List[Path]:
    """
    Compatibility wrapper for callers that pass various output arg names or extra kwargs.
    - If no output is provided, defaults to 'build/inputs'
    - If dry_run=True, parse plan and return [] after basic checks
    """
    out_root_path = _normalize_out_root(
        out=Path(out) if out else None,
        out_dir=Path(out_dir) if out_dir else None,
        out_root=Path(out_root) if out_root else None,
        output=Path(output) if output else None,
        outdir=Path(outdir) if outdir else None,
        default=Path("build/inputs"),
    )

    # Normalize --only if provided as comma-separated string in some wrappers
    only_set = None
    if only:
        if isinstance(only, (str, Path)):
            only_set = {str(only)}
        else:
            # Support comma-separated forms inside the iterable
            expanded: List[str] = []
            for item in only:
                expanded.extend([p for p in str(item).split(",") if p])
            only_set = set(expanded)

    if dry_run:
        # Validate plan shape and coordinates availability, but don't write files
        with Path(plan).open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        jobs = list(doc.get("jobs", []))
        if not jobs:
            raise RuntimeError(f"No jobs found in {plan}")
        if only_set:
            jobs = [j for j in jobs if j.get("id") in only_set]
        # light check for structure presence
        for j in jobs:
            sp = j.get("structure")
            if not sp:
                raise RuntimeError(f"Job {j.get('id')}: missing 'structure' field")
        # success: dry-run produces no files
        return []

    # Real render
    return render_plan_jobs(plan, out_root_path, only=only_set, overwrite=overwrite, verbose=False)
