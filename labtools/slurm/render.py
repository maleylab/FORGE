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

def render_template(template_name_or_path: str | Path, out_path: str | Path, params: dict) -> None:
    """
    Render a single Jinja2 template to a file.
    Accepts either a filename resolved via configured search paths, or an absolute path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = Path(template_name_or_path)
    if t.is_absolute():
        env = Environment(loader=FileSystemLoader(str(t.parent)), undefined=StrictUndefined)
        tpl = env.get_template(t.name)
    else:
        # allow legacy aliases like "orca/optfreq.inp.j2" → "orca_optfreq.inp.j2"
        last_err = None
        tpl = None
        for cand in _resolve_template_candidates(str(t)):
            try:
                tpl = _ENV.get_template(cand)
                break
            except TemplateNotFound as e:
                last_err = e
                continue
        if tpl is None:
            raise FileNotFoundError(
                f"Template not found: {template_name_or_path!r}. "
                f"Searched {ORCA_DIR}, {SBATCH_DIR}, {TEMPLATES_ROOT} and legacy aliases."
            ) from last_err

    out_path.write_text(tpl.render(**(params or {})), encoding="utf-8")

def render_plan_jobs(
    plan_path: str | Path,
    out_dir: str | Path,
    only: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> List[Path]:
    """
    Render ORCA inputs from a plan.yaml.
    Writes OUT/<job_id>/<job_id>.inp and returns the list of written paths.
    Injects coordinates from 'structure' (.xyz supported), provides multiplicity aliases,
    and exposes atoms (list) plus natoms (int).
    """
    plan_path = Path(plan_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with plan_path.open("r", encoding="utf-8") as f:
        plan = yaml.safe_load(f) or {}

    jobs: List[Dict[str, Any]] = list(plan.get("jobs", []))
    if not jobs:
        raise ValueError(f"No jobs found in {plan_path}")

    wanted = set(only) if only else None
    written: List[Path] = []

    for job in jobs:
        jid = job.get("id")
        if not jid:
            if verbose:
                print("[render] skipping job without id", file=sys.stderr)
            continue
        if wanted and jid not in wanted:
            continue

        tpl_name = job.get("template")
        if not tpl_name:
            raise ValueError(f"Job {jid} missing 'template'")

        # Resolve template
        last_err = None
        tpl = None
        for cand in _resolve_template_candidates(tpl_name):
            try:
                if cand.startswith("/"):
                    env = Environment(loader=FileSystemLoader(str(Path(cand).parent)), undefined=StrictUndefined)
                    tpl = env.get_template(Path(cand).name)
                else:
                    tpl = _ENV.get_template(cand)
                break
            except TemplateNotFound as e:
                last_err = e
                continue
        if tpl is None:
            raise FileNotFoundError(
                f"Template not found for job {jid!r}: {tpl_name!r} "
                f"(searched {ORCA_DIR}, {SBATCH_DIR}, {TEMPLATES_ROOT} and variants)"
            ) from last_err

        # ── Build context ────────────────────────────────────────────────────
        ctx: Dict[str, Any] = dict(job)  # shallow copy

        # Multiplicity: standardize on 'multiplicity' for templates; keep aliases
        mult_val = ctx.get("multiplicity", ctx.get("mult", 1))
        ctx["multiplicity"] = _coerce_int(mult_val, 1)
        ctx["mult"] = ctx["multiplicity"]
        ctx["spinmult"] = ctx["multiplicity"]

        # Charge (default to 0)
        ctx["charge"] = _coerce_int(ctx.get("charge", 0), 0)

        # Load coordinates & natoms
        structure_path = ctx.get("structure")
        ctx["structure_path"] = structure_path
        natoms_hdr, coords_text = _load_xyz_meta(structure_path)

        # Coordinate text aliases
        ctx["geometry_block"] = coords_text
        ctx["xyz_block"] = coords_text
        ctx["coords"] = coords_text
        # Legacy: some templates used {{ structure }} to mean the coordinate block
        ctx["structure"] = coords_text

        # Atoms list (for `{% for atom in atoms %}`)
        atoms_list = _parse_xyz_coords(coords_text)
        ctx["atoms"] = atoms_list
        ctx["atom_lines"] = [a["line"] for a in atoms_list]

        # Count aliases (ints)
        natoms_effective = natoms_hdr if natoms_hdr is not None else len(atoms_list)
        ctx["natoms"] = natoms_effective
        ctx["atom_count"] = natoms_effective

        # Preflight
        if not coords_text.strip():
            raise RuntimeError(
                f"Job {jid}: geometry is empty or unreadable (structure={structure_path!r}). "
                "Ensure the CSV points to a readable .xyz file or provide inline coordinates."
            )

        # ── Output path: OUT/<job_id>/<job_id>.inp ───────────────────────────
        job_dir = out_dir / jid
        job_dir.mkdir(parents=True, exist_ok=True)
        out_path = job_dir / f"{jid}.inp"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

        # ── Render with friendly error on missing variables ──────────────────
        try:
            text = tpl.render(**ctx)
        except UndefinedError as e:
            import re, json
            msg = str(e)
            m = re.search(r"'([^']+)' is undefined", msg)
            missing = m.group(1) if m else None

            # Helpful context dump (top-level keys and parameters.* keys)
            top_keys = sorted(ctx.keys())
            param_keys = sorted((ctx.get("parameters") or {}).keys())

            hints = []
            if missing in {"method","basis","grid","dispersion","ri","aux_basis","keywords","blocks"}:
                hints.append("This template now reads these under 'parameters.*'. "
                             f"Provide them via CSV → jobs[].parameters.{missing} or update the template to use p.{missing}.")

            details = (f"Job {jid}: template variable {missing!r} is missing in template {tpl_name!r}."
                       if missing else
                       f"Job {jid}: a required template variable is missing in template {tpl_name!r}.")
            details += "\nAvailable top-level keys: " + ", ".join(top_keys)
            details += "\nAvailable parameters.* keys: " + (", ".join(param_keys) or "<none>")
            if hints:
                details += "\nHint: " + " ".join(hints)

            raise RuntimeError(details) from None


        out_path.write_text(text, encoding="utf-8")
        written.append(out_path)

    return written

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
