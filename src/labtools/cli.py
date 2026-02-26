from __future__ import annotations

import datetime
import fnmatch
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as _pd
import typer

# Chemistry / descriptors
from labtools.chem import energetic_span as es_mod
from labtools.chem.descriptors import homo_lumo_gap  # noqa: F401

# IO helpers
from labtools.data.io import jsonl_append, jsonl_to_parquet
from labtools.orca.parse import parse_orca_file, collect_job_record  # noqa: F401
from labtools.orca.nudge_or_rebuild import nudge_ifreq_jobs, rebuild_failed_jobs
from labtools.orca.queues import make_queues

# Provenance
from labtools.prov import snapshot as snap_mod

# Templates / SLURM rendering
from labtools.slurm.render import render_template

# Optional legacy helper (older FORGE versions). Keep CLI importable even if missing.
try:
    from labtools.slurm.render import render_plan_jobs  # type: ignore
except Exception:  # pragma: no cover
    render_plan_jobs = None  # type: ignore

from labtools.slurm.render import render_plan_entrypoint as render_plan

# Submission dispatcher
from labtools.submit import dispatch

# CSV → Plan machinery
from labtools.csvpipe.loader import read_csv_rows, row_to_job
from labtools.csvpipe.mapping import build_mapping
from labtools.csvpipe.expand import gather_axes, expand_product, expand_zip
from labtools.csvpipe.emit import (
    emit_job_yaml_files,
    job_to_planentry,
    emit_planentries_jsonl,
)

from labtools.plans.factory import planentry_from_dict


# ----------------------------------------------------------
# Optional TS-FP imports (SOFT)
# ----------------------------------------------------------
_tsfp_import_error: Optional[str] = None
write_fingerprint_yaml = None
verify_against_fingerprint = None

try:
    from labtools.tsfp.build import write_fingerprint_yaml
    from labtools.tsfp.verify import verify_against_fingerprint
except Exception as e:  # pragma: no cover
    _tsfp_import_error = str(e)

# ==========================================================
# ROOT CLI
# ==========================================================


FORGE_BANNER = r"""
              _________
             |         |
_____________|         |_____________
\                                    /
 \__________________________________/
            ||                 ||
            ||       FORGE     ||
            ||_________________||
""".strip("\n")


def _should_print_banner(ctx: typer.Context) -> bool:
    # Suppress during shell completion and parsing-only phases
    if os.environ.get("_TYPER_COMPLETE"):
        return False
    if getattr(ctx, "resilient_parsing", False):
        return False
    env = (os.environ.get("FORGE_NO_BANNER") or "").strip().lower()
    if env in ("1", "true", "yes", "y"):
        return False
    return True


app = typer.Typer(help="FORGE / lab-tools CLI", invoke_without_command=True)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    no_banner: bool = typer.Option(False, "--no-banner", help="Suppress the FORGE banner."),
):
    """Root callback: print banner once per invocation; show help if no subcommand."""
    if not no_banner and _should_print_banner(ctx):
        typer.echo(FORGE_BANNER)

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


tsfp_app = typer.Typer(help="Transition-state fingerprint tools")
app.add_typer(tsfp_app, name="ts-fp")

tsgen2_app = typer.Typer(help="TSGen 2.0 transition-state workflow")
app.add_typer(tsgen2_app, name="tsgen2")

plan_app = typer.Typer(help="PlanEntry-based workflows")
app.add_typer(plan_app, name="plan")

job_app = typer.Typer(help="Single-job utilities")
app.add_typer(job_app, name="job")

watch_app = typer.Typer(help="Watch jobs (filesystem + scheduler sampler)")
app.add_typer(watch_app, name="watch")


# ==========================================================
# TS-FP commands (soft-dependency)
# ==========================================================


def _require_tsfp() -> None:
    if write_fingerprint_yaml is None or verify_against_fingerprint is None:
        raise typer.BadParameter(
            "TS-FP tools are unavailable (optional dependency import failed). "
            f"Import error: {_tsfp_import_error}"
        )


@tsfp_app.command("build")
def tsfp_build(
    out: Path = typer.Option(..., "--out"),
    atoms: List[int] = typer.Option(..., "--atom"),
    vector: List[float] = typer.Option(..., "--v"),
    id: str = typer.Option("fingerprint", "--id"),
    version: int = typer.Option(1, "--version"),
    n_imag: int = typer.Option(1, "--n-imag"),
    imag_min: float = typer.Option(-800.0, "--imag-min"),
    imag_max: float = typer.Option(-100.0, "--imag-max"),
    min_cosine: float = typer.Option(0.85, "--min-cosine"),
    min_localization: float = typer.Option(0.6, "--min-localization"),
):
    _require_tsfp()
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": id,
        "version": int(version),
        "atom_indices": list(map(int, atoms)),
        "vector": list(map(float, vector)),
        "expected": {
            "n_imag": int(n_imag),
            "imag_cm1_min": float(imag_min),
            "imag_cm1_max": float(imag_max),
        },
        "thresholds": {
            "min_cosine": float(min_cosine),
            "min_localization": float(min_localization),
        },
    }
    write_fingerprint_yaml(payload, out)
    typer.secho(f"Wrote fingerprint → {out}", fg=typer.colors.GREEN)


@tsfp_app.command("verify")
def tsfp_verify(
    fingerprint: Path = typer.Option(..., "--fingerprint"),
    job_dir: Path = typer.Option(..., "--job-dir"),
):
    _require_tsfp()
    fingerprint = fingerprint.expanduser().resolve()
    job_dir = job_dir.expanduser().resolve()
    result = verify_against_fingerprint(fingerprint, job_dir)
    typer.echo(json.dumps(result, indent=2))


# ==========================================================
# PlanEntry CLI
# ==========================================================


def _repo_templates_root() -> Path:
    return Path(__file__).resolve().parents[2] / "templates"

def _templates_root() -> Path:
    """
    Find the *package* templates directory robustly, both in editable installs
    and installed wheels/sdists.

    Expected layout:
      labtools/templates/orca/*.j2
      labtools/templates/sbatch/*.j2
    """
    # 1) Try importlib.resources (best for installed packages)
    try:
        from importlib import resources as importlib_resources  # py3.9+
        root = importlib_resources.files("labtools") / "templates"
        # convert Traversable -> Path if possible
        if hasattr(root, "__fspath__"):
            p = Path(os.fspath(root))
            if p.is_dir():
                return p
    except Exception:
        pass

    # 2) Fallback: walk up from this file and look for templates/
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "templates"
        if cand.is_dir():
            return cand

    raise FileNotFoundError("Could not locate labtools/templates directory")

@plan_app.command("validate")
def plan_validate_cmd(
    plan: Path = typer.Option(..., "--plan"),
):
    plan = plan.expanduser().resolve()
    if not plan.is_file():
        raise typer.BadParameter(f"Plan file not found: {plan}")

    with plan.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                planentry_from_dict(obj)
            except Exception as e:
                raise typer.BadParameter(f"Invalid PlanEntry on line {i}: {e}")

    typer.secho(f"Plan validated successfully → {plan}", fg=typer.colors.GREEN)

@plan_app.command("render")
def plan_render_cmd(
    plan: Path = typer.Option(..., "--plan"),
    outdir: Path = typer.Option("build/jobs", "--outdir"),
):
    from labtools.plans.render import render_planentries
    from jinja2 import Environment, FileSystemLoader, ChainableUndefined

    plan = plan.expanduser().resolve()
    outdir = outdir.expanduser().resolve()

    if not plan.is_file():
        raise typer.BadParameter(f"Plan file not found: {plan}")

    # -------------------------
    # Load + validate PlanEntries
    # -------------------------
    entries = []
    with plan.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entries.append(planentry_from_dict(json.loads(line)))
            except Exception as e:
                raise typer.BadParameter(f"Invalid PlanEntry on line {i}: {e}")

    if not entries:
        raise typer.BadParameter("No PlanEntries found in plan file")

    # -------------------------
    # Task → template mapping
    # -------------------------
    PLAN_TASK_TEMPLATES = {
        "sp": "orca/orca_sp.inp.j2",
        "opt": "orca/orca_opt.inp.j2",
        "freq": "orca/orca_freq.inp.j2",
        "optfreq": "orca/orca_optfreq.inp.j2",
        "irc": "orca/orca_irc.inp.j2",
        "nmr": "orca/orca_nmr.inp.j2",
        "sp-triplet": "orca/orca_sp_triplet.inp.j2",
        "gradient": "orca/orca_grad.inp.j2",
        "tsopt": "orca/orca_tsopt.inp.j2"
    }

    # -------------------------
    # Jinja environment (robust)
    # -------------------------
    templates_root = _templates_root()
    env = Environment(
        loader=FileSystemLoader(str(templates_root)),
        undefined=ChainableUndefined,
        autoescape=False,
    )

    # -------------------------
    # Single-job renderer
    # -------------------------
    def _render_single_job(job: Dict[str, Any], *, job_dir: Path):
        """Render a single PlanEntry render-context dict to job.inp.

        NOTE: `render_planentries()` passes `planentry_to_render_context(entry)` as `job`.
        That object is already normalized (scf/cpcm/freq dicts at top-level, defaults present).
        Do not re-parse `_row_raw` here.
        """

        # -------------------------
        # Resolve task → template
        # -------------------------
        task = job.get("job_type")
        if not task:
            raise typer.BadParameter("PlanEntry missing parameters.job_type")
        task = str(task).strip().lower()

        if task not in PLAN_TASK_TEMPLATES:
            raise typer.BadParameter(f"Unknown task '{task}'")

        # -------------------------
        # Structure + geometry
        # -------------------------
        structure = job.get("structure") or job.get("system")
        if not structure:
            raise typer.BadParameter("PlanEntry missing structure/system path")

        structure = Path(str(structure))
        if not structure.is_file():
            raise typer.BadParameter(f"Structure file not found: {structure}")

        # Prefer pre-populated geom_lines if present (may be empty if intent.system was a string).
        geom_lines = job.get("geom_lines") if isinstance(job.get("geom_lines"), list) else []
        if not geom_lines:
            lines = structure.read_text(encoding="utf-8").strip().splitlines()
            if lines and lines[0].strip().isdigit():
                lines = lines[2:]  # strip XYZ header
            geom_lines = lines

        # -------------------------
        # Required ORCA fields
        # -------------------------
        method = job.get("method")
        if not method:
            raise typer.BadParameter("PlanEntry missing parameters.method")

        charge = job.get("charge", 0)
        mult = job.get("multiplicity", job.get("mult", 1))

        try:
            charge = int(charge)
        except Exception:
            raise typer.BadParameter(f"Invalid charge: {charge!r}")

        try:
            mult = int(mult)
        except Exception:
            raise typer.BadParameter(f"Invalid multiplicity: {mult!r}")

        # -------------------------
                # -------------------------
        # Optional controls + flexible keyword injection
        # -------------------------
        def _as_dict(x):
            return x if isinstance(x, dict) else {}

        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                s = x.strip()
                return s.split() if s else []
            return []

        def _deep_set(d: Dict[str, Any], dotted: str, val: Any) -> None:
            parts = [p for p in str(dotted).split(".") if p]
            cur = d
            for p in parts[:-1]:
                nxt = cur.get(p)
                if not isinstance(nxt, dict):
                    nxt = {}
                    cur[p] = nxt
                cur = nxt
            if parts:
                cur[parts[-1]] = val

        # Merge sources (row raw → parameters → top-level) conservatively.
        # This restores support for dotted keys like scf.Convergence, freq.QRRHORefFreq, irc.MaxIter, etc.
        params = job.get("parameters") if isinstance(job.get("parameters"), dict) else {}

        row: Dict[str, Any] = {}
        if isinstance(job.get("_row_raw"), dict):
            row = job.get("_row_raw")  # type: ignore[assignment]
        elif isinstance(params.get("_row_raw"), dict):
            row = params.get("_row_raw")  # type: ignore[assignment]

        srcs = [row, params, job]

        def _first(key: str, default=None):
            for s in srcs:
                if isinstance(s, dict) and key in s and s[key] is not None:
                    return s[key]
            return default

        basis = _first("basis", None)
        grid = _first("grid", None)
        if isinstance(grid, str):
            grid = grid.strip() or None

        flags = _as_list(_first("flags", None))
        restart_flags = _as_list(_first("restart_flags", None))

        pal = _first("pal", None)
        pal = int(pal) if pal is not None and str(pal).strip() != "" else None

        maxcore = _first("maxcore_mb", None)
        maxcore = int(maxcore) if maxcore is not None and str(maxcore).strip() != "" else None

        scf_opts = _as_dict(_first("scf", {}))
        cpcm_opts = _as_dict(_first("cpcm", {}))
        freq_opts = _as_dict(_first("freq", {}))
        irc_opts = _as_dict(_first("irc", {}))
        geom_opts = _as_dict(_first("geom", {}))

        # Apply dotted overrides from any source dict
        for s in srcs:
            if not isinstance(s, dict):
                continue
            for k, v in s.items():
                if not isinstance(k, str) or "." not in k:
                    continue
                if k.startswith("scf."):
                    _deep_set(scf_opts, k[4:], v)
                elif k.startswith("cpcm."):
                    _deep_set(cpcm_opts, k[5:], v)
                elif k.startswith("freq."):
                    _deep_set(freq_opts, k[5:], v)
                elif k.startswith("irc."):
                    _deep_set(irc_opts, k[4:], v)
                elif k.startswith("geom."):
                    _deep_set(geom_opts, k[5:], v)

        # -------------------------
        # Template context (ALWAYS COMPLETE)
        # -------------------------
        ctx = {
            "method": method,
            "basis": basis,
            "grid": grid,
            "flags": flags,
            "restart_flags": restart_flags,
            "charge": charge,
            "mult": mult,
            "geom_lines": geom_lines,
            "geom": {
                "lines": geom_lines,
                "constraints": geom_opts.get("constraints", []),
                **{k: v for k, v in geom_opts.items() if k != "constraints"},
            },
            "scf": scf_opts,
            "cpcm": cpcm_opts,
            "freq": freq_opts,
            "irc": irc_opts,
        }

        if pal is not None:
            ctx["pal"] = pal
        if maxcore is not None:
            ctx["maxcore_mb"] = maxcore

# Render
        # -------------------------
        template = env.get_template(PLAN_TASK_TEMPLATES[task])
        text = template.render(**ctx)

        lines = text.splitlines()
        cleaned = []
        prev_blank = False
        for line in lines:
            blank = not line.strip()
            if blank and prev_blank:
                continue
            cleaned.append(line.rstrip())
            prev_blank = blank

        text = "\n".join(cleaned).strip() + "\n"

        (job_dir / "job.inp").write_text(text, encoding="utf-8")

    # -------------------------
    # Render all entries
    # -------------------------
    render_planentries(entries, render_func=_render_single_job, outdir=outdir)
    typer.secho(f"Rendered {len(entries)} jobs → {outdir}", fg=typer.colors.GREEN)



# ==========================================================
# TSGen 2.0
# ==========================================================

from labtools.tsgen.plan import TSGenPlan
from labtools.tsgen.controller import TSGenController


@tsgen2_app.command("init")
def tsgen2_init(
    reactant: Path = typer.Option(..., "--reactant"),
    product: Path = typer.Option(..., "--product"),
    work_dir: Path = typer.Option(..., "--work-dir"),
    charge: int = typer.Option(..., "--charge"),
    mult: int = typer.Option(..., "--mult"),
    fingerprint: Optional[Path] = typer.Option(None, "--fingerprint"),
    l0: str = typer.Option("XTB2", "--l0"),
    l1: str = typer.Option("r2SCAN-3c", "--l1"),
    l2: str = typer.Option("M06/Def2-SVP", "--l2"),
    profile: str = typer.Option("medium", "--profile"),
    execution_mode: str = typer.Option("array", "--mode"),
    out: Path = typer.Option("tsgen_plan.yaml", "--out"),
):
    plan_dict = {
        "reactant": str(reactant),
        "product": str(product),
        "work_dir": str(work_dir),
        "charge": charge,
        "mult": mult,
        "l0_method": l0,
        "l1_method": l1,
        "l2_method": l2,
        "execution_mode": execution_mode,
        "profile": profile,
        "fingerprint_file": str(fingerprint) if fingerprint else None,
    }

    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan_dict, indent=2), encoding="utf-8")
    typer.secho(f"TSGen 2.0 plan written → {out}", fg=typer.colors.GREEN)


@tsgen2_app.command("run")
def tsgen2_run(plan_file: Path):
    plan_file = plan_file.expanduser().resolve()
    if not plan_file.is_file():
        raise typer.BadParameter(f"Plan does not exist: {plan_file}")

    try:
        if plan_file.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        else:
            data = json.loads(plan_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise typer.BadParameter(f"Failed to parse plan file: {e}")

    plan = TSGenPlan(**data)
    controller = TSGenController(plan)
    out = controller.run()

    typer.echo(json.dumps(out, indent=2))

    if out.get("status") == "success":
        typer.secho("TSGen 2.0 pipeline completed.", fg=typer.colors.GREEN)
    else:
        typer.secho("TSGen 2.0 pipeline finished with issues.", fg=typer.colors.YELLOW)


# ==========================================================
# Helper Utilities
# ==========================================================


def _get(d: dict, path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def _sanitize_name(name: str) -> str:
    return _slugify(name)


def _repo_templates_root() -> Path:
    return Path(__file__).resolve().parents[2] / "templates"


def _coerce_value(v: str) -> Any:
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def _load_geometry_block(geometry_file: Optional[Path], geometry_literal: Optional[str]):
    if geometry_file:
        text = geometry_file.read_text(encoding="utf-8").strip().splitlines()
        if len(text) >= 2 and text[0].isdigit():
            text = text[2:]
        return "\n".join(text) + "\n"
    if geometry_literal:
        return geometry_literal.replace("\\n", "\n")
    return None


def _flatten_record(d: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten nested dict/list records into a single dict suitable for CSV.

    - Dicts are expanded recursively using dot-delimited keys.
    - Lists of dicts are expanded using positional indices (e.g., key.0.field).
    - Lists of scalars are JSON-encoded into a single cell.
    """
    if out is None:
        out = {}

    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"

        if isinstance(v, dict):
            _flatten_record(v, prefix=key, out=out)

        elif isinstance(v, list):
            if all(isinstance(x, dict) for x in v):
                for i, x in enumerate(v):
                    _flatten_record(x, prefix=f"{key}.{i}", out=out)
            else:
                out[key] = json.dumps(v)

        else:
            out[key] = v

    return out


# ==========================================================
# Sentinel utilities (exclusive state)
# ==========================================================

def _unlink_if_exists(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        return
    except IsADirectoryError:
        return


def _set_exclusive_sentinel(job_dir: Path, new_name: str, *, all_names: List[str], dry_run: bool = False) -> None:
    '''
    Enforce that exactly one sentinel from `all_names` exists in `job_dir` by:
      1) removing all sentinels in all_names
      2) touching `new_name`
    '''
    for nm in all_names:
        tgt = job_dir / nm
        if dry_run:
            if tgt.exists():
                typer.echo(f"Would remove {tgt}")
        else:
            _unlink_if_exists(tgt)

    tgt = job_dir / new_name
    if dry_run:
        typer.echo(f"Would touch {tgt}")
    else:
        tgt.write_text("", encoding="utf-8")


# ==========================================================

# ==========================================================
# GJF → XYZ helper (Gaussian input from GaussView)
# ==========================================================

def _parse_gjf_cartesian_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Parse Gaussian .gjf/.com input and extract one or more Cartesian coordinate blocks.

    Assumptions (covers the common GaussView6 export):
      - Link0 lines start with '%'
      - Route section starts with '#'
      - Title section is followed by a blank line
      - Then a line: "<charge> <multiplicity>"
      - Then Cartesian coordinates: "El  x  y  z" until a blank line or '--Link1--'

    Returns a list of blocks, each:
      {"charge": int, "mult": int, "atoms": [(sym, x, y, z), ...]}
    If no Cartesian block is found, returns [].
    """
    lines = text.splitlines()
    blocks: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)

    def _is_blank(s: str) -> bool:
        return not s.strip()

    def _is_charge_mult(s: str) -> Optional[tuple]:
        s2 = s.strip()
        if not s2:
            return None
        parts = s2.split()
        if len(parts) < 2:
            return None
        try:
            ch = int(parts[0])
            mu = int(parts[1])
            return ch, mu
        except Exception:
            return None

    def _is_cart_line(s: str) -> Optional[tuple]:
        # Accept: "C 0.0 1.0 2.0" or "Cl   -0.1 0.2 0.3"
        parts = s.strip().split()
        if len(parts) < 4:
            return None
        sym = parts[0]
        # Sym should be alphabetic (allow e.g. 'Cl', 'Br')
        if not re.match(r"^[A-Za-z]{1,3}$", sym):
            return None
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            return sym, x, y, z
        except Exception:
            return None

    while i < n:
        # Skip to a route line or Link1 separator; GaussView usually has one route line.
        if lines[i].strip().startswith("--Link1--"):
            i += 1
            continue

        # Find the charge/mult line by scanning; we don't attempt to fully parse the header.
        cm = _is_charge_mult(lines[i])
        if cm is None:
            i += 1
            continue

        charge, mult = cm
        # The coordinate block begins after the charge/mult line.
        j = i + 1
        atoms = []
        while j < n:
            if lines[j].strip().startswith("--Link1--"):
                break
            if _is_blank(lines[j]):
                break
            cart = _is_cart_line(lines[j])
            if cart is None:
                # Non-cartesian content (e.g., Z-matrix) → abort this candidate block
                atoms = []
                break
            atoms.append(cart)
            j += 1

        if atoms:
            blocks.append({"charge": charge, "mult": mult, "atoms": atoms})
            i = j
        else:
            i += 1

    return blocks


def _write_xyz(path: Path, atoms: List[tuple], comment: str = "") -> None:
    # XYZ format: natoms, comment, then lines "El x y z"
    nat = len(atoms)
    lines = [str(nat), comment]
    for sym, x, y, z in atoms:
        lines.append(f"{sym:<3s} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command("gjf-to-xyz")
def gjf_to_xyz_cmd(
    root: Path = typer.Argument(Path("."), help="File or directory to search for .gjf/.com files."),
    recurse: bool = typer.Option(True, "--recurse/--no-recurse", help="Recurse into subdirectories."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing .xyz outputs."),
    suffix: str = typer.Option("", "--suffix", help="Suffix to append to output stem (e.g., '_geom')."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be written; do not write files."),
):
    """
    Convert Gaussian input files (*.gjf, *.com) (e.g., from GaussView6) into XYZ files.

    - For each input, writes <stem><suffix>.xyz in the same directory.
    - Uses the first Cartesian coordinate block found (GaussView exports Cartesian by default).
    - If a file contains multiple Link1 sections, this converts each section to:
        <stem><suffix>.xyz, <stem><suffix>_link2.xyz, <stem><suffix>_link3.xyz, ...
    """
    root = root.expanduser().resolve()
    if not suffix:
        suffix = ""
    if suffix and not suffix.startswith("_"):
        # keep filenames readable without surprising users
        suffix = "_" + suffix

    if root.is_file():
        targets = [root]
    else:
        if not root.is_dir():
            raise typer.BadParameter(f"Not a file or directory: {root}")
        patterns = ["*.gjf", "*.com"]
        it = (root.rglob if recurse else root.glob)
        targets = []
        for pat in patterns:
            targets.extend([p for p in it(pat) if p.is_file()])
        targets = sorted(set(targets))

    if not targets:
        typer.secho("No .gjf/.com files found.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    converted = 0
    skipped = 0

    for gjf in targets:
        text = gjf.read_text(encoding="utf-8", errors="ignore")
        blocks = _parse_gjf_cartesian_blocks(text)
        if not blocks:
            typer.secho(f"[SKIP] No Cartesian coordinates found in {gjf}", fg=typer.colors.YELLOW)
            skipped += 1
            continue

        for bi, blk in enumerate(blocks, start=1):
            stem = gjf.stem + suffix
            out_name = f"{stem}.xyz" if bi == 1 else f"{stem}_link{bi}.xyz"
            out_path = gjf.parent / out_name

            if out_path.exists() and not overwrite:
                typer.secho(f"[SKIP] Exists (use --overwrite): {out_path}", fg=typer.colors.YELLOW)
                skipped += 1
                continue

            comment = f"from {gjf.name} | charge={blk['charge']} mult={blk['mult']}"
            if dry_run:
                typer.echo(f"Would write {out_path} ({len(blk['atoms'])} atoms)")
            else:
                _write_xyz(out_path, blk["atoms"], comment=comment)
            converted += 1

    typer.secho(f"Converted {converted} file(s); skipped {skipped}.", fg=typer.colors.GREEN)


# ORCA Tools
# ==========================================================


@app.command("orca-info")
def cli_orca_info(path: Path):
    info = collect_job_record(path)
    typer.echo(json.dumps(info, indent=2))


@app.command("orca-make-queues")
def cli_orca_make_queues(
    jsonl_path: Path,
    imag_list: Optional[Path] = None,
    failed_list: Optional[Path] = None,
    out_csv: Optional[Path] = None,
):
    jsonl_path = jsonl_path.expanduser().resolve()
    n_rows, n_imag, n_failed = make_queues(
        jsonl_path=jsonl_path,
        imag_list=imag_list,
        failed_list=failed_list,
        out_csv=out_csv,
    )
    typer.secho(f"Classified {n_rows} jobs (imag={n_imag}, failed={n_failed})", fg=typer.colors.GREEN)


@app.command("orca-nudge-imag")
def cli_orca_nudge_imag(list_path: Path, step: float = 0.1):
    list_path = list_path.expanduser().resolve()
    nudge_ifreq_jobs(list_path, step=step)


@app.command("orca-rebuild-failed")
def cli_orca_rebuild_failed(list_path: Path):
    list_path = list_path.expanduser().resolve()
    rebuild_failed_jobs(list_path)


# ==========================================================
# Results tools
# ==========================================================


@app.command("results-parquet")
def results_parquet(jsonl_path: Path, out_parquet: Optional[Path] = None):
    df = jsonl_to_parquet(jsonl_path, out_parquet)
    typer.secho(f"Wrote parquet with {len(df)} rows", fg=typer.colors.GREEN)


@app.command("results-csv")
def results_csv(jsonl_path: Path, out_csv: Path):
    df = _pd.read_json(jsonl_path, lines=True)
    flat = [_flatten_record(r) for r in df.to_dict(orient="records")]
    df2 = _pd.DataFrame(flat)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_csv, index=False)
    typer.secho(f"Wrote {out_csv}", fg=typer.colors.GREEN)


# ==========================================================
# Provenance
# ==========================================================


def _resolve_capture_snapshot_fn():
    for name in ("capture_snapshot", "make_snapshot", "get_snapshot", "snapshot", "build_provenance"):
        fn = getattr(snap_mod, name, None)
        if callable(fn):
            return fn

    import hashlib
    import platform
    import sys

    try:
        import importlib.metadata as importlib_metadata
    except Exception:  # pragma: no cover
        import importlib_metadata  # type: ignore

    def _fallback():
        pkg = "labtools"
        try:
            ver = importlib_metadata.version(pkg)
        except Exception:
            ver = None
        env = {k: v for k, v in os.environ.items() if k.startswith(("SLURM_", "EBROOT", "CC_", "PBS_"))}
        rec = {
            "host": platform.node(),
            "system": {"platform": platform.platform(), "python": sys.version},
            "package": {"name": pkg, "version": ver},
            "env": env,
        }
        rec["env_hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
        return rec

    return _fallback


CAPTURE_SNAPSHOT_FN = _resolve_capture_snapshot_fn()


@app.command("prov-snapshot")
def prov_snapshot(out: Optional[Path] = None):
    prov = CAPTURE_SNAPSHOT_FN()
    if out:
        out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
        typer.secho(f"Wrote {out}", fg=typer.colors.GREEN)
    else:
        typer.echo(json.dumps(prov, indent=2))


# ==========================================================
# Energetic Span
# ==========================================================


def _resolve_energetic_span_fn():
    for name in ("energetic_span", "compute_energetic_span", "compute_span", "calc_energetic_span"):
        fn = getattr(es_mod, name, None)
        if callable(fn):
            return fn
    raise RuntimeError("Could not find energetic-span function")


ES_FN = _resolve_energetic_span_fn()


@app.command("es")
def energetic_span_cmd(csv_path: Path, out_csv: Optional[Path] = None):
    df = _pd.read_csv(csv_path)
    span_df = ES_FN(df)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        span_df.to_csv(out_csv, index=False)
        typer.secho(f"Wrote {out_csv}", fg=typer.colors.GREEN)
    else:
        typer.echo(span_df.to_string(index=False))


# ==========================================================
# render-plan (legacy)
# ==========================================================


@app.command("render-plan")
def render_plan_cmd(
    plan: Path = typer.Option(..., "--plan"),
    out: Optional[Path] = typer.Option(None, "--outdir"),
    only: List[str] = typer.Option(None, "--only"),
    overwrite: bool = False,
):
    out = out or Path("build/inputs")
    wanted = None
    if only:
        expanded: List[str] = []
        for entry in only:
            expanded.extend(v for v in entry.split(",") if v)
        wanted = expanded
    if render_plan_jobs is None:
        raise typer.BadParameter(
            "Legacy 'render-plan' is unavailable (render_plan_jobs missing). "
            "Use: forge plan render --plan <plan.jsonl> --outdir <dir>"
        )
    written = render_plan_jobs(plan.expanduser().resolve(), out, only=wanted, overwrite=overwrite)
    typer.echo(f"Rendered {len(written)} inputs → {out}")


# ==========================================================
# PREP
# ==========================================================

JOB_TEMPLATES = {
    "optfreq": "orca_optfreq.inp.j2",
    "sp-triplet": "orca_sp_triplet.inp.j2",
    "nmr": "orca_nmr.inp.j2",
}


@app.command("prep")
def prep(
    job_type: str,
    name: str,
    charge: int,
    multiplicity: int,
    geometry_file: Optional[Path] = None,
    geometry_literal: Optional[str] = None,
    set: List[str] = typer.Option(None, "--set"),
    time_str: str = "24:00:00",
    cpus_per_task: int = 8,
    mem: str = "16G",
):
    jt = job_type.strip().lower()
    if jt not in JOB_TEMPLATES:
        raise typer.BadParameter(f"Unknown job_type '{jt}'")
    tpl_inp = JOB_TEMPLATES[jt]
    jobdir = Path(f"{_sanitize_name(name)}_{jt}")
    jobdir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {"charge": charge, "multiplicity": multiplicity, "pal": cpus_per_task}

    if set:
        for kv in set:
            if "=" not in kv:
                raise typer.BadParameter(f"Bad --set '{kv}'")
            k, v = kv.split("=", 1)
            params[k.strip()] = _coerce_value(v.strip())

    gb = _load_geometry_block(geometry_file, geometry_literal)
    if gb is None:
        raise typer.BadParameter("Geometry required.")
    params["geometry_block"] = gb

    inp_path = jobdir / f"{jt}.inp"
    render_template(tpl_inp, inp_path, params)

    orca_cmd = f'${{EBROOTORCA}}/orca "{inp_path.name}" > "{jt}.out"'

    sbatch_path = jobdir / "job.sbatch"
    from labtools.slurm.render import render_template as _rt

    _rt(
        _repo_templates_root() / "sbatch" / "single_orca_job.sbatch.j2",
        sbatch_path,
        {
            "job_name": jobdir.name,
            "time": time_str,
            "cpus_per_task": cpus_per_task,
            "mem": mem,
            "inp_basename": inp_path.name,
            "out_basename": f"{jt}.out",
            "orca_cmd": orca_cmd,
        },
    )

    prov = {
        "schema": "forge.provenance/1",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "job": {
            "job_type": jt,
            "job_name": jobdir.name,
            "base_name": name,
            "charge": charge,
            "multiplicity": multiplicity,
            "time_limit": time_str,
            "cpus_per_task": cpus_per_task,
            "mem": mem,
        },
        "inputs": {
            "plan": None,
            "template": tpl_inp,
            "geometry_file": str(geometry_file) if geometry_file else None,
            "geometry_literal": bool(geometry_literal),
            "extra_params": {k: v for k, v in params.items() if k != "geometry_block"},
        },
        "paths": {
            "jobdir": str(jobdir.resolve()),
            "inp": inp_path.name,
            "sbatch": "job.sbatch",
        },
    }

    try:
        (jobdir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    except Exception:
        pass

    typer.secho(f"Prepared {jobdir}/", fg=typer.colors.GREEN)


# ==========================================================
# SUBMIT
# ==========================================================


@app.command("submit")
def submit_job(
    inp: Path,
    profile: str = "medium",
    job_name: Optional[str] = None,
    cwd: Optional[Path] = None,
    job_chdir: Optional[Path] = None,
    validate_only: bool = False,
):
    inp = inp.expanduser().resolve()
    jobdir = inp.parent
    cwd = cwd or jobdir
    job_chdir = job_chdir or jobdir

    dispatch(
        inp,
        mode="job",
        profile=profile,
        job_name=job_name or inp.stem,
        submit_cwd=cwd,
        sbatch_chdir=job_chdir,
        validate_only=validate_only,
    )


@app.command("submit-array")
def submit_array(
    parents: List[Path],
    profile: str = "medium",
    job_name: Optional[str] = None,
    cwd: Optional[Path] = None,
    job_chdir: Optional[Path] = None,
    validate_only: bool = False,
):
    paths: List[Path] = []
    for parent in parents:
        parent = parent.expanduser().resolve()
        paths.extend(sorted(parent.glob("*.inp")))

    if not paths:
        raise typer.BadParameter("No *.inp files found")

    cwd = cwd or parents[0].resolve()
    job_chdir = job_chdir or parents[0].resolve()

    dispatch(
        paths,
        mode="array",
        profile=profile,
        job_name=job_name or "array",
        submit_cwd=cwd,
        sbatch_chdir=job_chdir,
        validate_only=validate_only,
    )

    typer.secho(f"Submitted array ({len(paths)} inputs)", fg=typer.colors.GREEN)


@app.command("submit-drone")
def submit_drone(
    queue_dir: Path = typer.Option(..., "--queue-dir"),
    n: int = typer.Option(1),
    profile: str = typer.Option("medium"),
    nprocs: int = typer.Option(8),
    mem_per_cpu: str = typer.Option("4G"),
    time: str = typer.Option("00:10:00"),
    validate_only: bool = typer.Option(False, "--validate-only"),
):
    from labtools.slurm.render import render_template as _render_template

    queue_dir = queue_dir.resolve()

    def _find_tpl(rel: str) -> Path:
        here = Path(__file__).resolve()
        for p in here.parents:
            cand = p / "templates" / "sbatch" / rel
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Missing sbatch template: templates/sbatch/{rel}")

    tpl = _find_tpl("drone_worker.sbatch.j2")

    for i in range(n):
        name = f"drone-{i+1}" if n > 1 else "drone"

        params = {
            "job_name": name,
            "jobdir": str(queue_dir),
            "time": time,
            "nprocs": nprocs,
            "mem_per_cpu": mem_per_cpu,
            "QUEUE_DIR": str(queue_dir),
            "SLEEP_SECS": 60,
        }

        if validate_only:
            preview = _render_template(tpl, None, params, return_text=True)
            typer.echo(preview)
            continue

        sbatch_path = queue_dir / f"{name}.sbatch"
        _render_template(tpl, sbatch_path, params)
        subprocess.run(["sbatch", str(sbatch_path)], check=True)


# ==========================================================
# CSV → PLAN (extended: emit jobs OR planentries)
# ==========================================================


@app.command("plan-from-csv")
def plan_from_csv(
    csv_path: Path = typer.Option(..., "--csv"),
    mapping_path: Path = typer.Option(..., "--mapping"),
    outdir: Path = typer.Option("plan", "--outdir"),
    mode: str = typer.Option("product", "--mode"),
    id_field: str = typer.Option("id", "--id-field"),
    emit: str = typer.Option("jobs", "--emit"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    csv_path = csv_path.expanduser().resolve()
    mapping_path = mapping_path.expanduser().resolve()
    outdir = outdir.expanduser().resolve()

    if not csv_path.is_file():
        raise typer.BadParameter(f"CSV not found: {csv_path}")
    if not mapping_path.is_file():
        raise typer.BadParameter(f"Mapping not found: {mapping_path}")

    import yaml

    mapping_spec = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    mapping = build_mapping(mapping_spec)

    rows = read_csv_rows(csv_path)

    jobs: List[Dict[str, Any]] = []
    for row in rows:
        job_doc = row_to_job(row, mapping)
        axes = gather_axes(job_doc, mapping_spec.get("axes", []))
        expanded = expand_zip(job_doc, axes) if mode == "zip" else expand_product(job_doc, axes)
        jobs.extend(expanded)

    if not jobs:
        raise typer.BadParameter("No jobs produced from CSV + mapping")

    for job in jobs:
        if "structure" not in job:
            raw = job.get("_row_raw", {})
            if "structure" in raw:
                job["structure"] = raw["structure"]

    emit = (emit or "jobs").strip().lower()
    if emit not in {"jobs", "planentries"}:
        raise typer.BadParameter("--emit must be one of: jobs, planentries")

    if emit == "planentries":
        schema_name = str(mapping_spec.get("schema", "csvpipe"))
        schema_version = int(mapping_spec.get("version", 1))
        task = str(mapping_spec.get("task", "orca"))

        entries = [
            job_to_planentry(
                job,
                schema_name=schema_name,
                schema_version=schema_version,
                task=task,
                index=i,
            )
            for i, job in enumerate(jobs)
        ]

        outpath = outdir / "planentries.jsonl"
        text = emit_planentries_jsonl(entries, outpath, dry_run=dry_run)

        if dry_run:
            typer.echo(text or "")
        else:
            typer.secho(
                f"Wrote {len(entries)} PlanEntries → {outpath}",
                fg=typer.colors.GREEN,
            )
        return

    written = emit_job_yaml_files(
        jobs,
        outdir=outdir,
        id_field=id_field,
        provenance={
            "source": "plan-from-csv",
            "csv": str(csv_path),
            "mapping": str(mapping_path),
            "mode": mode,
        },
        dry_run=dry_run,
    )

    if dry_run:
        typer.echo("\n---\n".join(written or []))
    else:
        typer.secho(
            f"Wrote {len(jobs)} plan YAMLs → {outdir}",
            fg=typer.colors.GREEN,
        )


# ==========================================================
# FORGE WATCH (deluxe v1: snapshot-only)
# ==========================================================

WATCH_SENTINELS = ("READY", "STARTED", "DONE", "FAIL", "GOOD", "BAD")


def _watch_state_dir(root: Path) -> Path:
    return root / ".forge_watch"


def _is_job_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    # quick ignores
    name = d.name
    if name.startswith('.'):
        return False
    for s in WATCH_SENTINELS:
        if (d / s).is_file():
            return True
    if (d / 'plan_entry.json').is_file():
        return True
    if (d / 'job.inp').is_file() or (d / 'job.sbatch').is_file():
        return True
    if (d / 'watch.json').is_file() or (d / '.job.out.tail').is_file():
        return True
    # any .out at all (lightweight glob)
    try:
        for _ in d.glob('*.out'):
            return True
    except Exception:
        pass
    return False


def _discover_job_dirs(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    jobs: List[Path] = []
    for d in sorted([p for p in root.rglob('*') if p.is_dir()]):
        # ignore hidden segments
        if any(part.startswith('.') for part in d.parts):
            continue
        if _is_job_dir(d):
            jobs.append(d)
    return jobs


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort JSON reader.
    Returns a dict on success; returns None on missing/invalid/partial JSON or non-dict payloads.
    """
    try:
        txt = path.read_text(encoding='utf-8', errors='ignore')
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_tail(path: Path, max_chars: int = 20000) -> str:
    try:
        s = path.read_text(encoding='utf-8', errors='ignore')
        return s[-max_chars:]
    except Exception:
        return ''



def _parse_iso8601_utc(ts: str) -> Optional[datetime.datetime]:
    """Parse ISO-8601 timestamps that may end with 'Z'. Returns aware UTC datetime."""
    if not ts or not isinstance(ts, str):
        return None
    s = ts.strip()
    try:
        # Handle trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception:
        return None


def _watch_age_secs(watch: Dict[str, Any], now_utc: datetime.datetime) -> Optional[int]:
    if not isinstance(watch, dict):
        return None
    ts = watch.get("ts") or watch.get("timestamp") or watch.get("time")
    dt = _parse_iso8601_utc(str(ts)) if ts else None
    if dt is None:
        return None
    delta = now_utc - dt
    try:
        return int(delta.total_seconds())
    except Exception:
        return None


def _infer_stage_from_tail(tail: str) -> str:
    if not tail:
        return 'NO_OUTPUT'
    if 'ORCA TERMINATED NORMALLY' in tail:
        return 'DONE'
    if re.search(r"Segmentation fault|FATAL ERROR|ORCA finished by error termination", tail, re.I):
        return 'FAIL'
    if re.search(r"VIBRATIONAL ANALYSIS|THERMOCHEMISTRY", tail, re.I):
        return 'FREQ'
    if re.search(r"GEOMETRY OPTIMIZATION|OPTIMIZATION CYCLE", tail, re.I):
        return 'OPT'
    if re.search(r"SCF ITERATIONS|SCF CONVERGENCE", tail, re.I):
        return 'SCF'
    return 'RUNNING'


def _collect_sentinels(d: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for s in WATCH_SENTINELS:
        out[s.lower()] = (d / s).is_file()
    return out


def _collect_watch_record(
    root: Path,
    job_dir: Path,
    *,
    now_utc: datetime.datetime,
    stale_secs: int,
    squeue_by_id: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    rel = str(job_dir.relative_to(root)) if job_dir.is_relative_to(root) else str(job_dir)
    rec: Dict[str, Any] = {
        "job_dir": str(job_dir),
        "rel": rel,
        "name": job_dir.name,
        "sentinels": _collect_sentinels(job_dir),
    }

    # Prefer heartbeat payload
    wj = job_dir / "watch.json"
    watch = None
    if wj.is_file():
        watch = _read_json(wj)
        if isinstance(watch, dict):
            rec["watch"] = watch

    # Tail: prefer heartbeat tail
    tail_path = job_dir / ".job.out.tail"
    tail = ""
    if tail_path.is_file():
        tail = _read_tail(tail_path)
        rec["out_tail"] = tail

    # If no heartbeat tail exists, optionally pull from latest .out (cheap)
    if not tail:
        outs = sorted(job_dir.glob("*.out"))
        if outs:
            try:
                out_file = max(outs, key=lambda p: p.stat().st_mtime)
            except Exception:
                out_file = outs[0]
            rec["out_file"] = str(out_file)
            try:
                txt = out_file.read_text(encoding="utf-8", errors="ignore")
                tail = txt[-20000:]
            except Exception:
                tail = ""

    # Stage: explicit heartbeat stage wins; otherwise infer from output tail
    stage = None
    if isinstance(watch, dict) and watch.get("stage"):
        stage = str(watch["stage"])
    if not stage:
        stage = _infer_stage_from_tail(tail)

    rec["stage_raw"] = stage

    # Age / staleness (heartbeat-driven; falls back to mtime of .job.out.tail if needed)
    age = None
    if isinstance(watch, dict):
        age = _watch_age_secs(watch, now_utc=now_utc)
    if age is None and tail_path.is_file():
        try:
            age = int((now_utc - datetime.datetime.fromtimestamp(tail_path.stat().st_mtime, tz=datetime.timezone.utc)).total_seconds())
        except Exception:
            age = None
    rec["watch_age_secs"] = age

    # Scheduler context (optional)
    sched = None
    job_id = None
    if isinstance(watch, dict):
        job_id = watch.get("slurm_job_id") or watch.get("job_id")
        if job_id is not None:
            job_id = str(job_id)
    if squeue_by_id and job_id:
        sched = squeue_by_id.get(job_id)
    if job_id:
        rec["slurm_job_id"] = job_id
    if sched:
        rec["scheduler"] = sched

    # Normalize stage into a high-level status label
    sent = rec["sentinels"]
    has_done = bool(sent.get("done")) or bool(sent.get("good"))
    has_fail = bool(sent.get("fail")) or bool(sent.get("bad"))

    if has_done:
        status = "DONE"
    elif has_fail:
        status = "FAIL"
    else:
        status = stage

    # Stale / hung / scheduler-lost logic (only meaningful for running-ish stages)
    runningish = str(stage).lower() in {"running", "staged", "syncing", "post_orca", "scf", "opt", "freq"}
    is_stale = bool(age is not None and age > int(stale_secs))

    if not has_done and not has_fail and runningish and is_stale:
        # If scheduler says this job id is still active -> heartbeat stalled (hung or I/O broken)
        if sched and str(sched.get("state", "")).upper() in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING"}:
            status = "HUNG"
        # If scheduler does not see it at all -> likely scheduler-lost / node failure / cancelled
        elif job_id:
            status = "SCHEDULER_LOST"
        else:
            status = "STALE"

    rec["status"] = status
    rec["stale"] = bool(is_stale)

    return rec



def _parse_squeue() -> List[Dict[str, str]]:
    """Best-effort: returns list of dict rows; non-fatal on error."""
    try:
        fmt = '%i|%T|%j|%N|%M|%l|%R'
        cmd = ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', fmt]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            return []
        rows: List[Dict[str, str]] = []
        for line in res.stdout.splitlines():
            parts = line.split('|')
            if len(parts) != 7:
                continue
            rows.append({
                'job_id': parts[0],
                'state': parts[1],
                'job_name': parts[2],
                'node': parts[3],
                'elapsed': parts[4],
                'time_limit': parts[5],
                'reason': parts[6],
            })
        return rows
    except Exception:
        return []


def _assign_groups(root: Path, job_dirs: List[Path], group_specs: List[str], auto_parent_min: int) -> Dict[str, List[Path]]:
    """Group specs: ['Name=glob', ...] where glob matches RELATIVE paths from root."""
    groups: Dict[str, List[Path]] = {}

    # explicit groups
    explicit: List[tuple[str, str]] = []
    for spec in group_specs or []:
        if '=' not in spec:
            continue
        name, pat = spec.split('=', 1)
        name = name.strip() or 'group'
        pat = pat.strip()
        if not pat:
            continue
        explicit.append((name, pat))

    assigned: Dict[Path, str] = {}
    for d in job_dirs:
        rel = str(d.relative_to(root)) if d.is_relative_to(root) else str(d)
        for name, pat in explicit:
            if fnmatch.fnmatch(rel, pat):
                groups.setdefault(name, []).append(d)
                assigned[d] = name
                break

    # auto-parent buckets for unassigned
    parent_counts: Dict[Path, int] = {}
    for d in job_dirs:
        if d in assigned:
            continue
        parent = d.parent
        parent_counts[parent] = parent_counts.get(parent, 0) + 1

    for d in job_dirs:
        if d in assigned:
            continue
        parent = d.parent
        if parent_counts.get(parent, 0) >= auto_parent_min:
            name = f"parent:{parent.name}"
            groups.setdefault(name, []).append(d)
            assigned[d] = name

    # loose bucket
    loose = [d for d in job_dirs if d not in assigned]
    if loose:
        groups['loose'] = loose

    return groups


@watch_app.command('snapshot')
def watch_snapshot(
    root: Path = typer.Option(Path('.'), '--root', help='Root directory to watch.'),
    out: Optional[Path] = typer.Option(None, '--out', help='Write snapshot JSON to this file.'),
    include_scheduler: bool = typer.Option(True, '--scheduler/--no-scheduler', help='Include best-effort squeue data.'),
    group: List[str] = typer.Option([], '--group', help='Grouping spec: Name=glob (glob matches paths relative to root). Can repeat.'),
    auto_parent_min: int = typer.Option(20, '--auto-parent-min', help='Auto-bucket parent dirs with at least this many jobs.'),
    stale_secs: int = typer.Option(1800, '--stale-secs', help='If heartbeat age exceeds this many seconds (and job not DONE/FAIL), classify as HUNG/SCHEDULER_LOST.'),
    write_state: bool = typer.Option(False, '--write-state', help='Write .forge_watch/latest.json under root.'),
):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"Not a directory: {root}")

    job_dirs = _discover_job_dirs(root)

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    squeue_rows: List[Dict[str, str]] = []
    squeue_by_id: Dict[str, Dict[str, str]] = {}
    if include_scheduler:
        squeue_rows = _parse_squeue()
        squeue_by_id = {r.get("job_id"): r for r in squeue_rows if r.get("job_id")}

    jobs = [
        _collect_watch_record(
            root,
            d,
            now_utc=now_utc,
            stale_secs=int(stale_secs),
            squeue_by_id=squeue_by_id if include_scheduler else None,
        )
        for d in job_dirs
    ]

    groups = _assign_groups(root, job_dirs, group_specs=group, auto_parent_min=int(auto_parent_min))
    group_summary: Dict[str, Dict[str, Any]] = {}
    for gname, ds in groups.items():
        stages: Dict[str, int] = {}
        for d in ds:
            # find rec
            rec = next((r for r in jobs if r['job_dir'] == str(d)), None)
            if not rec:
                continue
            st = rec.get('status', rec.get('stage', 'UNKNOWN'))
            stages[st] = stages.get(st, 0) + 1
        group_summary[gname] = {'n_jobs': len(ds), 'stages': stages}

    snapshot: Dict[str, Any] = {
        'schema': 'forge.watch.snapshot/1',
        'ts': datetime.datetime.utcnow().isoformat() + 'Z',
        'root': str(root),
        'n_jobs': len(jobs),
        'jobs': jobs,
        'groups': {k: [str(p) for p in v] for k, v in groups.items()},
        'group_summary': group_summary,
    }

    if include_scheduler:
        snapshot['squeue'] = squeue_rows

    text = json.dumps(snapshot, indent=2)

    if out:
        out = out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n', encoding='utf-8')
        typer.secho(f"Wrote snapshot → {out}", fg=typer.colors.GREEN)
    else:
        typer.echo(text)

    if write_state:
        sd = _watch_state_dir(root)
        sd.mkdir(parents=True, exist_ok=True)
        latest = sd / 'latest.json'
        latest.write_text(text + '\n', encoding='utf-8')
        typer.secho(f"Updated state → {latest}", fg=typer.colors.GREEN)


def main() -> None:
    app()


if __name__ == "__main__":
    main()


# ==========================================================
# Single-job CLI (create/import) + Operational tooling (scan/mark/clean)
# ==========================================================

from typing import Tuple


def _read_xyz_geom_lines(xyz_path: Path) -> List[str]:
    xyz_path = xyz_path.expanduser().resolve()
    if not xyz_path.is_file():
        raise typer.BadParameter(f"XYZ file not found: {xyz_path}")
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 3:
        try:
            int(lines[0].strip())
            return [ln.rstrip() for ln in lines[2:] if ln.strip()]
        except Exception:
            pass
    return [ln.rstrip() for ln in lines if ln.strip()]


def _orca_template_for_task(task: str) -> Path:
    task = task.strip().lower()
    root = _templates_root() / "orca"
    mapping = {
        "sp": "orca_sp.inp.j2",
        "opt": "orca_opt.inp.j2",
        "freq": "orca_freq.inp.j2",
        "optfreq": "orca_optfreq.inp.j2",
    }
    if task not in mapping:
        raise typer.BadParameter(f"Unsupported task: {task}. Choose from: {', '.join(sorted(mapping))}")
    tpl = root / mapping[task]
    if not tpl.is_file():
        raise typer.BadParameter(f"Template not found for task '{task}': {tpl}")
    return tpl


def _sbatch_template() -> Path:
    tpl = _templates_root() / "sbatch" / "single_orca_job.sbatch.j2"
    if not tpl.is_file():
        raise typer.BadParameter(f"SBATCH template not found: {tpl}")
    return tpl


def _default_job_name_from_xyz(xyz: Path, task: str) -> str:
    return f"{xyz.stem}_{task}"


def _build_minimal_payload(
    *,
    xyz: Path,
    task: str,
    method: str,
    charge: int,
    mult: int,
    basis: Optional[str],
    flags: List[str],
    restart_flags: List[str],
    nprocs: Optional[int],
    maxcore_mb: Optional[int],
    time: Optional[int],
    maxiter: Optional[int],
    cpcm_eps: Optional[float],
    cpcm_refrac: Optional[float],
) -> Dict[str, Any]:
    geom_lines = _read_xyz_geom_lines(xyz)

    geom = {
        "maxiter": int(maxiter) if maxiter is not None else 200,
        "constraints": [],
        "restart": False,
        "Restart": False,
    }

    payload: Dict[str, Any] = {
        "task": task,
        "method": method,
        "basis": (basis or ""),
        "flags": list(flags or []),
        "restart_flags": list(restart_flags or []),
        "pal": int(nprocs) if nprocs is not None else None,
        "maxcore_mb": int(maxcore_mb) if maxcore_mb is not None else 2000,
        "scf": {},
        "geom": geom,
        "freq": {"override": {}},
        "cpcm": None,
        "charge": int(charge),
        "mult": int(mult),
        "geom_lines": geom_lines,
        # explicit defaults for template compatibility
        "restart": {"enabled": False, "file": None, "flags": []},
    }

    if cpcm_eps is not None:
        payload["cpcm"] = {
            "epsilon": float(cpcm_eps),
            "refrac": float(cpcm_refrac) if cpcm_refrac is not None else None,
        }

    return payload



def _mem_per_cpu_from_maxcore(maxcore_mb: Optional[int]) -> str:
    # Minimal deterministic default for SLURM --mem-per-cpu (string like "4G").
    # If maxcore_mb is unset, fall back to 4G.
    if maxcore_mb is None:
        return "4G"
    gb = (int(maxcore_mb) + 1023) // 1024  # ceil(MB/1024)
    gb = max(1, gb)
    return f"{gb}G"

def _safe_clear_dir(d: Path) -> None:
    import shutil
    for child in d.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_job_dir(
    *,
    final_dir: Path,
    payload: Dict[str, Any],
    task: str,
    job_name: str,
    xyz: Optional[Path],
    write_ready: bool,
    write_sbatch: bool,
    nprocs: Optional[int],
    walltime: str,
    mem_per_cpu: str,
    exists: str,
) -> None:
    import tempfile
    import shutil

    if final_dir.exists():
        if exists == "fail":
            raise typer.BadParameter(f"Job directory already exists: {final_dir}")
        if exists == "skip":
            return
        if exists == "overwrite":
            _safe_clear_dir(final_dir)
        else:
            raise typer.BadParameter("--exists must be one of: fail|skip|overwrite")
    else:
        final_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix=f".tmp_{job_name}_", dir=str(final_dir.parent)))
    try:
        tpl = _orca_template_for_task(task)
        inp_text = render_template(tpl, tmp_root / "job.inp", payload, return_text=True)
        (tmp_root / "job.inp").write_text(inp_text or "", encoding="utf-8")
        (tmp_root / "plan_entry.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        if xyz is not None:
            xyz = xyz.expanduser().resolve()
            if xyz.is_file():
                shutil.copy2(xyz, tmp_root / xyz.name)

        if write_sbatch:
            sbatch_params = {
                "job_dir": str(final_dir),
                "job_name": job_name,
                "nprocs": int(nprocs or 1),
                "time": walltime,
                "mem_per_cpu": mem_per_cpu,
                "SRC_DIR": str(final_dir),
                "inp_basename": "job.inp",
                "orca_cmd": '${EBROOTORCA}/orca "${INP_BN}" > "job.out"',
            }
            sb_text = render_template(_sbatch_template(), tmp_root / "job.sbatch", sbatch_params, return_text=True)
            (tmp_root / "job.sbatch").write_text(sb_text or "", encoding="utf-8")

        if write_ready:
            (tmp_root / "READY").write_text("", encoding="utf-8")

        # atomic-ish move
        if final_dir.exists() and exists == "overwrite":
            _safe_clear_dir(final_dir)
        tmp_root.replace(final_dir)
    finally:
        if tmp_root.exists() and tmp_root != final_dir:
            shutil.rmtree(tmp_root, ignore_errors=True)


@job_app.command("create")
def job_create(
    xyz: Path = typer.Option(..., "--xyz", help="Input structure in XYZ format."),
    task: str = typer.Option("optfreq", "--task", help="Job task: sp|opt|freq|optfreq."),
    method: str = typer.Option(..., "--method", help="ORCA bang-line method fragment, e.g. 'r2scan-3c'."),
    charge: int = typer.Option(0, "--charge"),
    mult: int = typer.Option(1, "--mult"),
    basis: Optional[str] = typer.Option(None, "--basis"),
    flag: List[str] = typer.Option([], "--flag"),
    restart_flag: List[str] = typer.Option([], "--restart-flag"),
    name: Optional[str] = typer.Option(None, "--name"),
    outdir: Path = typer.Option(Path("jobs"), "--outdir"),
    nprocs: Optional[int] = typer.Option(None, "--nprocs"),
    time: str = typer.Option("08:00:00", "--time"),
    mem_per_cpu: Optional[str] = typer.Option(None, "--mem-per-cpu", help="SLURM mem-per-cpu (e.g. 4G). Default derived from --maxcore-mb."),
    maxcore_mb: Optional[int] = typer.Option(None, "--maxcore-mb"),
    maxiter: Optional[int] = typer.Option(None, "--maxiter"),
    cpcm_eps: Optional[float] = typer.Option(None, "--cpcm-epsilon"),
    cpcm_refrac: Optional[float] = typer.Option(None, "--cpcm-refrac"),
    write_ready: bool = typer.Option(True, "--ready/--no-ready"),
    write_sbatch: bool = typer.Option(True, "--sbatch/--no-sbatch"),
    submit: bool = typer.Option(False, "--submit"),
    exists: str = typer.Option("fail", "--exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    xyz = xyz.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    if exists not in ("fail", "skip", "overwrite"):
        raise typer.BadParameter("--exists must be one of: fail|skip|overwrite")

    job_name = name or _default_job_name_from_xyz(xyz, task)
    final_dir = outdir / job_name

    payload = _build_minimal_payload(
        xyz=xyz,
        task=task,
        method=method,
        charge=charge,
        mult=mult,
        basis=basis,
        flags=flag,
        restart_flags=restart_flag,
        nprocs=nprocs,
        time=time,
        maxcore_mb=maxcore_mb,
        maxiter=maxiter,
        cpcm_eps=cpcm_eps,
        cpcm_refrac=cpcm_refrac,
    )

    mpc = mem_per_cpu or _mem_per_cpu_from_maxcore(maxcore_mb)

    if dry_run:
        tpl = _orca_template_for_task(task)
        render_template(tpl, Path("/dev/null"), payload, return_text=True)
        if write_sbatch:
            render_template(
                _sbatch_template(),
                Path("/dev/null"),
                {
                    "job_dir": str(final_dir),
                    "job_name": job_name,
                    "nprocs": int(nprocs or 1),
                    "time": time,
                    "mem_per_cpu": mpc,
                    "SRC_DIR": str(final_dir),
                    "inp_basename": "job.inp",
                    "orca_cmd": '${EBROOTORCA}/orca "${INP_BN}" > "job.out"',
                },
                return_text=True,
            )
        typer.secho("Dry-run OK: templates rendered successfully.", fg=typer.colors.GREEN)
        return

    _write_job_dir(
        final_dir=final_dir,
        payload=payload,
        task=task,
        job_name=job_name,
        xyz=xyz,
        write_ready=write_ready,
        write_sbatch=write_sbatch,
        nprocs=nprocs,
        walltime=time,
        mem_per_cpu=mpc,
        exists=exists,
    )

    typer.secho(f"Created job → {final_dir}", fg=typer.colors.GREEN)

    if submit:
        dispatch(str(final_dir))
        typer.secho("Submitted job.", fg=typer.colors.GREEN)


@job_app.command("import")
def job_import(
    plan_entry: Path = typer.Option(..., "--plan-entry", help="Path to a plan_entry.json describing a single job."),
    name: Optional[str] = typer.Option(None, "--name"),
    outdir: Path = typer.Option(Path("jobs"), "--outdir"),
    write_ready: bool = typer.Option(True, "--ready/--no-ready"),
    write_sbatch: bool = typer.Option(True, "--sbatch/--no-sbatch"),
    submit: bool = typer.Option(False, "--submit"),
    exists: str = typer.Option("fail", "--exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    plan_entry = plan_entry.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    if exists not in ("fail", "skip", "overwrite"):
        raise typer.BadParameter("--exists must be one of: fail|skip|overwrite")
    if not plan_entry.is_file():
        raise typer.BadParameter(f"plan_entry.json not found: {plan_entry}")

    payload = json.loads(plan_entry.read_text(encoding="utf-8"))
    task = str(payload.get("task") or payload.get("job_type") or "optfreq").lower()

    job_name = name or payload.get("id") or plan_entry.parent.name
    final_dir = outdir / str(job_name)

    walltime = "08:00:00"
    mpc = _mem_per_cpu_from_maxcore(int(payload.get("maxcore_mb") or 2000))

    if dry_run:
        tpl = _orca_template_for_task(task)
        render_template(tpl, Path("/dev/null"), payload, return_text=True)
        typer.secho("Dry-run OK: template rendered successfully.", fg=typer.colors.GREEN)
        return

    # attempt to locate structure file if referenced
    xyz = None
    sp = payload.get("structure") or payload.get("xyz")
    if sp:
        p = Path(sp).expanduser()
        if p.is_file():
            xyz = p

    _write_job_dir(
        final_dir=final_dir,
        payload=payload,
        task=task,
        job_name=str(job_name),
        xyz=xyz,
        write_ready=write_ready,
        write_sbatch=write_sbatch,
        nprocs=int(payload.get("pal") or 1),
        walltime=walltime,
        mem_per_cpu=mpc,
        exists=exists,
    )

    typer.secho(f"Imported job → {final_dir}", fg=typer.colors.GREEN)

    if submit:
        dispatch(str(final_dir))
        typer.secho("Submitted job.", fg=typer.colors.GREEN)


# --------------------------
# Operational tooling (scan/mark/clean)
# --------------------------

def _find_out_file(job_dir: Path, out_glob: str, ignore_glob: str) -> Optional[Path]:
    outs = sorted(job_dir.glob(out_glob))
    if ignore_glob:
        ignored = set(job_dir.glob(ignore_glob))
        outs = [p for p in outs if p not in ignored]
    for p in outs:
        if p.name == "job.out":
            return p
    return outs[0] if outs else None


def _orca_terminated_normally(lines: List[str]) -> bool:
    return any("ORCA TERMINATED NORMALLY" in ln for ln in lines)


def _scan_one(job_dir: Path, out_glob: str, ignore_glob: str) -> Dict[str, Any]:
    out_file = _find_out_file(job_dir, out_glob, ignore_glob)
    if out_file is None:
        return {"path": str(job_dir), "out_file": "", "terminated_normally": False, "opt_done": False, "freq_done": False, "n_imag": "", "status": "FAIL", "fail_reason": "no_out"}

    lines = out_file.read_text(errors="ignore").splitlines()
    term = _orca_terminated_normally(lines)

    opt_done = any("OPTIMIZATION RUN DONE" in ln for ln in lines)

    n_imag = None
    for ln in lines:
        m = re.search(r"Number of imaginary frequencies\s*:\s*(\d+)", ln)
        if m:
            n_imag = int(m.group(1))
            break
    freq_done = n_imag is not None

    # Intent detection
    head = "\n".join(lines[:80])
    has_opt_intent = bool(re.search(r"\bOpt\b", head, re.I))
    has_freq_intent = bool(re.search(r"\bFreq\b", head, re.I)) or any("VIBRATIONAL ANALYSIS" in ln for ln in lines)

    ok = term
    if has_opt_intent:
        ok = ok and opt_done
    if has_freq_intent:
        ok = ok and freq_done

    if ok:
        status, reason = "OK", ""
    else:
        status = "FAIL"
        if not term:
            reason = "not_terminated_normally"
        elif has_opt_intent and not opt_done:
            reason = "opt_incomplete"
        elif has_freq_intent and not freq_done:
            reason = "freq_incomplete"
        else:
            reason = "unknown"

    return {
        "path": str(job_dir),
        "out_file": str(out_file),
        "terminated_normally": term,
        "opt_done": opt_done,
        "freq_done": freq_done,
        "n_imag": "" if n_imag is None else n_imag,
        "status": status,
        "fail_reason": reason,
    }


@app.command("scan")
def scan_cmd(
    root: Path = typer.Argument(..., help="Root directory to scan."),
    out_glob: str = typer.Option("*.out", "--out-glob"),
    ignore_glob: str = typer.Option("*atom44.out", "--ignore"),
    csv: Path = typer.Option(Path("results.csv"), "--csv"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    root = root.expanduser().resolve()
    csv = csv.expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"Not a directory: {root}")

    rows: List[Dict[str, Any]] = []
    for d in sorted([p for p in root.rglob("*") if p.is_dir()]):
        if any(part.startswith(".") for part in d.parts):
            continue
        rec = _scan_one(d, out_glob, ignore_glob)
        if rec["fail_reason"] == "no_out":
            if not (d / "plan_entry.json").exists() and not any(d.glob("*.inp")):
                continue
        rows.append(rec)

    if dry_run:
        typer.echo(f"Would write {len(rows)} rows to {csv}")
        return

    import csv as _csv
    csv.parent.mkdir(parents=True, exist_ok=True)
    with csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["path"]
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    typer.secho(f"Wrote results → {csv}", fg=typer.colors.GREEN)


@app.command("mark")
def mark_cmd(
    csv: Path = typer.Option(..., "--csv"),
    good: str = typer.Option("GOOD", "--good"),
    bad: str = typer.Option("BAD", "--bad"),
    only_fail: bool = typer.Option(False, "--only-fail"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    csv = csv.expanduser().resolve()
    if not csv.is_file():
        raise typer.BadParameter(f"CSV not found: {csv}")
    import pandas as _pd
    df = _pd.read_csv(csv)

    touched = 0
    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip().upper()
        p = Path(str(row["path"]))
        if not p.is_dir():
            continue

        if status == "OK" and not only_fail:
            target = p / good
            new_name = good
        elif status == "FAIL":
            target = p / bad
            new_name = bad
        else:
            continue

        # Enforce exclusive state among common sentinels (+ caller-specified good/bad).
        sentinel_names = ["READY", "STARTED", "DONE", "FAIL", str(good), str(bad)]
        _set_exclusive_sentinel(p, str(new_name), all_names=sentinel_names, dry_run=dry_run)

        touched += 1

    typer.secho(f"Touched {touched} sentinel(s).", fg=typer.colors.GREEN)


@app.command("clean")
def clean_cmd(
    csv: Path = typer.Option(..., "--csv"),
    only_fail: bool = typer.Option(True, "--only-fail/--all"),
    keep: List[str] = typer.Option(["*.inp", "plan_entry.json"], "--keep"),
    write_ready: bool = typer.Option(True, "--ready/--no-ready"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    csv = csv.expanduser().resolve()
    if not csv.is_file():
        raise typer.BadParameter(f"CSV not found: {csv}")
    import pandas as _pd
    df = _pd.read_csv(csv)

    cleaned = 0
    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip().upper()
        if only_fail and status != "FAIL":
            continue
        p = Path(str(row["path"]))
        if not p.is_dir():
            continue

        keep_set = set()
        for pat in keep:
            for k in p.glob(pat):
                keep_set.add(k.resolve())

        for child in p.iterdir():
            if child.resolve() in keep_set:
                continue
            if dry_run:
                typer.echo(f"Would remove {child}")
                continue
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()

        if write_ready:
            # Enforce exclusive state: clean implies "READY" (not DONE/FAIL/GOOD/BAD/STARTED).
            sentinel_names = ["READY", "STARTED", "DONE", "FAIL", "GOOD", "BAD"]
            _set_exclusive_sentinel(p, "READY", all_names=sentinel_names, dry_run=dry_run)

        cleaned += 1

    typer.secho(f"Cleaned {cleaned} job directory(ies).", fg=typer.colors.GREEN)


@app.command("parse")
def parse_cmd(
    csv: Optional[Path] = typer.Option(
        None,
        "--csv",
        help="CSV produced by `forge scan`. Must contain columns: path, status.",
    ),
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Root directory containing job subdirectories (or a single job dir).",
    ),
    out_jsonl: Path = typer.Option(..., "--out-jsonl", help="Output JSONL path (one record per line)."),
    status: str = typer.Option("OK", "--status", help="(CSV mode) Which rows to parse: OK|FAIL|ALL."),
    only_sentinel: str = typer.Option(
        "ANY",
        "--only-sentinel",
        help="(Root mode) Only parse jobdirs with this sentinel: GOOD|BAD|DONE|FAIL|READY|STARTED|ANY.",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optional max number of jobs to parse."),
    skip_missing: bool = typer.Option(True, "--skip-missing/--no-skip-missing", help="Skip rows whose path is missing."),
    recurse: bool = typer.Option(True, "--recurse/--no-recurse", help="(Root mode) Recurse into subdirectories."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be parsed; do not write."),
):
    """Parse ORCA results for selected jobs and collate into a single JSONL.

    Two input modes:
      1) CSV mode: --csv provided (scan→parse workflow)
      2) Root mode: --root provided (sentinel-driven workflow)
    """
    out_jsonl = out_jsonl.expanduser().resolve()

    # Exactly one input mode
    if (csv is None) == (root is None):
        raise typer.BadParameter("Provide exactly one of --csv or --root")

    def _is_job_dir(d: Path) -> bool:
        # Heuristic: treat directories with plan_entry.json OR any *.inp OR any *.out as job dirs.
        if not d.is_dir():
            return False
        if (d / "plan_entry.json").is_file():
            return True
        if any(d.glob("*.inp")):
            return True
        if any(d.glob("*.out")):
            return True
        return False

    def _has_sentinel(d: Path, name: str) -> bool:
        return (d / name).exists()

    # Resolve paths to parse
    paths: List[Path] = []

    if csv is not None:
        csv = csv.expanduser().resolve()
        if not csv.is_file():
            raise typer.BadParameter(f"CSV not found: {csv}")

        df = _pd.read_csv(csv)

        if "path" not in df.columns:
            raise typer.BadParameter("CSV missing required column: path")
        if "status" not in df.columns:
            raise typer.BadParameter("CSV missing required column: status")

        want = status.strip().upper()
        if want not in ("OK", "FAIL", "ALL"):
            raise typer.BadParameter("--status must be one of: OK|FAIL|ALL")

        if want != "ALL":
            df = df[df["status"].astype(str).str.upper() == want]

        paths = [Path(p).expanduser() for p in df["path"].astype(str).tolist()]

    else:
        root = root.expanduser().resolve()
        if not root.exists():
            raise typer.BadParameter(f"Root not found: {root}")

        only = (only_sentinel or "ANY").strip().upper()
        allowed = {"GOOD", "BAD", "DONE", "FAIL", "READY", "STARTED", "ANY"}
        if only not in allowed:
            raise typer.BadParameter(f"--only-sentinel must be one of: {', '.join(sorted(allowed))}")

        # root itself may be a jobdir
        if root.is_dir() and _is_job_dir(root):
            candidates = [root]
        else:
            it = root.rglob("*") if recurse else root.glob("*")
            candidates = [p for p in it if p.is_dir() and _is_job_dir(p)]

        if only == "ANY":
            paths = candidates
        else:
            paths = [p for p in candidates if _has_sentinel(p, only)]

        paths = sorted(paths, key=lambda p: str(p))

    if limit is not None:
        paths = paths[: int(limit)]

    if dry_run:
        for p in paths:
            typer.echo(str(p))
        typer.secho(f"Would parse {len(paths)} job(s).", fg=typer.colors.GREEN)
        return

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in paths:
            if not p.is_dir():
                if skip_missing:
                    continue
                raise typer.BadParameter(f"Job directory not found: {p}")

            rec = collect_job_record(p)
            f.write(json.dumps(rec) + "\n")
            n_written += 1

    typer.secho(f"Wrote {n_written} record(s) → {out_jsonl}", fg=typer.colors.GREEN)

