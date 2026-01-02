from __future__ import annotations

import datetime
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
from labtools.slurm.render import render_plan_jobs
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

app = typer.Typer(help="FORGE / lab-tools CLI")

tsfp_app = typer.Typer(help="Transition-state fingerprint tools")
app.add_typer(tsfp_app, name="ts-fp")

tsgen2_app = typer.Typer(help="TSGen 2.0 transition-state workflow")
app.add_typer(tsgen2_app, name="tsgen2")

plan_app = typer.Typer(help="PlanEntry-based workflows")
app.add_typer(plan_app, name="plan")

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
        "nmr": "orca/orca_nmr.inp.j2",
        "sp-triplet": "orca/orca_sp_triplet.inp.j2",
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
        # -------------------------
        # Extract CSV row
        # -------------------------
        params = job.get("parameters") if isinstance(job.get("parameters"), dict) else {}

        row = {}
        if isinstance(job.get("_row_raw"), dict):
            row = job["_row_raw"]
        elif isinstance(params.get("_row_raw"), dict):
            row = params["_row_raw"]

        # -------------------------
        # Resolve task
        # -------------------------
        task = (
            job.get("job_type")
            or row.get("job_type")
            or params.get("job_type")
        )
        if not task:
            raise typer.BadParameter("CSV row missing job_type")
        task = task.strip().lower()

        if task not in PLAN_TASK_TEMPLATES:
            raise typer.BadParameter(f"Unknown task '{task}'")

        # -------------------------
        # Required chemistry
        # -------------------------
        structure = (
            row.get("structure")
            or params.get("structure")
            or job.get("system")
        )
        if not structure:
            raise typer.BadParameter("CSV row missing structure")

        structure = Path(structure)
        if not structure.is_file():
            raise typer.BadParameter(f"Structure file not found: {structure}")

        charge = int(row.get("charge", 0))
        mult = int(row.get("multiplicity", 1))

        method = row.get("method")
        if not method:
            raise typer.BadParameter("CSV row missing method")

        # -------------------------
        # Read geometry explicitly
        # -------------------------
        lines = structure.read_text(encoding="utf-8").strip().splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[2:]  # strip XYZ header
        geom_lines = lines

        # -------------------------
        # Optional controls
        # -------------------------
        pal = row.get("pal")
        pal = int(pal) if pal is not None else None

        maxcore = row.get("maxcore_mb")
        maxcore = int(maxcore) if maxcore is not None else None

        scf_opts = row.get("scf")
        if not isinstance(scf_opts, dict):
            scf_opts = {}

        cpcm_opts = row.get("cpcm")
        if not isinstance(cpcm_opts, dict):
            cpcm_opts = {}

        # -------------------------
        # Template context (ALWAYS COMPLETE)
        # -------------------------
        ctx = {
            "method": method,
            "charge": charge,
            "mult": mult,
            "geom_lines": geom_lines,
            "geom": {
                "lines": geom_lines,
                "constraints": [],
            },
            "scf": scf_opts,        # always dict
            "cpcm": cpcm_opts,      # always dict
        }

        if pal is not None:
            ctx["pal"] = pal

        if maxcore is not None:
            ctx["maxcore_mb"] = maxcore

        # -------------------------
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


        inp = job_dir / "job.inp"
        inp.write_text(text, encoding="utf-8")
    
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
    if out is None:
        out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten_record(v, prefix=key, out=out)
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


# ==========================================================
# ORCA Tools
# ==========================================================


@app.command("orca-info")
def cli_orca_info(path: Path):
    info = collect_job_record(path)
    typer.echo(json.dumps(info, indent=2))


@app.command("orca-parse")
def cli_orca_parse(
    path: Path,
    out_jsonl: Optional[Path] = None,
    wait: bool = False,
    poll: float = 30.0,
    timeout: Optional[int] = None,
):
    path = path.expanduser().resolve()

    if wait:
        start = time.time()
        while True:
            if path.is_file() and path.stat().st_size > 0:
                s1 = path.stat().st_size
                time.sleep(poll)
                s2 = path.stat().st_size
                if s1 == s2:
                    break
            if timeout and time.time() - start > timeout:
                raise typer.Exit(code=1)
            time.sleep(poll)

    record = collect_job_record(path)
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_append(out_jsonl, rec=record)
        typer.secho(f"Appended to {out_jsonl}", fg=typer.colors.GREEN)
    else:
        typer.echo(json.dumps(record, indent=2))


@app.command("orca-batch-parse")
def cli_orca_batch_parse(
    root: Path,
    pattern: str = "*.out",
    out_jsonl: Path = typer.Option(..., "--out-jsonl"),
    max_files: Optional[int] = None,
    recurse: bool = True,
    workers: int = 0,
):
    root = root.expanduser().resolve()
    out_jsonl = out_jsonl.expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    paths = (root.rglob(pattern) if recurse else root.glob(pattern))
    files = [p for p in paths if p.is_file()]
    if max_files:
        files = files[:max_files]

    def serial():
        for p in files:
            yield collect_job_record(p.parent)

    def parallel(nw: int):
        nw = max(2, min(nw, len(files)))
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = {ex.submit(collect_job_record, p.parent): p for p in files}
            for fut in as_completed(futures):
                yield fut.result()

    iterator = parallel(workers) if workers > 1 else serial()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in iterator:
            f.write(json.dumps(rec) + "\n")

    typer.secho(f"Wrote JSONL → {out_jsonl}", fg=typer.colors.GREEN)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
