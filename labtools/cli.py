# src/labtools/cli.py
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as _pd
import typer

# Descriptor functionality
from labtools.chem import energetic_span as es_mod
from labtools.chem.descriptors import homo_lumo_gap
# Utility stuff
from labtools.data.io import jsonl_append, jsonl_to_parquet
from labtools.orca.parse import parse_orca_file, collect_job_record
# Provenance snapshot
from labtools.prov import snapshot as snap_mod
# Render inputs yaml -> j2 -> input
from labtools.slurm.render import render_template
from labtools.slurm.render import render_plan_jobs
from labtools.slurm.render import render_plan_entrypoint as render_plan
# SLURM submission
from labtools.submit import dispatch  # unified submitter (job|array|drone)
# TSGen Pipeline
from labtools.tsgen.pipeline import TSPipeline, run_tsgen  # keep legacy import working

# --- TS Fingerprint (tsfp) light-weight wiring (aligned with your actual modules) ---
try:
    from labtools.tsgen.tsfp import (
        write_fingerprint_yaml,
        verify_against_fingerprint,
        VerifyResult,
    )
    _tsfp_import_error: Exception | None = None
except Exception as e:
    write_fingerprint_yaml = None  # type: ignore
    verify_against_fingerprint = None  # type: ignore
    VerifyResult = None  # type: ignore
    _tsfp_import_error = e

app = typer.Typer(help="lab-tools CLI")

# ts-fp command group
tsfp_app = typer.Typer(help="Transition-state fingerprint tools (ts-fp).")
app.add_typer(tsfp_app, name="ts-fp")

# ---------------------------------------------------------------------------
# Small helpers (tabular export, provenance, energetic span)
# ---------------------------------------------------------------------------

def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return cur if cur is not None else default


def _flatten_record(rec: dict) -> dict:
    p = rec.get("parsed", {})
    out = {
        # identity
        "job_dir": rec.get("dir"),
        "job_name": rec.get("job_name"),
        "primary_out": rec.get("primary_out"),
        "primary_out_size": rec.get("primary_out_size"),

        # meta
        "job_type": _get(p, "meta.job_type_guess"),
        "orca_version": _get(p, "meta.orca_version"),
        "charge": _get(p, "meta.charge"),
        "multiplicity": _get(p, "meta.multiplicity"),
        "PAL": _get(p, "meta.pal_threads"),
        "OMP": _get(p, "meta.omp_threads"),

        # method
        "method_kind": _get(p, "method.method"),
        "xc_exchange": _get(p, "method.xc_exchange"),
        "xc_correlation": _get(p, "method.xc_correlation"),
        "hybrid_fraction": _get(p, "method.hybrid_fraction"),
        "ri_coulomb": _get(p, "method.ri_coulomb"),
        "rij_cosx": _get(p, "method.rij_cosx"),
        "grid": _get(p, "method.grid"),
        "basis_main": _get(p, "method.basis_main"),
        "basis_aux_j": _get(p, "method.basis_aux_j"),
        "basis_aux_c": _get(p, "method.basis_aux_c"),

        # SCF
        "scf_converged": _get(p, "scf.converged"),
        "scf_iters": _get(p, "scf.iterations"),
        "scf_cpu_time": _get(p, "scf.cpu_time"),
        "energy_Eh": _get(p, "scf.energy_final_au") or _get(p, "final_single_point_energy_au"),

        # SCF tolerances (useful for audits)
        "TolE_Eh": _get(p, "scf.tol_energy_Eh"),
        "TolMAXG_Eh_per_bohr": _get(p, "scf.tol_max_grad_Eh_per_bohr"),
        "TolRMSG_Eh_per_bohr": _get(p, "scf.tol_rms_grad_Eh_per_bohr"),
        "TolMAXD_bohr": _get(p, "scf.tol_max_disp_bohr"),
        "TolRMSD_bohr": _get(p, "scf.tol_rms_disp_bohr"),
        "StrictConv": _get(p, "scf.strict_convergence"),

        # OPT/FREQ
        "opt_status": _get(p, "opt.status"),
        "opt_steps": _get(p, "opt.steps"),
        "n_imag": _get(p, "freq.n_imag"),
        "lowest_cm1": _get(p, "freq.lowest_cm1"),
        "ZPE_Eh": _get(p, "freq.zpe_au") or _get(p, "thermo.zpe_au"),

        # Thermochemistry (Eh)
        "E_elec_Eh": _get(p, "thermo.electronic_energy_au"),
        "therm_vib_Eh": _get(p, "thermo.thermal_vib_corr_au"),
        "therm_rot_Eh": _get(p, "thermo.thermal_rot_corr_au"),
        "therm_trans_Eh": _get(p, "thermo.thermal_trans_corr_au"),
        "E_thermal_total_Eh": _get(p, "thermo.total_thermal_energy_au"),
        "thermal_correction_total_Eh": _get(p, "thermo.total_thermal_correction_au"),
        "zpe_correction_Eh": _get(p, "thermo.non_thermal_zpe_correction_au"),
        "correction_total_Eh": _get(p, "thermo.total_correction_au"),
        "H_correction_Eh": _get(p, "thermo.thermal_enthalpy_correction_au"),
        "H_total_Eh": _get(p, "thermo.total_enthalpy_au"),
        "TS_Eh": _get(p, "thermo.final_entropy_term_au") or (
            -_get(p, "thermo.total_entropy_correction_au") if _get(p, "thermo.total_entropy_correction_au") is not None else None
        ),
        "G_total_Eh": _get(p, "thermo.gibbs_free_energy_au"),
        "G_minus_Eel_Eh": _get(p, "thermo.g_minus_electronic_au"),
        "Temp_K": _get(p, "thermo.temperature_K"),

        # Orbitals (Alpha)
        "HOMO_Eh": _get(p, "orbitals_alpha.homo_Eh"),
        "LUMO_Eh": _get(p, "orbitals_alpha.lumo_Eh"),
        "gap_eV": _get(p, "orbitals_alpha.gap_eV"),
    }

    Eh_to_eV = 27.211386245988
    Eh_to_kcal = 627.509474
    if out["energy_Eh"] is not None:
        out["energy_eV"] = out["energy_Eh"] * Eh_to_eV
        out["energy_kcal_mol"] = out["energy_Eh"] * Eh_to_kcal
    if out["ZPE_Eh"] is not None:
        out["ZPE_kcal_mol"] = out["ZPE_Eh"] * Eh_to_kcal
    if out["TS_Eh"] is not None and out["Temp_K"] is not None:
        out["S_kcal_mol_K"] = (out["TS_Eh"] * Eh_to_kcal) / out["Temp_K"]

    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    return rows


def _resolve_capture_snapshot_fn():
    for name in ("capture_snapshot", "make_snapshot", "get_snapshot", "snapshot", "build_provenance"):
        fn = getattr(snap_mod, name, None)
        if callable(fn):
            return fn

    import hashlib, platform

    def _fallback():
        def _git_sha():
            try:
                return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                return None

        env_keys = ["SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_CPUS_ON_NODE", "EBROOTORCA"]
        env = {k: os.environ.get(k) for k in env_keys if k in os.environ}
        rec = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "python": platform.python_version(),
            "git_sha": _git_sha(),
            "env": env,
        }
        rec["env_hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
        return rec

    return _fallback


CAPTURE_SNAPSHOT_FN = _resolve_capture_snapshot_fn()


def _resolve_energetic_span_fn():
    for name in ("energetic_span", "compute_energetic_span", "compute_span", "calc_energetic_span"):
        fn = getattr(es_mod, name, None)
        if callable(fn):
            return fn

    def _fallback(states):
        ts_list = [s for s in states if str(s.get("kind", "")).upper() == "TS"]
        i_list = [s for s in states if str(s.get("kind", "")).upper() == "I"]
        if not ts_list or not i_list:
            return {"delta_E": None, "TDTS": None, "TDI": None, "pair": None, "error": "Need at least one TS and one I"}
        best = None
        for ts in ts_list:
            for i in i_list:
                delta = ts["G"] - i["G"]
                if (best is None) or (delta > best[0]):
                    best = (delta, ts, i)
        delta_E, ts, i = best
        return {"delta_E": float(delta_E), "TDTS": ts["label"], "TDI": i["label"], "pair": (ts["label"], i["label"])}

    return _fallback


ENERGETIC_SPAN_FN = _resolve_energetic_span_fn()

# ---------------------------------------------------------------------------
# Repo paths & tsfp storage
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # .../src/labtools/cli.py → repo root is parents[2]
    return Path(__file__).resolve().parents[2]

def _repo_tsfp_dir() -> Path:
    return _repo_root() / "resources" / "tsfp"

def _resolve_tsfp_ref(ref: str) -> Path:
    """
    Resolve a fingerprint reference within <repo>/resources/tsfp or from a direct path.
    Accepts:
      - absolute/relative path (to repo root),
      - an ID from index.csv,
      - a bare stem; searches for *.yaml/yml/json under resources/tsfp.
    """
    base = _repo_tsfp_dir()

    p = Path(ref)
    if not p.is_absolute():
        p = (_repo_root() / p)
    if p.exists():
        return p.resolve()

    idx = base / "index.csv"
    if idx.exists():
        try:
            with open(idx, "r", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("id", "")).strip() == ref:
                        target = (base / row["path"]).resolve()
                        if target.exists():
                            return target
        except Exception:
            pass

    patterns = [f"{ref}.yaml", f"{ref}.yml", f"{ref}.json",
                f"*{ref}*.yaml", f"*{ref}*.yml", f"*{ref}*.json"]
    for pat in patterns:
        for cand in base.rglob(pat):
            return cand.resolve()

    raise typer.BadParameter(
        f"Fingerprint '{ref}' not found. "
        f"Put YAML/JSON under {base} or list it in {idx} with columns id,path."
    )

# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

@app.command()
def gap(
    homo: float = typer.Option(..., "--homo", "-h", help="HOMO energy (eV or Eh)"),
    lumo: float = typer.Option(..., "--lumo", "-l", help="LUMO energy (eV or Eh)"),
):
    """Compute HOMO–LUMO gap in eV (auto-detect units)."""
    g = homo_lumo_gap(homo, lumo)
    typer.echo(f"gap_eV={g:.6f}")


@app.command("orca-parse")
def orca_parse(
    path: Path = typer.Argument(..., help="Path to ORCA .out"),
    out: Optional[Path] = typer.Option(None, "--out", help="Append JSON record to this JSONL"),
):
    """Parse an ORCA output file to a JSON record."""
    rec = parse_orca_file(path)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        jsonl_append(out, rec)
        typer.secho(f"Appended record to {out}", fg=typer.colors.GREEN)
    else:
        typer.echo(json.dumps(rec, indent=2))


@app.command("results-parquet")
def jsonl2parquet_cmd(jsonl: Path, parquet: Path):
    """Convert JSONL → Parquet."""
    parquet.parent.mkdir(parents=True, exist_ok=True)
    jsonl_to_parquet(jsonl, parquet)
    typer.secho(f"Wrote {parquet}", fg=typer.colors.GREEN)


@app.command("results-csv")
def results_csv(
    jsonl: Path = typer.Argument(..., help="Input JSONL from parse-all"),
    out_csv: Path = typer.Option(Path("results.csv"), "--out", "-o", help="Output CSV"),
):
    """Flatten JSONL results into a human-friendly CSV."""
    jsonl = jsonl.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    rows = _read_jsonl(jsonl)
    flat = [_flatten_record(r) for r in rows]
    df = _pd.DataFrame(flat)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    typer.secho(f"Wrote {out_csv} ({len(df)} rows)", fg=typer.colors.GREEN)


@app.command("prov-snapshot")
def prov_snapshot(out: Optional[Path] = typer.Option(None, "--out", help="Write JSON to this path")):
    """Capture a lightweight provenance snapshot (env hash, module list, git SHA)."""
    prov = CAPTURE_SNAPSHOT_FN()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prov, indent=2))
        typer.secho(f"Wrote {out}", fg=typer.colors.GREEN)
    else:
        typer.echo(json.dumps(prov, indent=2))


@app.command("es")
def energetic_span_cmd(states_json: Path):
    """Compute energetic span (kcal/mol) from JSON list of states."""
    states = json.loads(Path(states_json).read_text())
    res = ENERGETIC_SPAN_FN(states)
    typer.echo(json.dumps(res, indent=2))

# ---------------------------------------------------------------------------
# Luxury TS pipeline (keeps the name `tsgen`)  [unchanged behavior]
# ---------------------------------------------------------------------------

@app.command("tsgen")
def tsgen_lux_cmd(
    reactant: Path = typer.Option(..., "--reactant", help="Reactant XYZ"),
    product: Path  = typer.Option(..., "--product", help="Product XYZ"),
    outdir: Path   = typer.Option(..., "--outdir", help="Output directory"),
    charge: int    = typer.Option(0, "--charge"),
    mult: int      = typer.Option(1, "--mult"),
    fingerprint: Optional[str] = typer.Option(None, "--fingerprint", help="TS fingerprint ID or path"),

    # method ladder (defaults agreed)
    L0: str = typer.Option("XTB2", "--L0"),
    L1: str = typer.Option("r2SCAN-3c", "--L1"),
    L2: str = typer.Option("M06/def2-SVP", "--L2"),
    L3: Optional[str] = typer.Option("DLPNO-CCSD(T)/def2-TZVPP", "--L3"),

    # unified submission knobs
    mode: str = typer.Option("array", "--mode", help="job|array|drone"),
    profile: str = typer.Option("medium", "--profile", help="short|medium|long"),
):
    """
    Luxury TS pipeline (L0→L1→L2→L3) using unified submission templates.
    """
    # Resolve fingerprint reference if provided (ID→resources/tsfp or direct path)
    fp_path: Optional[Path] = None
    if fingerprint:
        try:
            fp_path = _resolve_tsfp_ref(fingerprint)
        except Exception:
            # also allow direct filesystem path by user intent
            p = Path(fingerprint)
            if p.exists():
                fp_path = p.resolve()
            else:
                raise

    methods = {"L0": L0, "L1": L1, "L2": L2}
    if L3:
        methods["L3"] = L3

    pipe = TSPipeline(
        reactant=reactant, product=product,
        charge=charge, mult=mult,
        outdir=outdir,
        methods=methods,
        fingerprint=fp_path,
        mode=mode, profile=profile,
    )
    pipe.run()

# ---------------------------------------------------------------------------
# Template rendering + job prep  [unchanged]
# ---------------------------------------------------------------------------

JOB_TEMPLATES: Dict[str, str] = {
    "optfreq": "orca_optfreq.inp.j2",
    "sp-triplet": "orca_sp_triplet.inp.j2",
    "nmr": "orca_nmr.inp.j2",
}


def _coerce_value(v: str) -> Any:
    try:
        return json.loads(v)
    except Exception:
        try:
            return float(v) if "." in v else int(v)
        except Exception:
            return v


def _load_geometry_block(geom_file: Optional[Path], geom_literal: Optional[str]) -> Optional[str]:
    if geom_file:
        return Path(geom_file).read_text().strip() + "\n"
    if geom_literal:
        return geom_literal.strip() + "\n"
    return None


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


def _repo_templates_root() -> Path:
    return Path(__file__).resolve().parents[2] / "templates"


def _render_sbatch_with_fallback(dst: Path, params: Dict[str, Any]) -> None:
    root = _repo_templates_root()
    candidates = [
        root / "sbatch" / "single_orca_simple.sbatch.j2",
        root / "sbatch" / "single_orca_job.sbatch.j2",
        root / "orca" / "sbatch_orca.sbatch.j2",
        root / "sbatch_orca.sbatch.j2",
    ]
    last_err: Optional[Exception] = None
    for tpl in candidates:
        try:
            render_template(str(tpl), dst, params)
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not render any sbatch template. Tried: {', '.join(map(str, candidates))}. Last error: {last_err}"
    )


@app.command()
def render(
    template: Path = typer.Argument(..., help="Template file name (e.g., orca_optfreq.inp.j2) or absolute path"),
    out: Path = typer.Argument(..., help="Where to write the rendered file"),
    set: List[str] = typer.Option(None, "--set", "-s", help="key=value (repeatable)"),
    geometry_file: Optional[Path] = typer.Option(None, "--geometry-file", "-g", help="Inline coordinates from file"),
    geometry_literal: Optional[str] = typer.Option(None, "--geometry-literal", help="Inline coordinates text; use \\n"),
):
    """Render a Jinja2 template to a file."""
    params: Dict[str, Any] = {}
    if set:
        for kv in set:
            if "=" not in kv:
                raise typer.BadParameter(f"--set expects key=value, got: {kv}")
            k, v = kv.split("=", 1)
            params[k.strip()] = _coerce_value(v.strip())
    gb = _load_geometry_block(geometry_file, geometry_literal)
    if gb is not None:
        params["geometry_block"] = gb
    out.parent.mkdir(parents=True, exist_ok=True)
    render_template(template, out, params)
    typer.secho(f"Wrote {out}", fg=typer.colors.GREEN)


@app.command("render-plan")
def render_plan_cmd(
    plan: Path = typer.Option(..., "--plan", exists=True, readable=True, help="Path to plans/plan.yaml"),
    out: Optional[Path] = typer.Option(None, "--out", "--output", "--outdir", help="Directory for rendered ORCA inputs"),
    only: List[str] = typer.Option(None, "--only", help="Render only these job IDs (repeat or comma-separated)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs"),
):
    out = out or Path("build/inputs")

    wanted: Optional[List[str]] = None
    if only:
        expanded: List[str] = []
        for item in only:
            expanded.extend([p for p in str(item).split(",") if p])
        wanted = expanded

    written = render_plan_jobs(plan, out, only=wanted, overwrite=overwrite, verbose=False)
    typer.echo(f"Rendered {len(written)} input(s) to {out}")


@app.command()
def prep(
    job_type: str = typer.Argument(..., help="Job type: optfreq | sp-triplet | nmr"),
    name: str = typer.Argument(..., help="Base name (e.g., H2, Ru_NHC)"),
    charge: int = typer.Option(..., "--charge"),
    multiplicity: int = typer.Option(..., "--multiplicity"),
    geometry_file: Optional[Path] = typer.Option(None, "--geometry-file", "-g"),
    geometry_literal: Optional[str] = typer.Option(None, "--geometry-literal"),
    set: List[str] = typer.Option(None, "--set", "-s", help="Extra params key=value (repeatable)"),
    time_str: str = typer.Option("24:00:00", "--time", "-t", help="SLURM time limit (HH:MM:SS)"),
    cpus_per_task: int = typer.Option(8, "--cpus", help="CPUs per task"),
    mem: str = typer.Option("16G", "--mem", help="Memory, e.g. 16G"),
):
    """Create NAME_JOBTYPE/ with <job_type>.inp and job.sbatch (node-local scratch run)."""
    jt = job_type.strip().lower()
    if jt not in JOB_TEMPLATES:
        raise typer.BadParameter(f"Unknown job_type '{jt}'. Choose one of {list(JOB_TEMPLATES)}")
    tpl_inp = JOB_TEMPLATES[jt]
    jobdir = Path(f"{_sanitize_name(name)}_{jt}")
    jobdir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {"charge": charge, "multiplicity": multiplicity}
    params["pal"] = cpus_per_task
    if set:
        for kv in set:
            if "=" not in kv:
                raise typer.BadParameter(f"--set expects key=value, got: {kv}")
            k, v = kv.split("=", 1)
            params[k.strip()] = _coerce_value(v.strip())

    gb = _load_geometry_block(geometry_file, geometry_literal)
    if not gb:
        raise typer.BadParameter("Geometry required. Provide --geometry-file or --geometry-literal.")
    params["geometry_block"] = gb

    inp_path = jobdir / f"{jt}.inp"
    render_template(tpl_inp, inp_path, params)

    inp_basename = inp_path.name
    out_basename = f"{jt}.out"
    orca_cmd = f'${{EBROOTORCA}}/orca "{inp_basename}" > "{out_basename}"'
    _render_sbatch_with_fallback(
        jobdir / "job.sbatch",
        {
            "job_name": jobdir.name,
            "time": time_str,
            "cpus_per_task": cpus_per_task,
            "mem": mem,
            "inp_basename": inp_basename,
            "out_basename": out_basename,
            "orca_cmd": orca_cmd,
        },
    )

    typer.secho(f"Prepared {jobdir}/", fg=typer.colors.GREEN)

# ---------------------------------------------------------------------------
# Submission commands (unchanged)
# ---------------------------------------------------------------------------

@app.command("submit-job")
def submit_job(
    inp: Path = typer.Argument(..., help="Path to a single ORCA .inp"),
    profile: str = typer.Option("medium", "--profile", help="short|medium|long"),
    job_name: Optional[str] = typer.Option(None, "--job-name"),
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Directory to run sbatch from"),
    job_chdir: Optional[Path] = typer.Option(None, "--job-chdir", help="SLURM working directory for the job"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Run sbatch --test-only (do not submit)"),
):
    inp = inp.expanduser().resolve()
    if not inp.exists():
        raise typer.BadParameter(f"Input not found: {inp}")
    name = job_name or inp.parent.name
    job_chdir = job_chdir or inp.parent
    dispatch(
        inp, mode="job", profile=profile, job_name=name,
        submit_cwd=cwd, sbatch_chdir=job_chdir, validate_only=validate_only
    )
    typer.secho(
        f"Submitted single job for {inp} (cwd={cwd or os.getcwd()}, job-chdir={job_chdir}, validate_only={validate_only})",
        fg=typer.colors.GREEN
    )


@app.command("submit-array")
def submit_array(
    inps: List[Path] = typer.Argument(None, help="One or more ORCA .inp files"),
    glob: Optional[str] = typer.Option(None, "--glob", help="Glob for .inp files (e.g., '*/opt/*.inp')"),
    profile: str = typer.Option("medium", "--profile", help="short|medium|long"),
    job_name: Optional[str] = typer.Option("array", "--job-name"),
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Directory to run sbatch from"),
    job_chdir: Optional[Path] = typer.Option(None, "--job-chdir", help="SLURM working directory for the array job"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Run sbatch --test-only (do not submit)"),
):
    paths: List[Path] = []
    if glob:
        paths.extend([p for p in Path(".").glob(glob)])
    if inps:
        paths.extend(inps)
    paths = [p.expanduser().resolve() for p in paths if p]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise typer.BadParameter("No inputs provided. Use positional .inp files and/or --glob.")

    if job_chdir is None:
        parents = {str(p.parent) for p in paths}
        job_chdir = Path(os.path.commonpath(list(parents))) if parents else Path.cwd()

    dispatch(
        paths, mode="array", profile=profile, job_name=job_name or "array",
        submit_cwd=cwd, sbatch_chdir=job_chdir, validate_only=validate_only
    )
    typer.secho(
        f"Submitted array ({len(paths)} inputs) (cwd={cwd or os.getcwd()}, job-chdir={job_chdir}, validate_only={validate_only})",
        fg=typer.colors.GREEN
    )


@app.command("submit-drone")
def submit_drone(
    n: int = typer.Option(1, "--n", help="Number of worker jobs to submit"),
    profile: str = typer.Option("long", "--profile", help="test|short|medium|long"),
    job_name: Optional[str] = typer.Option("drone", "--job-name"),
    queue_dir: str = typer.Option("forge_queue", "--queue-dir", help="Watched directory for .inp files"),
    sleep_secs: int = typer.Option(60, "--sleep-secs", help="Poll interval inside worker"),
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Directory to run sbatch from"),
    job_chdir: Optional[Path] = typer.Option(None, "--job-chdir", help="SLURM working directory inside the worker"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Run sbatch --test-only (do not submit)"),
):
    extra = {"QUEUE_DIR": queue_dir, "SLEEP_SECS": sleep_secs}
    for i in range(max(1, n)):
        name = f"{job_name}-{i+1}" if n > 1 else job_name
        dispatch(
            Path("."), mode="drone", profile=profile, job_name=name,
            extra_params=extra, submit_cwd=cwd, sbatch_chdir=(job_chdir or Path.cwd()),
            validate_only=validate_only
        )
    typer.secho(
        f"Submitted {n} drone worker(s) (cwd={cwd or os.getcwd()}, job-chdir={job_chdir or Path.cwd()}, validate_only={validate_only})",
        fg=typer.colors.GREEN
    )

# ---------------------------------------------------------------------------
# Legacy aliases (kept so older scripts don't break)
# ---------------------------------------------------------------------------

@app.command("sbatch-submit")
def sbatch_submit_alias(
    job: Path = typer.Argument(..., help="Job DIR or job.sbatch file"),
    profile: str = typer.Option("medium", "--profile", help="short|medium|long"),
):
    job = job.expanduser().resolve()
    if job.is_dir():
        inp = next((p for p in job.glob("*.inp")), None)
        if not inp:
            raise typer.BadParameter(f"No .inp found in {job}")
        return submit_job(inp, profile)
    else:
        if job.suffix == ".sbatch":
            guess = next((p for p in job.parent.glob("*.inp")), None)
            if not guess:
                raise typer.BadParameter(f"No .inp found next to {job}")
            return submit_job(guess, profile)
        return submit_job(job, profile)


@app.command("sbatch-wait-parse")
def sbatch_wait_parse(
    jobdir: Path = typer.Argument(..., help="Folder created by `prep`"),
    poll: float = typer.Option(20.0, help="Poll interval (seconds)"),
    timeout: int = typer.Option(0, help="Timeout seconds (0 = no timeout)"),
    out_jsonl: Optional[Path] = typer.Option(None, "--out-jsonl", help="Append parsed record here"),
):
    jobdir = Path(jobdir)
    inp = next((p for p in jobdir.glob("*.inp")), None)
    if not inp:
        raise typer.BadParameter(f"No .inp in {jobdir}")
    out = jobdir / (inp.stem + ".out")

    typer.echo(f"Waiting for ORCA output: {out} ...")
    start = time.time()
    while True:
        if out.exists() and out.stat().st_size > 0:
            break
        if timeout and (time.time() - start) > timeout:
            raise typer.Exit(code=1)
        time.sleep(poll)

    record = parse_orca_file(out)
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_append(out_jsonl, record)
        typer.secho(f"Parsed {out.name} → {out_jsonl}", fg=typer.colors.GREEN)
    else:
        typer.echo(json.dumps(record, indent=2))


@app.command("tsgen-workers")
def tsgen_workers_alias(
    outdir: Path = typer.Option(..., "--outdir"),
    n: int = typer.Option(4, "--n"),
    pal: int = typer.Option(8, "--pal"),
    maxcore: int = typer.Option(3000, "--maxcore"),
    time: str = typer.Option("24:00:00", "--time"),
    mem: str = typer.Option(24, "--mem"),
    partition: str = typer.Option(None, "--partition"),
    account: str = typer.Option(None, "--account"),
    qos: str = typer.Option(None, "--qos"),
):
    """[DEPRECATED] Use: submit-drone"""
    return submit_drone(n=n)

# -----------------------------
# Jobs subcommands (Typer group)
# -----------------------------
jobs_app = typer.Typer(help="Job plan utilities (CSV→YAML generator, etc.)")
app.add_typer(jobs_app, name="jobs")

def _read_yaml_file(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _sha256(path: Path):
    import hashlib
    if not path or not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@jobs_app.command("from-csv")
def jobs_from_csv(
    csv_path: Path = typer.Option(..., "--csv", exists=True, readable=True, help="Input CSV path"),
    mapping: Path = typer.Option(..., "--mapping", exists=True, readable=True, help="Mapping YAML (column→path, types, fanout)"),
    combine: Optional[Path] = typer.Option(None, "--combine", help="Write a combined plan YAML (version + jobs[])"),
    out: Optional[Path] = typer.Option(None, "--out", help="Also write per-job debug YAMLs here"),
    defaults: Optional[Path] = typer.Option(None, "--defaults", exists=True, readable=True, help="Defaults YAML to pre-seed documents"),
    plan_schema: Path = typer.Option(Path("labtools/schemas/plan.schema.json"), "--plan-schema", help="JSON Schema for plan validation"),
    fanout_strategy: str = typer.Option("product", "--fanout-strategy", help="Reserved: product|zip"),
    id_pattern: Optional[Path] = typer.Option(None, "--id-pattern", help="Jinja pattern for per-job filenames"),
    validate: bool = typer.Option(False, "--validate", help="Validate the combined plan against --plan-schema"),
    provenance: bool = typer.Option(False, "--provenance", help="Write per-file provenance sidecars for per-job YAMLs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse & validate only; write nothing"),
):
    try:
        from labtools.csvpipe.mapping import build_mapping
        from labtools.csvpipe.loader import row_to_job, read_csv_rows
        from labtools.csvpipe.emit import emit_job_entries, emit_combined_plan
        from labtools.csvpipe.validate import validate_doc
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise typer.BadParameter(f"csvpipe modules are not available: {e!r}")

    if not dry_run and (combine is None and out is None):
        raise typer.BadParameter("Specify at least one output (--combine and/or --out), or use --dry-run.")

    import yaml
    with open(mapping, "r", encoding="utf-8") as f:
        mapping_spec = build_mapping(yaml.safe_load(f))
    defaults_doc = {}
    if defaults:
        with open(defaults, "r", encoding="utf-8") as f:
            defaults_doc = yaml.safe_load(f) or {}

    rows = read_csv_rows(csv_path)
    jobs = [row_to_job(r, mapping_spec, defaults_doc) for r in rows]

    if validate:
        combined_in_mem = {
            "version": "1",
            "jobs": [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs],
        }
        validate_doc(combined_in_mem, str(plan_schema))

    if dry_run:
        typer.echo(f"[DRY RUN] Parsed {len(rows)} CSV row(s) → {len(jobs)} job(s). No files written.")
        raise typer.Exit(0)

    inputs_meta = {
        "csv": str(csv_path),
        "mapping": str(mapping),
        "defaults": str(defaults) if defaults else None,
        "fanout_strategy": fanout_strategy,
    }

    if out is not None:
        emit_job_entries(
            jobs,
            out,
            id_pattern=id_pattern,
            provenance=provenance,
            inputs_meta=inputs_meta,
        )
        typer.echo(f"Wrote per-job YAMLs to {out}")

    if combine is not None:
        emit_combined_plan(jobs, combine, version="1")
        typer.echo(f"Wrote combined plan to {combine}")

# ---------------------------------------------------------------------------
# ts-fp subcommands
# ---------------------------------------------------------------------------

@tsfp_app.command("make")
def tsfp_make(
    out_file: Path = typer.Option(None, "--out-file", help="Where to write the fingerprint (YAML). "
                              "If omitted, writes to <repo>/resources/tsfp/<name>.yaml"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Fingerprint ID (used when --out-file not given)"),
    atoms: str = typer.Option(..., "--atoms", help="Comma-separated atom indices, e.g. 0,1,9,12"),
    src_out: Path = typer.Argument(..., help="ORCA 6.1 .out file to fingerprint"),
):
    """
    Create a fingerprint YAML (geometry + Mode-6 selected vectors).
    """
    if _tsfp_import_error:
        raise typer.BadParameter(f"ts-fp modules unavailable: {_tsfp_import_error!r}")

    idx = [int(x) for x in re.split(r"[,\s]+", atoms.strip()) if x != ""]

    # Decide output path
    if out_file is None:
        if not name:
            raise typer.BadParameter("Provide --name when --out-file is omitted.")
        dst = (_repo_tsfp_dir() / f"{name}.yaml").resolve()
    else:
        dst = out_file.resolve()

    dst.parent.mkdir(parents=True, exist_ok=True)
    path = write_fingerprint_yaml(Path(src_out), idx, dst)
    typer.secho(f"Wrote fingerprint → {path}", fg=typer.colors.GREEN)


@tsfp_app.command("verify")
def tsfp_verify(
    reference: str = typer.Argument(..., help="Fingerprint ID or path (YAML under resources/tsfp)"),
    test_out: Path = typer.Argument(..., help="ORCA .out to verify"),
    atoms: Optional[str] = typer.Option(None, "--atoms", help="Override atom indices; default = use reference YAML"),
    tol: float = typer.Option(0.85, "--tol", help="Cosine similarity threshold for PASS"),
    require_mode6: bool = typer.Option(True, "--require-mode6/--no-require-mode6",
                                       help="Fail early if Mode-6 block is absent"),
):
    """
    Verify a TS by comparing its Mode-6 vectors (selected atoms) against a stored fingerprint.
    """
    if _tsfp_import_error:
        raise typer.BadParameter(f"ts-fp modules unavailable: {_tsfp_import_error!r}")

    ref_path = _resolve_tsfp_ref(reference)
    idx_override: Optional[List[int]] = None
    if atoms:
        idx_override = [int(x) for x in re.split(r"[,\s]+", atoms.strip()) if x != ""]

    res: VerifyResult = verify_against_fingerprint(
        ref_path=ref_path,
        out_path=Path(test_out),
        atoms_override=idx_override,
        tol=tol,
        require_mode6=require_mode6,
    )

    # Support either .passed attribute or property; also print score if present
    passed = getattr(res, "passed", None)
    if passed is None and hasattr(res, "ok"):
        passed = getattr(res, "ok")

    score = getattr(res, "score", None)
    msg = getattr(res, "message", None)

    status = "PASS" if passed else "FAIL"
    if score is not None and msg:
        typer.echo(f"{status}  score={score:.6f}  {msg}")
    elif score is not None:
        typer.echo(f"{status}  score={score:.6f}")
    elif msg:
        typer.echo(f"{status}  {msg}")
    else:
        typer.echo(status)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
