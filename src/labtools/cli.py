from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from labtools.chem.descriptors import homo_lumo_gap
from labtools.chem import energetic_span as es_mod
from labtools.data.io import jsonl_append, jsonl_to_parquet
from labtools.orca.parse import parse_orca_file, collect_job_record
from labtools.prov import snapshot as snap_mod
from labtools.slurm.render import render_template

app = typer.Typer(help="lab-tools CLI")


# ---- tabular export helpers ----
from typing import Any
import pandas as _pd

# dotted-path getter
def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return cur if cur is not None else default

# pick + flatten one record
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

    # Nice derived columns
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




# -----------------------------
# Bulk parsing: parse-all
# -----------------------------
import hashlib
from datetime import datetime, timedelta

def _hash_file_quick(path: Path, limit: int = 1024 * 1024) -> str:
    """Hash up to `limit` bytes (1 MiB by default) for idempotency."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        chunk = f.read(limit)
        h.update(chunk)
    st = path.stat()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]

def _load_index(idx_path: Path) -> dict:
    if not idx_path.exists():
        return {}
    try:
        return json.loads(idx_path.read_text())
    except Exception:
        return {}

def _save_index(idx_path: Path, idx: dict) -> None:
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(idx, indent=2))
# -----------------------------
# Bulk parsing: parse-all (robust to SLURM logs; with fallback parser)
# -----------------------------
import hashlib
from datetime import datetime, timedelta
import re

def _hash_file_quick(path: Path, limit: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        chunk = f.read(limit)
        h.update(chunk)
    st = path.stat()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]

def _looks_like_orca_out(path: Path) -> bool:
    """
    Cheap content sniff: treat as ORCA if we see a strong marker within first ~200KB.
    Skips SLURM logs that don’t contain ORCA signatures.
    """
    try:
        with path.open("r", errors="ignore") as f:
            buf = f.read(200_000)
    except Exception:
        return False
    markers = (
        "TOTAL SCF ENERGY",           # common across versions
        "ORCA TERMINATED NORMALLY",   # end marker
        "--------------- ORBITAL ENERGIES ---------------",
        "O   R   C   A",              # banner sometimes present
    )
    return any(m in buf for m in markers)

_E_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"

def _fallback_orca_parse(path: Path) -> dict:
    """
    Minimal, robust ORCA parser for energy + HOMO/LUMO (Eh) and gap (eV).
    """
    total = None
    homo = None
    lumo = None
    in_occ = False
    in_vir = False
    with path.open("r", errors="ignore") as f:
        for line in f:
            if total is None:
                m = re.search(r"TOTAL SCF ENERGY\s+(" + _E_FLOAT + r")", line)
                if m:
                    total = float(m.group(1))
                    continue
            if "Alpha occ. orbital energies" in line:
                in_occ, in_vir = True, False
                continue
            if "Alpha virt. orbital energies" in line:
                in_occ, in_vir = False, True
                continue
            if in_occ:
                # lines may contain multiple numbers; take last valid number seen in occ block
                for m in re.finditer(_E_FLOAT, line):
                    homo = float(m.group(0))
            if in_vir and lumo is None:
                m = re.search(_E_FLOAT, line)
                if m:
                    lumo = float(m.group(0))
                    in_vir = False  # first virtual is LUMO
    rec = {"path": str(path)}
    if total is not None:
        rec["total_scf_energy_Eh"] = total
    if homo is not None:
        rec["HOMO_Eh"] = homo
    if lumo is not None:
        rec["LUMO_Eh"] = lumo
    if (homo is not None) and (lumo is not None):
        rec["gap_eV"] = (lumo - homo) * 27.211386
    return rec

def _load_index(idx_path: Path) -> dict:
    if not idx_path.exists():
        return {}
    try:
        return json.loads(idx_path.read_text())
    except Exception:
        return {}

def _save_index(idx_path: Path, idx: dict) -> None:
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(idx, indent=2))


@app.command("parse-all")
def parse_all(
    root: Path = typer.Option(Path("."), "--root", "-r", help="Root directory to scan"),
    recurse: bool = typer.Option(True, "--recurse/--no-recurse"),
    out_jsonl: Path = typer.Option(Path("results.jsonl"), "--out-jsonl"),
    max_dirs: int = typer.Option(0, "--max", help="Max directories to parse (0 = no limit)"),
    include_text_blobs: bool = typer.Option(False, "--include-text", help="Include head/tail clips"),
):
    root = root.expanduser().resolve()
    out_jsonl = out_jsonl.expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # walk directories
    dirs = []
    if recurse:
        for d, subdirs, files in os.walk(root):
            dirs.append(Path(d))
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]

    n = 0
    for d in sorted(dirs):
        rec = collect_job_record(d, include_text_blobs=include_text_blobs)
        if not rec:
            continue
        jsonl_append(out_jsonl, rec)
        n += 1
        typer.secho(f"Parsed {d} → {out_jsonl}", fg=typer.colors.GREEN)
        if max_dirs and n >= max_dirs:
            break

    typer.echo(f"Done. wrote {n} record(s) to {out_jsonl}")


def _resolve_capture_snapshot_fn():
    # common names we might have used
    for name in ("capture_snapshot", "make_snapshot", "get_snapshot", "snapshot", "build_provenance"):
        fn = getattr(snap_mod, name, None)
        if callable(fn):
            return fn

    # Fallback: minimal provenance snapshot (host, python, git sha, env slice, hash)
    import hashlib, json, os, platform, subprocess, time
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

# Resolve energetic span function from the module, or provide a fallback.
def _resolve_energetic_span_fn():
    # common names we might have used in that file
    for name in ("energetic_span", "compute_energetic_span", "compute_span", "calc_energetic_span"):
        fn = getattr(es_mod, name, None)
        if callable(fn):
            return fn

    # Fallback: minimal energetic span implementation
    def _fallback(states):
        """
        states: list of dicts with keys:
          - 'label' (str), 'kind' ('TS' or 'I'), 'G' (float, kcal/mol)
        Returns: dict with delta_E, TDTS, TDI, and (TS, I) pair.
        """
        ts_list = [s for s in states if str(s.get("kind","")).upper() == "TS"]
        i_list  = [s for s in states if str(s.get("kind","")).upper() == "I"]
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

# -----------------------------
# Basic utilities
# -----------------------------

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

# -----------------------------
# Template rendering + SLURM helpers
# -----------------------------

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

def _default_orca_cmd(inp: Path, out: Path) -> str:
    return f"orca {inp.name} > {out.name}"

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

    # Render input
    inp_path = jobdir / f"{jt}.inp"
    render_template(tpl_inp, inp_path, params)

    # Prepare sbatch with strict path template (templates/orca/sbatch_orca.sbatch.j2)
    sbatch_tpl = "sbatch_orca.sbatch.j2"
    # Basenames only (sbatch runs in scratch; returns to submit dir)
    inp_basename = inp_path.name
    out_basename = f"{jt}.out"
    orca_cmd = f"orca {inp_basename} > {out_basename}"
    render_template(
        sbatch_tpl,
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


@app.command("sbatch-submit")
def sbatch_submit(job: Path = typer.Argument(..., help="Job DIR or job.sbatch file")):
    """Submit SLURM job and print the job id."""
    job = Path(job).expanduser().resolve()
    if job.is_dir():
        jobdir = job
        sbfile = jobdir / "job.sbatch"
    else:
        sbfile = job
        jobdir = sbfile.parent
    if not sbfile.exists():
        raise typer.BadParameter(f"{sbfile} not found")
    out = subprocess.check_output(["sbatch", "job.sbatch"], text=True, cwd=str(jobdir)).strip()
    m = re.search(r"(\d+)", out)
    jobid = m.group(1) if m else "UNKNOWN"
    typer.secho(f"Submitted: {out}", fg=typer.colors.GREEN)
    typer.echo(jobid)


@app.command("sbatch-wait-parse")
def sbatch_wait_parse(
    jobdir: Path = typer.Argument(..., help="Folder created by `prep`"),
    poll: float = typer.Option(20.0, help="Poll interval (seconds)"),
    timeout: int = typer.Option(0, help="Timeout seconds (0 = no timeout)"),
    out_jsonl: Optional[Path] = typer.Option(None, "--out-jsonl", help="Append parsed record here"),
):
    """Wait for .out to appear in JOBDIR, then parse it to JSON (optionally append to JSONL)."""
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

if __name__ == "__main__":
    app()
