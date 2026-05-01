"""Compatibility helpers for creating one-off ORCA job directories.

This module contains the backend used by ``forge job create`` and
``forge job import``.  It intentionally preserves the older single-job CLI
behavior while moving the rendering/writing logic out of ``cli.py`` so the
command can later be routed through the canonical FORGE pipeline renderer.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from labtools.plans.render import render_single_job_orca


RenderTemplateFn = Callable[[Path, Path, Dict[str, Any]], Optional[str]]


def read_xyz_geom_lines(xyz_path: Path) -> List[str]:
    """Read geometry lines from an XYZ-like file.

    Standard XYZ headers are stripped when present; otherwise all non-empty
    lines are returned unchanged apart from trailing whitespace removal.
    """
    xyz_path = xyz_path.expanduser().resolve()
    if not xyz_path.is_file():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 3:
        try:
            int(lines[0].strip())
            return [ln.rstrip() for ln in lines[2:] if ln.strip()]
        except Exception:
            pass
    return [ln.rstrip() for ln in lines if ln.strip()]


def orca_template_for_task(task: str, *, templates_root: Path) -> Path:
    """Return the ORCA input template for a single-job task."""
    task = task.strip().lower()
    root = templates_root / "orca"
    mapping = {
        "sp": "orca_sp.inp.j2",
        "opt": "orca_opt.inp.j2",
        "freq": "orca_freq.inp.j2",
        "optfreq": "orca_optfreq.inp.j2",
    }
    if task not in mapping:
        raise ValueError(f"Unsupported task: {task}. Choose from: {', '.join(sorted(mapping))}")
    tpl = root / mapping[task]
    if not tpl.is_file():
        raise FileNotFoundError(f"Template not found for task '{task}': {tpl}")
    return tpl


def sbatch_template(*, templates_root: Path) -> Path:
    """Return the single ORCA job SBATCH template."""
    tpl = templates_root / "sbatch" / "single_orca_job.sbatch.j2"
    if not tpl.is_file():
        raise FileNotFoundError(f"SBATCH template not found: {tpl}")
    return tpl


def default_job_name_from_xyz(xyz: Path, task: str) -> str:
    """Return the historical default single-job directory name."""
    return f"{xyz.stem}_{task}"


def build_minimal_payload(
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
    time: Optional[str] = None,
    maxiter: Optional[int],
    cpcm_eps: Optional[float],
    cpcm_refrac: Optional[float],
) -> Dict[str, Any]:
    """Build the minimal PlanEntry-like payload used by one-off jobs.

    The unused ``time`` parameter is retained for call-site compatibility with
    the older CLI helper signature.
    """
    del time
    geom_lines = read_xyz_geom_lines(xyz)

    geom = {
        "maxiter": int(maxiter) if maxiter is not None else 200,
        "constraints": [],
        "restart": False,
        "Restart": False,
    }

    xyz_resolved = xyz.expanduser().resolve()

    payload: Dict[str, Any] = {
        "id": default_job_name_from_xyz(xyz, task),
        "task": task,
        "job_type": task,
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
        "multiplicity": int(mult),
        "structure": str(xyz_resolved),
        "system": str(xyz_resolved),
        "xyz": str(xyz_resolved),
        "geom_lines": geom_lines,
        "restart": {"enabled": False, "file": None, "flags": []},
    }

    if cpcm_eps is not None:
        payload["cpcm"] = {
            "epsilon": float(cpcm_eps),
            "refrac": float(cpcm_refrac) if cpcm_refrac is not None else None,
        }

    return payload


def mem_per_cpu_from_maxcore(maxcore_mb: Optional[int]) -> str:
    """Derive a simple SLURM ``--mem-per-cpu`` value from ORCA maxcore."""
    if maxcore_mb is None:
        return "4G"
    gb = (int(maxcore_mb) + 1023) // 1024
    gb = max(1, gb)
    return f"{gb}G"


def safe_clear_dir(d: Path) -> None:
    """Remove all direct children of a directory."""
    for child in d.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()



def _canonical_render_payload(payload: Dict[str, Any], *, task: str, xyz: Optional[Path], tmp_root: Path) -> Dict[str, Any]:
    """Return a payload suitable for the canonical ORCA single-job renderer.

    Older one-off jobs could be rendered from ``geom_lines`` alone. The
    canonical FORGE renderer expects a structure/system path, so when no input
    structure is available we materialize a temporary XYZ from ``geom_lines``
    inside the staging directory.
    """
    render_payload = dict(payload)
    render_payload.setdefault("task", task)
    render_payload.setdefault("job_type", task)

    structure_value = render_payload.get("structure") or render_payload.get("system") or render_payload.get("xyz")
    structure_path: Optional[Path] = None
    if structure_value:
        candidate = Path(str(structure_value)).expanduser()
        if candidate.is_file():
            structure_path = candidate.resolve()

    if structure_path is None and xyz is not None:
        candidate = xyz.expanduser()
        if candidate.is_file():
            structure_path = candidate.resolve()

    if structure_path is None:
        geom_lines = render_payload.get("geom_lines")
        if isinstance(geom_lines, list) and geom_lines:
            clean_lines = [str(line).rstrip() for line in geom_lines if str(line).strip()]
            if clean_lines:
                structure_path = tmp_root / "structure.xyz"
                structure_path.write_text(
                    f"{len(clean_lines)}\nGenerated by forge job create/import\n" + "\n".join(clean_lines) + "\n",
                    encoding="utf-8",
                )

    if structure_path is None:
        raise ValueError("Single-job render requires a structure/system/xyz path or geom_lines")

    render_payload["structure"] = str(structure_path)
    render_payload["system"] = str(structure_path)
    render_payload.setdefault("xyz", str(structure_path))
    return render_payload


def validate_job_input_render(*, payload: Dict[str, Any], task: str, xyz: Optional[Path] = None) -> None:
    """Validate that the canonical ORCA renderer can render this single job."""
    with tempfile.TemporaryDirectory(prefix="forge_job_render_check_") as td:
        tmp_root = Path(td)
        render_payload = _canonical_render_payload(payload, task=task, xyz=xyz, tmp_root=tmp_root)
        render_single_job_orca(render_payload, job_dir=tmp_root)

def write_job_dir(
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
    templates_root: Path,
    render_template_fn: Callable[..., Optional[str]],
) -> None:
    """Write a one-off ORCA job directory using the historical layout."""
    if final_dir.exists():
        if exists == "fail":
            raise FileExistsError(f"Job directory already exists: {final_dir}")
        if exists == "skip":
            return
        if exists == "overwrite":
            safe_clear_dir(final_dir)
        else:
            raise ValueError("--exists must be one of: fail|skip|overwrite")
    else:
        final_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix=f".tmp_{job_name}_", dir=str(final_dir.parent)))
    try:
        render_payload = _canonical_render_payload(payload, task=task, xyz=xyz, tmp_root=tmp_root)
        render_single_job_orca(render_payload, job_dir=tmp_root)
        (tmp_root / "plan_entry.json").write_text(json.dumps(render_payload, indent=2) + "\n", encoding="utf-8")

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
            sb_text = render_template_fn(
                sbatch_template(templates_root=templates_root),
                tmp_root / "job.sbatch",
                sbatch_params,
                return_text=True,
            )
            (tmp_root / "job.sbatch").write_text(sb_text or "", encoding="utf-8")

        if write_ready:
            (tmp_root / "READY").write_text("", encoding="utf-8")

        if final_dir.exists() and exists == "overwrite":
            safe_clear_dir(final_dir)
        tmp_root.replace(final_dir)
    finally:
        if tmp_root.exists() and tmp_root != final_dir:
            shutil.rmtree(tmp_root, ignore_errors=True)
