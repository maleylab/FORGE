from __future__ import annotations

import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from labtools.plans.adapters import planentry_to_dict, planentry_to_render_context
from labtools.plans.types import PlanEntry

PLAN_TASK_TEMPLATES = {
    "sp": "orca/orca_sp.inp.j2",
    "opt": "orca/orca_opt.inp.j2",
    "freq": "orca/orca_freq.inp.j2",
    "optfreq": "orca/orca_optfreq.inp.j2",
    "nmr": "orca/orca_nmr.inp.j2",
    "sp-triplet": "orca/orca_sp_triplet.inp.j2",
    "gradient": "orca/orca_grad.inp.j2",
    "tsopt": "orca/orca_tsopt.inp.j2",
    "irc": "orca/orca_irc.inp.j2",
}


def _serialize_planentry(entry: PlanEntry) -> dict:
    return {
        "id": entry.id,
        "schema": {
            "name": entry.schema_name,
            "version": entry.schema_version,
        },
        "intent": {
            "task": entry.task,
            "system": entry.system,
        },
        "parameters": entry.parameters,
        "metadata": {
            "tags": entry.tags,
            "notes": entry.notes,
        },
    }


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_dirname(s: str) -> str:
    """
    Filesystem-safe name:
      - allow A–Z a–z 0–9 . _ -
      - replace everything else with _
    """
    s = (s or "").strip()
    if not s:
        return "job"
    return _SAFE_CHARS_RE.sub("_", s)


def assign_job_dirnames(entries: Sequence[PlanEntry], *, outdir: Path) -> List[str]:
    """
    Assign deterministic, unique job directory names for a batch of PlanEntries.

    This is intentionally computed in the parent process so workers do not race on
    filesystem existence checks when plan rendering is parallelized.

    Existing directories already present under ``outdir`` are treated as occupied so
    rerendering into a populated output directory preserves the old serial behavior of
    suffixing collisions rather than failing immediately.
    """
    used = {p.name for p in outdir.iterdir() if p.exists()} if outdir.exists() else set()
    names: List[str] = []

    for i, entry in enumerate(entries):
        base_raw = entry.id if getattr(entry, "id", None) else f"job_{i:05d}"
        base = _sanitize_dirname(base_raw)
        candidate = base
        n = 1
        while candidate in used:
            candidate = f"{base}__{n}"
            n += 1
        used.add(candidate)
        names.append(candidate)

    return names


def get_orca_template_env() -> Environment:
    """
    Canonical ORCA Jinja environment.
    Safe defaults, no StrictUndefined explosions.
    """

    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "templates" / "orca"
        if cand.is_dir():
            return Environment(
                loader=FileSystemLoader(str(cand.parent)),
                undefined=ChainableUndefined,
                autoescape=False,
            )

    raise RuntimeError("Could not locate templates/orca directory")


def render_single_job_orca(job: Dict[str, Any], *, job_dir: Path) -> None:
    """Render a single normalized ORCA render-context dict to ``job.inp``."""

    task = job.get("job_type") or job.get("task")
    if not task:
        raise ValueError("PlanEntry missing parameters.job_type / task")
    task = str(task).strip().lower()
    if task not in PLAN_TASK_TEMPLATES:
        raise ValueError(f"Unknown task {task!r}")

    structure = job.get("structure") or job.get("system")
    if not structure:
        raise ValueError("PlanEntry missing structure/system path")

    structure = Path(str(structure))
    if not structure.is_file():
        raise FileNotFoundError(f"Structure file not found: {structure}")

    geom_lines = job.get("geom_lines") if isinstance(job.get("geom_lines"), list) else []
    if not geom_lines:
        lines = structure.read_text(encoding="utf-8").strip().splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[2:]
        geom_lines = lines

    method = job.get("method")
    if not method:
        raise ValueError("PlanEntry missing parameters.method")

    charge = job.get("charge", 0)
    mult = job.get("multiplicity", job.get("mult", 1))

    try:
        charge = int(charge)
    except Exception as exc:
        raise ValueError(f"Invalid charge: {charge!r}") from exc

    try:
        mult = int(mult)
    except Exception as exc:
        raise ValueError(f"Invalid multiplicity: {mult!r}") from exc

    def _as_dict(x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _as_list(x: Any) -> List[Any]:
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

    params = job.get("parameters") if isinstance(job.get("parameters"), dict) else {}
    row: Dict[str, Any] = {}
    if isinstance(job.get("_row_raw"), dict):
        row = job.get("_row_raw")  # type: ignore[assignment]
    elif isinstance(params.get("_row_raw"), dict):
        row = params.get("_row_raw")  # type: ignore[assignment]

    srcs = [params, job, row]

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

    ctx: Dict[str, Any] = {
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

    env = get_orca_template_env()
    template = env.get_template(PLAN_TASK_TEMPLATES[task])
    text = template.render(**ctx)

    lines = text.splitlines()
    cleaned: List[str] = []
    prev_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        cleaned.append(line.rstrip())
        prev_blank = blank

    text = "\n".join(cleaned).strip() + "\n"
    (job_dir / "job.inp").write_text(text, encoding="utf-8")




def _ensure_parallel_safe_render_func(render_func: Callable[..., None]) -> None:
    try:
        pickle.dumps(render_func)
    except Exception as exc:
        name = getattr(render_func, "__qualname__", repr(render_func))
        raise TypeError(
            "jobs>1 requires a top-level pickleable render_func; "
            f"got {name}"
        ) from exc

def _render_one_preassigned_job(item) -> str:
    entry, outdir_str, job_dir_name, write_plan_entry_json, render_func = item
    outdir = Path(outdir_str)
    job_dir = outdir / job_dir_name

    try:
        job_dir.mkdir(exist_ok=False)

        legacy_job = planentry_to_render_context(entry)
        render_func(legacy_job, job_dir=job_dir)

        if write_plan_entry_json:
            (job_dir / "plan_entry.json").write_text(
                json.dumps(planentry_to_dict(entry), indent=2),
                encoding="utf-8",
            )

        return job_dir_name
    except Exception as exc:
        entry_id = getattr(entry, "id", "<unknown>")
        raise RuntimeError(
            f"Plan render failed for entry {entry_id!r} in {job_dir}"
        ) from exc


def render_planentries(
    entries: Iterable[PlanEntry],
    *,
    render_func: Callable[..., None],
    outdir: Path,
    system_key: str = "system",
    write_plan_entry_json: bool = True,
    jobs: int = 1,
):
    """Render PlanEntries using a renderer function.

    ``jobs=1`` preserves serial behavior. ``jobs>1`` uses a process pool and
    requires a top-level, pickleable ``render_func``.
    """
    del system_key

    if jobs < 1:
        raise ValueError("render_planentries() requires jobs >= 1")

    outdir.mkdir(parents=True, exist_ok=True)

    entries_list = list(entries)
    job_dir_names = assign_job_dirnames(entries_list, outdir=outdir)
    work_items = [
        (entry, str(outdir), job_dir_name, write_plan_entry_json, render_func)
        for entry, job_dir_name in zip(entries_list, job_dir_names)
    ]

    if jobs == 1 or len(work_items) <= 1:
        for item in work_items:
            _render_one_preassigned_job(item)
        return

    _ensure_parallel_safe_render_func(render_func)

    max_workers = min(jobs, len(work_items))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_render_one_preassigned_job, item) for item in work_items]
        try:
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
