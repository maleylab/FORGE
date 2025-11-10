# src/labtools/tsgen/slurm.py
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

def _env(templates_root: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_root)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        trim_blocks=True, lstrip_blocks=True,
    )

def write_array_sbatch(path: Path, outdir: Path, n_seeds: int, parallel: int, pal_threads: int,
                       *, account: str | None = None, time: str = "24:00:00", mem: str = "24G"):
    templates_root = Path(__file__).resolve().parents[3] / "templates" / "sbatch"
    env = _env(templates_root)
    tmpl = env.get_template("tsgen_array.sbatch.j2")
    text = tmpl.render(
        job_name="tsgen_array",
        account=account,
        time=time,
        mem=mem,
        cpus_per_task=pal_threads,
        array_last=n_seeds - 1,
        array_parallel=parallel,
        seeds_dir=str(outdir / "seeds"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def write_collect_sbatch(path: Path, outdir: Path, array_jobid: str | None = None,
                         *, account: str | None = None, time: str = "01:00:00", mem: str = "2G"):
    templates_root = Path(__file__).resolve().parents[3] / "templates" / "sbatch"
    env = _env(templates_root)
    tmpl = env.get_template("tsgen_collect.sbatch.j2")
    text = tmpl.render(
        job_name="tsgen_collect",
        account=account,
        time=time,
        mem=mem,
        array_jobid=array_jobid or "${ARRAY_JOBID}",  # allow filling later
        seeds_dir=str(outdir / "seeds"),
        outdir=str(outdir),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _env(templates_root: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_root)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        trim_blocks=True, lstrip_blocks=True,
    )

def write_worker_sbatch(path: Path, outdir: Path, *,
                        pal: int = 8, maxcore_mb: int = 3000,
                        time: str = "24:00:00", mem: str = "24G",
                        partition: str | None = None, account: str | None = None, qos: str | None = None,
                        max_tasks: int = 1000, stop_after_min: int = 120, cooldown_sec: int = 5):
    templates_root = Path(__file__).resolve().parents[3] / "templates" / "sbatch"
    env = _env(templates_root)
    tmpl = env.get_template("tsgen_worker.sbatch.j2")
    text = tmpl.render(
        seeds_dir=str(outdir / "seeds"),
        pal=pal, maxcore_mb=maxcore_mb,
        time=time, mem=mem,
        partition=partition, account=account, qos=qos,
        max_tasks=max_tasks, stop_after_min=stop_after_min, cooldown_sec=cooldown_sec,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

