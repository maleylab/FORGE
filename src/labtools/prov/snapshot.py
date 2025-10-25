from __future__ import annotations
import os, sys, json, hashlib, platform, subprocess, datetime
from typing import Dict, Any, Optional

def _run(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=10)
        return out.strip()
    except Exception as e:
        return f"__ERROR__: {e}"

def provenance_snapshot(project: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"Collect a lightweight, reproducible runtime snapshot (safe on local or HPC).\"\"\"
    module_list = _run("module list -t")
    git_sha = _run("git rev-parse HEAD")
    if git_sha.startswith("__ERROR__"):
        git_sha = None

    py_ver = sys.version.split()[0]
    try:
        pip_freeze = _run("python -m pip freeze")
        if pip_freeze.startswith("__ERROR__"):
            pip_freeze = None
    except Exception:
        pip_freeze = None

    env_subset = {k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in ("MODULEPATH", "LOADEDMODULES", "EBROOTORCA")}

    data_for_hash = json.dumps({
        "module_list": module_list,
        "py_ver": py_ver,
        "pip_freeze": pip_freeze,
        "env": env_subset,
    }, sort_keys=True)

    env_hash = hashlib.sha256(data_for_hash.encode("utf-8")).hexdigest()

    snap = {
        "schema_version": "0.1.0",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "project": project,
        "host": platform.node(),
        "system": {
            "platform": platform.platform(),
            "python": py_ver,
        },
        "vcs": {"git_sha": git_sha},
        "hpc": {
            "module_list": module_list,
            "env": env_subset,
        },
        "packages": {"pip_freeze": pip_freeze},
        "env_hash": env_hash,
    }
    return snap
