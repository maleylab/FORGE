# labtools/tsgen/fp/fingerprint_path.py

from __future__ import annotations
from pathlib import Path
import os
import labtools


def resolve_fingerprint_path(path: str | Path, plan_dir: Path | None = None) -> Path:
    """
    Resolve fingerprint file using the priority:

    1. Absolute path
    2. CWD-relative
    3. Plan directory (plan.work_dir)
    4. $LABTOOLS_HOME/resources/tsfp/<name>
    5. Installed labtools package: labtools/resources/tsfp/<name>

    Returns:
        Path to fingerprint file

    Raises:
        FileNotFoundError if not found anywhere.
    """

    path = Path(path)
    name = path.name

    # --- 1. Absolute path -----------------------------------------------
    if path.is_absolute() and path.is_file():
        return path.resolve()

    # --- 2. CWD ----------------------------------------------------------
    cwd_candidate = Path.cwd() / name
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()

    # --- 3. Plan directory ----------------------------------------------
    if plan_dir:
        plan_candidate = Path(plan_dir) / name
        if plan_candidate.is_file():
            return plan_candidate.resolve()

    # --- 4. LABTOOLS_HOME -----------------------------------------------
    env_home = os.environ.get("LABTOOLS_HOME")
    if env_home:
        env_candidate = Path(env_home).expanduser() / "resources" / "tsfp" / name
        if env_candidate.is_file():
            return env_candidate.resolve()

    # --- 5. Installed package location ---------------------------------
    pkg_root = Path(labtools.__file__).resolve().parents[0]
    pkg_candidate = pkg_root / "resources" / "tsfp" / name
    if pkg_candidate.is_file():
        return pkg_candidate.resolve()

    # --------------------------------------------------------------------
    # Not found → error
    # --------------------------------------------------------------------
    raise FileNotFoundError(
        f"Could not resolve fingerprint file '{name}'. Checked:\n"
        f"  - absolute path: {path}\n"
        f"  - cwd: {cwd_candidate}\n"
        f"  - plan_dir: {plan_dir}\n"
        f"  - $LABTOOLS_HOME/resources/tsfp/{name}\n"
        f"  - installed package: {pkg_candidate}\n\n"
        f"Hint: set LABTOOLS_HOME to your repo root:\n"
        f"  export LABTOOLS_HOME=/home/$USER/lab-tools\n"
    )
