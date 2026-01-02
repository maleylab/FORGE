"""
TSGen 2.1 | Fingerprint path resolver
Ensures fingerprint files can be located in any of the following locations:

1. Absolute path
2. Relative to CWD
3. Relative to the TSGen work directory
4. <repo-root>/resources/tsfp/
"""

from __future__ import annotations
from pathlib import Path


def _find_repo_tsfp_dir(start: Path, max_up: int = 8) -> Path | None:
    """
    Walk upward from `start` to locate <repo-root>/resources/tsfp.

    This supports editable installs where the repo root is
    not the same as the Python package root.
    """
    p = start.resolve()
    for _ in range(max_up):
        candidate = p / "resources" / "tsfp"
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return None


def resolve_fingerprint_path(path: str | Path, base: Path | None = None) -> Path:
    """
    Returns a canonical absolute Path to the fingerprint YAML.

    Resolution order:
    1. Absolute path
    2. Relative to current working directory
    3. Relative to the TSGen work directory
    4. <repo-root>/resources/tsfp/
    """

    p = Path(path)

    # 1. Absolute path
    if p.is_absolute() and p.is_file():
        return p.resolve()

    # 2. Relative to CWD
    p2 = Path.cwd() / p
    if p2.is_file():
        return p2.resolve()

    # 3. Relative to TSGen work directory
    if base:
        p3 = Path(base) / p
        if p3.is_file():
            return p3.resolve()

    # 4. Repo-root resources/tsfp
    tsfp_root = _find_repo_tsfp_dir(Path(__file__))
    if tsfp_root:
        p4 = tsfp_root / p.name
        if p4.is_file():
            return p4.resolve()

    raise FileNotFoundError(
        f"Could not resolve fingerprint file '{path}'. "
        f"Checked: absolute, CWD, plan workdir, <repo-root>/resources/tsfp/"
    )
