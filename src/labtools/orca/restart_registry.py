from __future__ import annotations
from labtools.orca.restart_strategies import (
    scf_loosen, scf_aggressive, scf_reset_guess,
    geom_restart, geom_loosen, geom_trustradius,
    ts_relax, ts_recompute_hessian,
    freq_numeric
)


STRATEGY_REGISTRY = {
    # SCF failures
    "scf_convergence": [scf_loosen, scf_aggressive, scf_reset_guess],

    # Geometry failures
    "geom_convergence": [geom_restart, geom_loosen, geom_trustradius],

    # TS-specific failures
    "ts_failed": [ts_relax, ts_recompute_hessian],

    # Frequency failures
    "freq_imag": [freq_numeric],

    # Fallback for unknown
    "unknown_failure": [geom_restart],
}
