from __future__ import annotations
from labtools.orca.restart_overrides import RestartOverrides


# ------------------------
# SCF Strategies
# ------------------------

def scf_loosen() -> RestartOverrides:
    return RestartOverrides(
        global_flags_add=["SlowConv"],
        scf={
            "MaxIter": 300,
        }
    )


def scf_aggressive() -> RestartOverrides:
    return RestartOverrides(
        global_flags_add=["SlowConv"],
        scf={
            "MaxIter": 500,
            "DIISMaxEq": 10,
        }
    )


def scf_reset_guess() -> RestartOverrides:
    return RestartOverrides(
        global_flags_add=["SlowConv"],
        scf={
            "MaxIter": 300,
            "SOSCF": True,
            "InitialGuess": "HCore",
        }
    )


# ------------------------
# GEOMETRY Strategies
# ------------------------

def geom_restart() -> RestartOverrides:
    """
    Tell ORCA to restart geometry optimization.
    """
    return RestartOverrides(
        geom={
            "Restart": True,
            "MaxIter": 200,
        }
    )


def geom_loosen() -> RestartOverrides:
    return RestartOverrides(
        global_flags_add=["LooseOpt"],
        geom={
            "MaxIter": 200,
        }
    )


def geom_trustradius() -> RestartOverrides:
    return RestartOverrides(
        geom={
            "TrustRadius": 0.1,
        }
    )


# ------------------------
# TS Strategies
# ------------------------

def ts_relax() -> RestartOverrides:
    return RestartOverrides(
        geom={
            "TS": True,
            "Calc_Hess": False,
            "MaxIter": 200,
            "Restart": True,
        }
    )


def ts_recompute_hessian() -> RestartOverrides:
    return RestartOverrides(
        geom={
            "TS": True,
            "Calc_Hess": True,
            "MaxIter": 200,
        }
    )


# ------------------------
# Frequency Strategies
# ------------------------

def freq_numeric() -> RestartOverrides:
    return RestartOverrides(
        freq={"NumFreq": True}
    )
