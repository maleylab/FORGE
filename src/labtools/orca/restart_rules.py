# restart_rules.py

RESTART_RULES = {

    # ==============================
    # SCF failure modes (subruleset)
    # ==============================
    "SCF_NOT_CONVERGED": {
        "scf": {
            "MaxIter": {"increment": 100},
            "Damp": [0.50, 5],
            "Shift": [0.30, 3],
            "Convergence": "Looser",
            "SOSCF": True,
        }
    },

    "SCF_OSCILLATION": {
        "scf": {
            "Damp": [0.70, 10],
            "Shift": [0.50, 5],
            "MaxIter": {"increment": 200},
            "SOSCF": True,
        }
    },

    "SCF_SLOW_CONVERGENCE": {
        "scf": {
            "MaxIter": {"increment": 200},
            "Convergence": "Looser",
            "SOSCF": True,
        }
    },

    # ===================================================
    # OPT general failure modes (from ORCA manual)
    # ===================================================

    "GRAD_TOO_LARGE": {
        "geom": {
            "Trust": {"scale": 0.5},
            "StepSize": {"scale": 0.5},
            "MaxIter": {"increment": 50},
        }
    },

    "STEP_TOO_SMALL": {
        "geom": {
            "Trust": {"scale": 1.5},
            "Recalc_Hess": 1,
            "MaxIter": {"increment": 100},
        }
    },

    "HESSIAN_SINGULAR_OR_BAD": {
        "geom": {
            "Calc_Hess": True,
            "Recalc_Hess": 1,
            "Trust": 0.1,
            "StepSize": 0.1,
        }
    },

    "GEOMETRY_EXPLODED": {
        "geom": {
            "Trust": 0.05,
            "StepSize": 0.05,
            "Recalc_Hess": 1,
            "MaxIter": {"increment": 200},
        }
    },

    # ===================================================
    # TSOPT-specific failures
    # ===================================================

    "TS_WRONG_IMAG_MODE": {
        "geom": {
            "Calc_Hess": True,
            "Recalc_Hess": 1,
            "Trust": 0.1,
        }
    },

    "TS_NO_IMAGINARY_FREQ": {
        "geom": {
            "Calc_Hess": True,
            "Recalc_Hess": 1,
            "Trust": 0.05,
            "StepSize": 0.05,
        }
    },

    "TS_NOT_CONVERGING": {
        "geom": {
            "Trust": {"scale": 0.5},
            "StepSize": {"scale": 0.5},
            "MaxIter": {"increment": 100},
        }
    },

    "TS_GEOMETRY_EXPLODED": {
        "geom": {
            "Trust": 0.02,
            "StepSize": 0.02,
            "Recalc_Hess": 1,
            "MaxIter": {"increment": 200},
        }
    },

    "TS_HESSIAN_BAD": {
        "geom": {
            "Calc_Hess": True,
            "Recalc_Hess": 1,
            "Trust": 0.1,
        }
    },

    # ===================================================
    # FREQ failures
    # ===================================================

    "FREQ_SCF_FAIL": {
        "scf": "inherit_scf"
    },

    "FREQ_ANALYTIC_FAIL": {
        "freq": {
            "Analytic": False,
        }
    },

    "FREQ_GENERAL_FAIL": {
        "freq": {
            "Analytic": False,
        }
    },
}
