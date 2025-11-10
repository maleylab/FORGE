"""
tsgen.pipeline
Orchestrates the luxury transition-state workflow.
Stages: L0 → L1 → L2 → L3
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from .stages import l0_seed, l1_opt, l2_opt, l3_sp, verify
from .fingerprint import load_fingerprint


class TSPipeline:
    def __init__(
        self,
        reactant: Path,
        product: Path,
        charge: int,
        mult: int,
        outdir: Path,
        methods: dict[str, str],
        fingerprint: Path | None = None,
        mode: str = "array",
        profile: str = "medium",
    ):
        self.reactant = Path(reactant)
        self.product = Path(product)
        self.charge = charge
        self.mult = mult
        self.outdir = Path(outdir)
        self.methods = methods
        self.fingerprint = load_fingerprint(fingerprint) if fingerprint else None
        self.mode = mode
        self.profile = profile
        self._init_outdir()

    def _init_outdir(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        meta = {
            "created": datetime.now().isoformat(),
            "reactant": str(self.reactant),
            "product": str(self.product),
            "charge": self.charge,
            "mult": self.mult,
            "methods": self.methods,
            "mode": self.mode,
            "profile": self.profile,
        }
        (self.outdir / "run.json").write_text(json.dumps(meta, indent=2))

    def run(self):
        print("[FORGE] Starting luxury TS workflow")

        # L0: Interpolation + XTB2 preopt + Freq + fingerprint gate
        ts_seeds = l0_seed.generate_guesses(
            self.reactant, self.product,
            method=self.methods["L0"],
            charge=self.charge, mult=self.mult,
            outdir=self.outdir / "L0",
            fingerprint=self.fingerprint,
            mode=self.mode, profile=self.profile,
        )

        # L1: r2SCAN-3c OptTS (+ verify)
        ts_l1 = l1_opt.run_opt(
            ts_seeds, method=self.methods["L1"],
            outdir=self.outdir / "L1",
            charge=self.charge, mult=self.mult,
            fingerprint=self.fingerprint,
            mode=self.mode, profile=self.profile,
        )

        # Verify L1 products (optional extra gate)
        verify.check(self.outdir / "L1", self.fingerprint, outdir=self.outdir / "verify_L1")

        # L2: production DFT OptTS Freq (+ restart + verify)
        ts_l2 = l2_opt.run_opt(
            ts_l1, method=self.methods["L2"],
            outdir=self.outdir / "L2",
            charge=self.charge, mult=self.mult,
            fingerprint=self.fingerprint,
            mode=self.mode, profile=self.profile,
        )
        verify.check(self.outdir / "L2", self.fingerprint, outdir=self.outdir / "verify_L2")

        # L3: optional DLPNO-CCSD(T) SP
        if "L3" in self.methods:
            l3_sp.run_sp(
                ts_l2, method=self.methods["L3"],
                outdir=self.outdir / "L3",
                charge=self.charge, mult=self.mult,
                mode=self.mode, profile=self.profile,
            )

        print("[FORGE] Pipeline complete ✔")
        return ts_l2

def run_tsgen(*args, **kwargs):
    """
    Back-compat shim used by the legacy tsgen CLI.

    Supports either:
      1) run_tsgen(namespace)  # where namespace has .reactant, .product, .charge, .mult, .outdir, .methods, etc.
      2) run_tsgen(reactant=..., product=..., charge=..., mult=..., outdir=..., methods=..., fingerprint=None, mode="array", profile="medium")

    Returns whatever TSPipeline.run() returns.
    """
    # Case 1: single argparse/typer namespace passed in positionally
    if len(args) == 1 and not kwargs:
        ns = args[0]
        methods = getattr(ns, "methods", None)
        if methods is None:
            # Some older CLIs pass L0/L1/L2/L3 as separate fields
            methods = {"L0": ns.L0, "L1": ns.L1, "L2": ns.L2}
            if getattr(ns, "L3", None):
                methods["L3"] = ns.L3
        pipe = TSPipeline(
            reactant=getattr(ns, "reactant"),
            product=getattr(ns, "product"),
            charge=getattr(ns, "charge"),
            mult=getattr(ns, "mult"),
            outdir=getattr(ns, "outdir"),
            methods=methods,
            fingerprint=getattr(ns, "fingerprint", None),
            mode=getattr(ns, "mode", "array"),
            profile=getattr(ns, "profile", "medium"),
        )
        return pipe.run()

    # Case 2: keyword-style invocation
    reactant = kwargs["reactant"]
    product  = kwargs["product"]
    charge   = kwargs["charge"]
    mult     = kwargs["mult"]
    outdir   = kwargs["outdir"]
    methods  = kwargs["methods"]
    fingerprint = kwargs.get("fingerprint", None)
    mode     = kwargs.get("mode", "array")
    profile  = kwargs.get("profile", "medium")

    pipe = TSPipeline(
        reactant=reactant, product=product,
        charge=charge, mult=mult,
        outdir=outdir,
        methods=methods,
        fingerprint=fingerprint,
        mode=mode, profile=profile,
    )
    return pipe.run()