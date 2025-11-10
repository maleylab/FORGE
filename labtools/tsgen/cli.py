import argparse
from pathlib import Path
from .pipeline import run_tsgen


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `labtools tsgen` subcommand."""
    p = subparsers.add_parser(
        "tsgen",
        help="Generate TS-guess seeds, ORCA inputs, and SLURM array+collector",
        description=(
            "Local-region interpolation with optional IDPP smoothing; render "
            "ORCA B97-3c Opt Freq (±CPCM) with Cartesian-only constraints; "
            "write a SLURM array job and afterok collector that extracts ONLY "
            "imaginary frequencies. 0-based indices everywhere."
        ),
    )

    # Required
    p.add_argument("--from", dest="src_A", required=True, help="Endpoint A XYZ")
    p.add_argument("--to", dest="src_B", required=True, help="Endpoint B XYZ")
    p.add_argument("--outdir", required=True, help="Output campaign directory")
    p.add_argument("--campaign-id", required=True, help="Campaign ID")

    # Geometry prep
    p.add_argument("--n", dest="n_frames", type=int, default=21, help="Frame count (incl. endpoints)")
    p.add_argument("--map", dest="mapping_csv", default="auto", help="'auto' or CSV old->new indices for B")
    p.add_argument("--anchors", default="auto", help="'auto' or i,j,k (0-based)")
    p.add_argument("--align", default="anchors", choices=["none","kabsch","anchors"], help="Rigid alignment mode")
    p.add_argument("--interp", default="idpp", choices=["linear","mass","idpp"], help="Interpolation mode for reacting region")
    p.add_argument(
        "--region",
        required=True,
        help="reactants:i,j,k spectators:freeze|restrain(k)",
    )

    # Constraints & phases (Cartesian only)
    p.add_argument("--cart-freeze", default="", help="'full: i,j; x: ; y: ; z:'")
    p.add_argument(
        "--phases",
        default="phase1:freeze=reactants+spectators+full; phase2:unfreeze=spectators",
        help="Freeze/unfreeze DSL across phases",
    )

    # Calc options (fixed policy)
    p.add_argument("--solvent", default="none", help="none|<name>|custom (CPCM)")
    p.add_argument("--solvent.epsilon", dest="epsilon", type=float, help="CPCM dielectric if custom")
    p.add_argument("--solvent.refrac", dest="refrac", type=float, help="CPCM refractive index (optional)")

    # SLURM array
    p.add_argument("--array-parallel", type=int, default=40, help="Max concurrent array tasks (%P)")
    p.add_argument("--pal", dest="pal_threads", type=int, default=8, help="PAL threads; also cpus-per-task")
    p.add_argument("--maxcore", dest="maxcore_mb", type=int, default=3000, help="%maxcore per thread (MB)")

    # Seed selection
    p.add_argument("--seed.select", dest="seed_mode", default="heuristic", choices=["heuristic","mid","all"], help="Which frames become seeds")
    p.add_argument("--seed.k", dest="seed_k", type=int, default=3, help="Top-k seeds if heuristic")

    p.set_defaults(func=_dispatch)


def _dispatch(args: argparse.Namespace) -> None:
    run_tsgen(args)


