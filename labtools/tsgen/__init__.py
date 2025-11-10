"""Transition‑state guess generation (tsgen) package for labtools.

Exports the subcommand add_subparser for CLI wiring.
"""
from .cli import add_subparser  # re-export for labtools.cli

__all__ = ["add_subparser"]

