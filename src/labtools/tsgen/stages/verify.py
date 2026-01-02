"""
TSGen 2.1 | VERIFY Stage
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

from ..fp.fingerprint import (
    extract_ts_mode_from_orca,
    cosine_similarity,
    mode_localization_fraction,
)
from ..tsgen_orca import parse_frequencies_and_modes


def run_verify(controller, out_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    VERIFY stage.

    Accepts either:
      • TSGenController (pipeline use)
      • TSGenPlan (stand-alone use)
    """

    # -------------------------------------------------
    # Normalize inputs (controller vs plan)
    # -------------------------------------------------
    if hasattr(controller, "plan"):
        plan = controller.plan
        fingerprint = controller.fingerprint
        ref_mode = controller.ref_mode
        cos_thresh = controller.cos_thresh
        loc_thresh = controller.loc_thresh
        debug = controller.debug
    else:
        plan = controller
        fingerprint = None
        ref_mode = None
        cos_thresh = None
        loc_thresh = None
        debug = False

    mode = plan.mode
    results: List[Dict[str, Any]] = []

    for i, out_path in enumerate(out_paths):

        entry = {
            "index": i,
            "path": str(out_path),
            "status": None,
            "cosine": None,
            "localization": None,
            "passed": False,
        }

        # ------------------------------
        # DEBUG → force pass
        # ------------------------------
        if debug:
            entry["status"] = "debug_force_pass"
            entry["passed"] = True
            results.append(entry)
            continue

        # ------------------------------
        # Parse frequencies
        # ------------------------------
        try:
            freqs, _ = parse_frequencies_and_modes(out_path)
        except Exception:
            entry["status"] = "parse_error"
            results.append(entry)
            continue

        imag = [f for f in freqs if f < 0] if isinstance(freqs, list) \
               else [v for v in freqs.values() if v < 0]

        # ------------------------------
        # Exploratory mode
        # ------------------------------
        if mode == "exploratory":
            if len(imag) == 1:
                entry["status"] = "ok"
                entry["passed"] = True
            else:
                entry["status"] = "bad_imag_count"
            results.append(entry)
            continue

        # ------------------------------
        # Confirmatory mode
        # ------------------------------
        try:
            _, ts_mode = extract_ts_mode_from_orca(
                out_path,
                fingerprint=fingerprint
            )
        except Exception:
            entry["status"] = "parse_error"
            results.append(entry)
            continue

        cos = cosine_similarity(ref_mode, ts_mode)
        loc = mode_localization_fraction(ts_mode, fingerprint)

        entry["cosine"] = cos
        entry["localization"] = loc

        if cos >= cos_thresh and loc >= loc_thresh:
            entry["status"] = "ok"
            entry["passed"] = True
        else:
            entry["status"] = "failed"

        results.append(entry)

    return results
