from __future__ import annotations
from typing import List, Dict

def compute_energetic_span(states: List[Dict]) -> Dict:
    \"\"\"Compute energetic span deltaE (kcal/mol).
    Input: list of dicts with keys: label, kind in {'TS','I'}, G (kcal/mol).
    Returns: dict with deltaE_kcal_mol, TDTS, TDI, pair.
    \"\"\"
    ts = [s for s in states if s.get("kind") == "TS"]
    ints = [s for s in states if s.get("kind") in ("I", "Int", "INT")]
    if not ts or not ints:
        raise ValueError("Need at least one TS and one intermediate (I).")
    max_delta = None
    chosen = None
    for t in ts:
        for i in ints:
            delta = float(t["G"]) - float(i["G"])
            if (max_delta is None) or (delta > max_delta):
                max_delta = delta
                chosen = (t, i)
    return {
        "schema_version": "0.1.0",
        "deltaE_kcal_mol": float(max_delta),
        "TDTS": chosen[0]["label"],
        "TDI": chosen[1]["label"],
        "pair": {"TS": chosen[0], "I": chosen[1]},
    }
