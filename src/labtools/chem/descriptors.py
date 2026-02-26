from __future__ import annotations

EV_PER_EH = 27.211386245988

def homo_lumo_gap(homo, lumo):
    """
    Safe HOMO–LUMO gap calculator.
    Returns None if either value is missing.
    """
    try:
        if homo is None or lumo is None:
            return None
        return abs(lumo - homo)
    except Exception:
        return None