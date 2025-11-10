from __future__ import annotations

EV_PER_EH = 27.211386245988

def homo_lumo_gap(homo: float, lumo: float) -> float:
    '''Compute HOMO-LUMO gap in eV. If inputs look like eV (|num|<50), assume eV; else Eh→eV.'''
    dh = abs(homo); dl = abs(lumo)
    if dh < 50 and dl < 50:
        return lumo - homo
    return (lumo - homo) * EV_PER_EH
