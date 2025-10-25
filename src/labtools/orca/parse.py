from __future__ import annotations
import re, hashlib, pathlib
from typing import Dict, Any

ENERGY_RE = re.compile(r"TOTAL SCF ENERGY\s+(-?\d+\.\d+)", re.IGNORECASE)
HOMO_RE = re.compile(r"Alpha\s+occ\.\s+eigenvalues\s+--\s+(.+)", re.I)
LUMO_RE = re.compile(r"Alpha\s+virt\.\s+eigenvalues\s+--\s+(.+)", re.I)

def _hashfile(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_orca_file(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    text = p.read_text(errors="ignore")

    energy = None
    m = ENERGY_RE.search(text)
    if m:
        energy = float(m.group(1))  # Eh

    homo_e = None
    lumo_e = None
    homo_line = HOMO_RE.findall(text)
    virt_line = LUMO_RE.findall(text)

    if homo_line:
        try:
            homo_e = float(homo_line[-1].split()[-1])  # last value on the line
        except Exception:
            pass
    if virt_line:
        try:
            lumo_e = float(virt_line[0].split()[0])  # first value
        except Exception:
            pass

    gap_ev = None
    if homo_e is not None and lumo_e is not None:
        gap_ev = (lumo_e - homo_e) * 27.211386245988

    rec = {
        "schema_version": "0.1.0",
        "file": str(p),
        "sha256": _hashfile(path),
        "energy_Eh": energy,
        "HOMO_Eh": homo_e,
        "LUMO_Eh": lumo_e,
        "gap_eV": gap_ev,
        "parser": "labtools.orca.parse/0.1.0",
    }
    return rec
