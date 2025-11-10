# src/labtools/tsgen/collect_imag.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re

_VIB_START = re.compile(r'^\s*-+\s*$', re.MULTILINE)  # for safety if needed

def _slice_freq_block(txt: str) -> str:
    """Return the substring between 'VIBRATIONAL FREQUENCIES' and 'NORMAL MODES'."""
    # Case-insensitive, allow extra dashes/spaces
    m_start = re.search(r'^\s*-+\s*\n\s*VIBRATIONAL\s+FREQUENCIES\s*\n\s*-+\s*$',
                        txt, re.IGNORECASE | re.MULTILINE)
    if not m_start:
        return ""
    # Find the start position just after the header block
    start_pos = m_start.end()

    # Find the first 'NORMAL MODES' header after start
    m_end = re.search(r'^\s*-+\s*\n\s*NORMAL\s+MODES\s*\n\s*-+\s*$',
                      txt[start_pos:], re.IGNORECASE | re.MULTILINE)
    end_pos = start_pos + m_end.start() if m_end else len(txt)
    return txt[start_pos:end_pos]

def parse_imag_from_orca_output(path: Path) -> List[float]:
    """
    Return ONLY imaginary frequencies (negative cm^-1) from ORCA output.
    We strictly parse lines in the VIBRATIONAL FREQUENCIES → NORMAL MODES block of the form:
        '    5:    -123.45 cm**-1'
    """
    txt = path.read_text(errors="ignore")

    block = _slice_freq_block(txt)
    if not block:
        return []

    # Match lines like: "   5:    -123.45 cm**-1" (index, colon, value, cm**-1)
    pat = re.compile(
        r'^\s*\d+\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*cm\*\*\-1\s*$',
        re.MULTILINE
    )

    vals = []
    for m in pat.finditer(block):
        v = float(m.group(1))
        if v < 0.0:
            vals.append(v)
    return vals



