from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import pathlib, yaml, re

RANGE_RE = re.compile(r'^\s*(\d+)\s*-\s*(\d+)\s*$')

def _parse_atom_tokens(tokens: List[str]) -> List[int]:
    # Parse tokens like ['0-11', '14', '16-18'] into a sorted unique zero-based list
    idxs = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        m = RANGE_RE.match(t)
        if m:
            a = int(m.group(1)); b = int(m.group(2))
            step = 1 if b >= a else -1
            idxs.extend(list(range(a, b + step, step)))
        else:
            idxs.append(int(t))
    seen = set(); out = []
    for x in idxs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _flatten_atoms_list(atom_fields: List[str], indexing: str, atom_count: Optional[int]) -> List[int]:
    # Flatten list of 'atoms' strings to zero-based ints; convert from 1-based if requested
    raw = _parse_atom_tokens(atom_fields)
    if indexing == '1-based':
        raw = [i - 1 for i in raw]
    return raw

def load_fragments_yaml(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    import yaml as _yaml
    data = _yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError('fragments.yaml must be a mapping at top level')
    data.setdefault('by_structure', {})
    data.setdefault('by_job', {})
    opts = data.setdefault('options', {})
    opts.setdefault('indexing', '0-based')
    opts.setdefault('validate_from_xyz', True)
    opts.setdefault('fail_on_missing', False)
    return data

def _basename_without_ext(structure_path: str) -> str:
    return pathlib.Path(structure_path).stem

def select_fragments(frags: Dict[str, Any], job_id: str, structure_path: str) -> List[Dict[str, Any]]:
    # Return fragment list for a job: by_job[job_id] if present else by_structure[basename]
    by_job = frags.get('by_job', {})
    by_structure = frags.get('by_structure', {})
    if job_id in by_job:
        return by_job[job_id] or []
    base = _basename_without_ext(structure_path)
    return by_structure.get(base, []) or []

def _count_atoms_from_xyz(xyz_path: str) -> int:
    text = pathlib.Path(xyz_path).read_text(errors='ignore').splitlines()
    if not text:
        raise ValueError(f'Empty XYZ: {xyz_path}')
    try:
        n = int(text[0].strip()); return n
    except Exception:
        return max(0, len(text) - 2)

def validate_fragments_for_job(frags: Dict[str, Any], job: Dict[str, Any], xyz_path: str) -> Tuple[bool, List[str], List[List[int]]]:
    # Validate and normalize fragments for a given job
    frag_list = select_fragments(frags, job.get('id',''), xyz_path)
    errors = []
    if not frag_list:
        if frags.get('options', {}).get('fail_on_missing', False):
            return False, [f'No fragments found for job {job.get('id')} ({xyz_path})'], []
        return True, [], []
    atom_count = _count_atoms_from_xyz(xyz_path)
    indexing = frags.get('options', {}).get('indexing', '0-based')
    per_frag = []
    for entry in frag_list:
        atoms_field = entry.get('atoms', [])
        if not isinstance(atoms_field, list) or not atoms_field:
            errors.append(f"Fragment '{entry.get('name','unnamed')}' has empty or invalid 'atoms' list")
            continue
        idxs = _flatten_atoms_list([str(s) for s in atoms_field], indexing, atom_count)
        bad = [i for i in idxs if i < 0 or i >= atom_count]
        if bad:
            errors.append(f"Fragment '{entry.get('name','unnamed')}' has out-of-range indices {bad} for atom_count={atom_count}")
        per_frag.append(idxs)
    ok = len(errors) == 0
    return ok, errors, per_frag

def build_fragment_index_list(per_fragment_atoms: List[List[int]], atom_count: int) -> List[int]:
    # Produce a list of length atom_count with fragment numbers starting at 1; 0 if unassigned
    mapping = [0] * atom_count
    for frag_num, atoms in enumerate(per_fragment_atoms, start=1):
        for i in atoms:
            if 0 <= i < atom_count:
                mapping[i] = frag_num
    return mapping
