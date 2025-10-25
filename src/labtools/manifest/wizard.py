from __future__ import annotations
import pathlib, yaml
from typing import List, Dict

def init_specs(base_dir: str):
    base = pathlib.Path(base_dir)
    specs = base / '00_specs'
    specs.mkdir(parents=True, exist_ok=True)
    (specs / 'env.yaml').write_text(yaml.safe_dump({
        'cluster': 'YOUR_CLUSTER',
        'account': 'def-yourpi',
        'orca_module': 'orca/6.1',
        'defaults': {'cpus': 8, 'mem': '8G', 'time': '02:00:00'}
    }, sort_keys=False))
    (specs / 'metadata.yaml').write_text(yaml.safe_dump({
        'project': base.name,
        'title': 'Project Title',
        'pi': 'Your Name',
        'grant': 'NSERC DG',
        'keywords': ['ORCA','DFT','automation']
    }, sort_keys=False))
    (specs / 'plan.yaml').write_text(yaml.safe_dump({'version': '1.0', 'jobs': []}, sort_keys=False))
    return str(specs)

def scan_structures(structures_dir: str) -> List[str]:
    base = pathlib.Path(structures_dir)
    return [str(p) for p in sorted(base.rglob('*.xyz'))]

def generate_plan(structs: List[str], jobset: List[str]=None, charge: int=0, mult: int=1) -> Dict:
    if jobset is None:
        jobset = ['optfreq','sp_triplet','nmr']
    jobs = []
    for s in structs:
        stem = pathlib.Path(s).stem
        prev_id = None
        for jt in jobset:
            jid = f'{stem}-{jt}'
            j = {
                'id': jid,
                'type': jt,
                'structure': s,
                'charge': charge,
                'mult': (1 if jt != 'sp_triplet' else 3),
                'template': f'20_calcs/templates/{jt}.inp.j2',
                'parameters': {}
            }
            if prev_id is not None:
                j['depends_on'] = [prev_id]
            jobs.append(j)
            prev_id = jid
    return {'version': '1.0', 'jobs': jobs}

def write_plan_yaml(plan: Dict, out_path: str):
    pathlib.Path(out_path).write_text(yaml.safe_dump(plan, sort_keys=False))
