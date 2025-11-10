# lab-tools

General-purpose, reusable tools for our lab: ORCA parsing, SLURM/Jinja helpers, JSONL/Parquet IO, and a small CLI.

Install (editable):
```bash
pip install -e .
```

CLI:
```bash
labtools --help
```

Highlights
- `labtools.orca.parse`: fast regex parser for ORCA 6.x outputs (E, HOMO/LUMO, gap, walltime)
- `labtools.data.io`: JSONL append + Parquet rollups
- `labtools.slurm.render`: Jinja2-based sbatch/input rendering
- `labtools.chem.descriptors`: simple helpers (e.g., HOMO-LUMO gap)
- `labtools.plots.energetic_span`: quick plot helper (matplotlib)

Conventions
- JSON records are flat, friendly for JSONL/Parquet
- CLI favors stdin/stdout for piping
- Always include `schema_version` in emitted records

## New: provenance + energetic span + templates

- `labtools prov-snapshot` -> JSON with env hash, module list, git SHA, SLURM env.
- `labtools es` -> compute energetic span from a JSON list of states.
- Jinja2 templates:
  - `templates/sbatch/orca_job.sbatch.j2`
  - `templates/orca/sp.inp.j2`

### Examples

Provenance:
```bash
labtools prov-snapshot --project proj-iedda-bond-timing-2025 --out prov.json
```

Energetic span:
```json
// states.json
[
  {"label":"I0","kind":"I","G":0.0},
  {"label":"TS1","kind":"TS","G":18.2},
  {"label":"I1","kind":"I","G":-2.3},
  {"label":"TS2","kind":"TS","G":16.1}
]
```
```bash
labtools es states.json --out span.json
```

Render sbatch:
```bash
labtools render templates/sbatch/orca_job.sbatch.j2 out.sbatch job_name=test account=def-yourpi cpus=8 mem=8G input_path=/path/to/input.inp
```

## Schemas (Pydantic)

Programmatic validation is available in `labtools.schemas`:
- `OrcaRecord`
- `EnergeticSpanResult`
- `Provenance`

Example:
```python
from labtools.schemas import OrcaRecord
rec = OrcaRecord(**some_dict)
json_str = rec.model_dump_json(indent=2)
```

Added fragments schema and CLI: fragments-validate, fragments-map.
