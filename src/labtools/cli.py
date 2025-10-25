from typer import Typer, Option
from rich import print
from .orca.parse import parse_orca_file
from .data.io import jsonl_append, jsonl_to_parquet
from .chem.descriptors import homo_lumo_gap
from .slurm.render import render_template

app = Typer(help="General lab tools CLI")

@app.command()
def orca_parse(path: str, out: str = Option(None, help="JSONL output file")):
    """Parse an ORCA .out into a JSON record; write to JSONL if --out given, else print."""
    rec = parse_orca_file(path)
    if out:
        jsonl_append(out, rec)
        print(f"[green]Wrote[/green] {out}")
    else:
        import json; print(json.dumps(rec, indent=2))

@app.command()
def gap(homo: float, lumo: float):
    """Compute HOMO-LUMO gap in eV from orbital energies (in Eh or eV guessed)."""
    g = homo_lumo_gap(homo, lumo)
    print(f"gap_eV={g:.6f}")

@app.command()
def render(inp: str, out: str, **params):
    """Render a Jinja2 template with key=val overrides: labtools render tmpl.j2 sbatch.sbatch cpus=8 mem='8G'"""
    render_template(inp, out, params)
    print(f"[green]Rendered[/green] {out}")

@app.command()
def jsonl2parquet(jsonl: str, parquet: str):
    jsonl_to_parquet(jsonl, parquet)
    print(f"[green]Wrote[/green] {parquet}")


from .prov.snapshot import provenance_snapshot
from .chem.energetic_span import compute_energetic_span

@app.command("prov-snapshot")
def prov_snapshot(project: str = Option(None, help="Optional project name"), out: str = Option(None, help="Write JSON to file")):
    \"\"\"Capture a lightweight provenance snapshot (env hash, module list, git SHA).\"\"\"
    snap = provenance_snapshot(project=project)
    if out:
        import json
        with open(out, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
        print(f"[green]Wrote[/green] {out}")
    else:
        import json; print(json.dumps(snap, indent=2))

@app.command("es")
def energetic_span(states_json: str, out: str = Option(None, help="Write JSON result to file")):
    \"\"\"Compute energetic span from a JSON list of states: [{\"label\":\"TS1\",\"kind\":\"TS\",\"G\":25.0}, ...].\"\"\"
    import json
    with open(states_json, "r", encoding="utf-8") as f:
        states = json.load(f)
    res = compute_energetic_span(states)
    if out:
        with open(out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2)
        print(f"[green]Wrote[/green] {out}")
    else:
        print(json.dumps(res, indent=2))

from .manifest.wizard import init_specs, scan_structures, generate_plan, write_plan_yaml

@app.command('manifest-init')
def manifest_init(base_dir: str = '.', echo: bool = True):
    p = init_specs(base_dir)
    if echo:
        print(f'[green]Created[/green] {p}')

@app.command('manifest-generate')
def manifest_generate(structures_dir: str = '10_structures/structures',
                      out: str = '00_specs/plan.yaml',
                      jobset: str = 'optfreq,sp_triplet,nmr',
                      charge: int = 0, mult: int = 1):
    structs = scan_structures(structures_dir)
    js = [x.strip() for x in jobset.split(',') if x.strip()]
    plan = generate_plan(structs, jobset=js, charge=charge, mult=mult)
    write_plan_yaml(plan, out)
    print(f'[green]Wrote[/green] {out} with {len(plan['jobs'])} jobs')

@app.command('manifest-validate')
def manifest_validate(plan_path: str = '00_specs/plan.yaml', schema_path: str = 'schemas/plan.schema.json'):
    import yaml, json, jsonschema, pathlib
    data = yaml.safe_load(pathlib.Path(plan_path).read_text())
    schema = json.loads(pathlib.Path(schema_path).read_text())
    jsonschema.validate(data, schema)
    print('[green]OK[/green] plan.yaml matches schema')

from .fragments.manager import load_fragments_yaml, validate_fragments_for_job, build_fragment_index_list
@app.command('fragments-validate')
def fragments_validate(plan: str = '00_specs/plan.yaml', frags: str = '00_specs/fragments.yaml', structures_dir: str = '10_structures/structures'):
    import yaml, pathlib
    plan_data = yaml.safe_load(pathlib.Path(plan).read_text())
    fr = load_fragments_yaml(frags)
    jobs = plan_data.get('jobs', [])
    errors = []
    total = 0
    from .fragments.manager import _count_atoms_from_xyz
    for j in jobs:
        struct_path = j.get('structure')
        xyz_path = pathlib.Path(structures_dir) / pathlib.Path(struct_path).name if struct_path and not pathlib.Path(struct_path).exists() else pathlib.Path(struct_path)
        if not xyz_path.exists():
            errors.append(f'missing XYZ for job {j.get("id")}: {xyz_path}')
            continue
        ok, errs, perfrag = validate_fragments_for_job(fr, j, str(xyz_path))
        total += 1
        if not ok:
            errors.extend([f'{j.get("id")}: {e}' for e in errs])
    if errors:
        for e in errors:
            print(f'[red]{e}[/red]')
        raise SystemExit(1)
    print(f'[green]OK[/green] fragments validated for {total} jobs')

@app.command('fragments-map')
def fragments_map(job_id: str, plan: str = '00_specs/plan.yaml', frags: str = '00_specs/fragments.yaml', structures_dir: str = '10_structures/structures', out: str = 'fragment_map.json'):
    import yaml, pathlib, json
    plan_data = yaml.safe_load(pathlib.Path(plan).read_text())
    j = next((x for x in plan_data.get('jobs', []) if x.get('id') == job_id), None)
    if not j:
        raise SystemExit(f'no such job: {job_id}')
    fr = load_fragments_yaml(frags)
    struct_path = j.get('structure')
    xyz_path = pathlib.Path(structures_dir) / pathlib.Path(struct_path).name if struct_path and not pathlib.Path(struct_path).exists() else pathlib.Path(struct_path)
    from .fragments.manager import _count_atoms_from_xyz, validate_fragments_for_job
    n = _count_atoms_from_xyz(str(xyz_path))
    ok, errs, perfrag = validate_fragments_for_job(fr, j, str(xyz_path))
    if not ok:
        for e in errs:
            print(f'[red]{e}[/red]')
        raise SystemExit(1)
    index_list = build_fragment_index_list(perfrag, n)
    payload = {'job_id': job_id, 'structure': str(xyz_path), 'atom_count': n, 'fragments': perfrag, 'index_list': index_list}
    pathlib.Path(out).write_text(json.dumps(payload, indent=2))
    print(f'[green]Wrote[/green] {out}')
