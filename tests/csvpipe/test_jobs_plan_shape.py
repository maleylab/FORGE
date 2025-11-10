import yaml
from labtools.csvpipe.mapping import build_mapping
from labtools.csvpipe.loader import row_to_job
from labtools.csvpipe.emit import emit_combined_plan

def test_job_entry_and_combined_plan(tmp_path):
    mapping_yaml = '''
version: "1"
delimiter: ";"
columns:
  - { name: id,        path: jobs[].id,        type: str }
  - { name: type,      path: jobs[].type,      type: str, enum: ["optfreq","sp","sp_triplet","nmr"] }
  - { name: structure, path: jobs[].structure, type: str }
  - { name: template,  path: jobs[].template,  type: str }
  - { name: grid,      path: jobs[].parameters.grid, type: list, fanout: true }
'''
    m = build_mapping(yaml.safe_load(mapping_yaml))
    row = {"id":"ligA__M06","type":"optfreq","structure":"ligA.xyz","template":"orca/optfreq.inp.j2","grid":"Grid4;Grid5","_rownum":1}
    job = row_to_job(row, m, defaults=None)
    assert job["id"] == "ligA__M06"
    assert job["parameters"]["grid"] == ["Grid4","Grid5"]
    plan_path = tmp_path/"plan.yaml"
    emit_combined_plan([job], plan_path, version="1")
    plan = yaml.safe_load(plan_path.read_text())
    assert plan["version"] == "1"
    assert isinstance(plan["jobs"], list) and plan["jobs"][0]["id"] == "ligA__M06"
