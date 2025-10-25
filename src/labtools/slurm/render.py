from __future__ import annotations
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any
import pathlib

def render_template(inp: str, out: str, params: Dict[str, Any]):
    inp_p = pathlib.Path(inp)
    env = Environment(loader=FileSystemLoader(str(inp_p.parent)))
    tmpl = env.get_template(inp_p.name)
    rendered = tmpl.render(**params)
    out_p = pathlib.Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(rendered)
