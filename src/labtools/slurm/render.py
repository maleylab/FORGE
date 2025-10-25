from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def render_template(template_path: Path, out_path: Path, params: dict) -> None:
    template_path = Path(template_path)

    # repo_root = .../lab-tools
    repo_root = Path(__file__).resolve().parents[3]
    strict_templates = repo_root / "templates" / "orca"   # /home/smaley/lab-tools/templates/orca

    if template_path.is_file():
        search_dirs = [template_path.parent]              # allow explicit absolute/relative file path
        template_name = template_path.name
    else:
        search_dirs = [strict_templates]                  # ONLY this directory
        template_name = template_path.name                # e.g., "orca_optfreq.inp.j2"

    env = Environment(loader=FileSystemLoader([str(d) for d in search_dirs]))
    tpl = env.get_template(template_name)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tpl.render(**params))
