# labtools/workflow/engine.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import jinja2
import yaml

from labtools.orca.restart_overrides import RestartOverrides


# ---------------------------------------------------------------
# Canonical template locations
# ---------------------------------------------------------------
TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "templates"
ORCA_TPL_DIR = TEMPLATE_ROOT / "orca"

CANONICAL_TEMPLATES = {
    "opt": "orca_opt.inp.j2",
    "freq": "orca_freq.inp.j2",
    "sp": "orca_sp.inp.j2",
    "sp_triplet": "orca_sp_triplet.inp.j2",
    "nmr": "orca_nmr.inp.j2",
    "gradient": "orca_grad.inp.j2"
}


# ===============================================================
class WorkflowEngine:
    """
    Resolves workflow stages, loads geometry, and renders ORCA
    input files into stage directories.
    """

    jenv = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            [
                str(TEMPLATE_ROOT),
                str(ORCA_TPL_DIR),
                str(TEMPLATE_ROOT / "sbatch"),
            ]
        ),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def __init__(self, job_config: Dict[str, Any], jobdir: Path) -> None:
        self.job_config = job_config
        self.jobdir = Path(jobdir)

    # ---------------------------------------------------------------
    # Template resolution
    # ---------------------------------------------------------------
    def _resolve_template(self, stage_name: str) -> str:
        if stage_name not in CANONICAL_TEMPLATES:
            raise ValueError(f"Unknown stage '{stage_name}'")

        tpl = CANONICAL_TEMPLATES[stage_name]
        tpl_path = ORCA_TPL_DIR / tpl

        if not tpl_path.is_file():
            raise FileNotFoundError(
                f"Missing template '{tpl}' for stage '{stage_name}'. "
                f"Expected at: {tpl_path}"
            )

        return tpl

    def resolve_stages(self) -> List[Dict[str, str]]:
        pipeline = self.job_config.get("pipeline", ["opt", "freq", "sp"])
        return [{"name": s, "template": self._resolve_template(s)} for s in pipeline]

    # ---------------------------------------------------------------
    # GEOMETRY LOADING
    # ---------------------------------------------------------------
    def load_geometry(self, stage_name: str) -> List[str]:
        """
        Logic:
          • First stage → always YAML geometry
          • Later stages:
                - If prev/final.xyz exists → use it
                - Otherwise → fallback to YAML geometry
        """
        cfg = self.job_config
        stages = self.resolve_stages()

        idx = next(i for i, s in enumerate(stages) if s["name"] == stage_name)

        # -------------------
        # First stage
        # -------------------
        if idx == 0:
            geom = cfg.get("xyz")
            if not geom:
                raise ValueError("job.yaml missing 'xyz' geometry block")
            return geom

        # -------------------
        # Subsequent stages
        # -------------------
        prev_stage = stages[idx - 1]["name"]
        prev_dir = self.jobdir / prev_stage
        final_xyz = prev_dir / "final.xyz"

        # If previous stage finished → use final.xyz
        if final_xyz.exists():
            raw = final_xyz.read_text().splitlines()
            if raw and raw[0].strip().isdigit():
                raw = raw[2:]
            return raw

        # No final.xyz yet → fallback to original YAML geometry
        return cfg.get("xyz")

    # ---------------------------------------------------------------
    # CONTEXT BUILDING
    # ---------------------------------------------------------------
    def build_context(
        self,
        stage: Dict[str, Any],
        overrides: Optional[RestartOverrides] = None,
    ) -> Dict[str, Any]:

        cfg = self.job_config
        stage_name = stage["name"]

        ctx: Dict[str, Any] = {
            "job_id": cfg.get("id"),
            "stage": stage_name,
            "method": cfg.get("method"),
            "basis": cfg.get("basis"),
            "flags": cfg.get("flags", []),
            "charge": cfg.get("charge", 0),
            "mult": cfg.get("mult", 1),
            "geom_lines": self.load_geometry(stage_name),
            "restart_flags": [],
            "scf": {},
            "geom": {},
            "freq": {},
            "nmr": {},
            "cpcm": cfg.get("cpcm"),
            "maxcore_mb": cfg.get("maxcore_mb", 2000),
        }

        # Restart overrides
        if overrides:
            if overrides.global_flags_add:
                ctx["restart_flags"].extend(overrides.global_flags_add)
            if overrides.scf:
                ctx["scf"].update(overrides.scf)
            if overrides.geom:
                ctx["geom"].update(overrides.geom)
            if overrides.freq:
                ctx["freq"].update(overrides.freq)
            if overrides.nmr:
                ctx["nmr"].update(overrides.nmr)

        return ctx

    # ---------------------------------------------------------------
    # TEMPLATE RENDERING
    # ---------------------------------------------------------------
    def render_stage_input(
        self,
        stage: Dict[str, Any],
        stage_dir: Path,
        overrides=None
    ) -> Path:

        tpl = self.jenv.get_template(stage["template"])
        ctx = self.build_context(stage, overrides)
        rendered = tpl.render(**ctx)

        out_path = stage_dir / f"{stage['name']}.inp"
        stage_dir.mkdir(exist_ok=True)
        out_path.write_text(rendered)

        return out_path

    # ---------------------------------------------------------------
    # INITIAL WORKFLOW STATE
    # ---------------------------------------------------------------
    def initialize_state(self):
        from labtools.workflow.state import WorkflowState

        stages = [s["name"] for s in self.resolve_stages()]

        return WorkflowState(
            job_id=self.job_config.get("id"),
            stages=stages,
            stage_index=0,
            attempt=0,
            status="PENDING",
        )
