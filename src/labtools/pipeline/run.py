from __future__ import annotations

import json
import os
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import ArtifactRef
from .stages import Stage, StageResult, BaseStage


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file_tree(root: Path, *, include_glob: str = "**/*") -> str:
    """Stable hash of a directory tree.

    Hashes file *paths relative to root* and file content digests.
    Ignores directories.
    """

    root = root.expanduser().resolve()
    parts: List[str] = []
    if not root.exists():
        return ""
    for p in sorted(root.glob(include_glob)):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        parts.append(rel)
        parts.append(_sha256_file(p))
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def _canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass
class PipelineRun:
    """Resumable pipeline run with structured audit log.

    Layout:
      run_dir/
        manifest.json
        audit.jsonl
        artifacts/
        workspace/   (scratch / rendered jobs, etc.)
    """

    run_dir: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: str = field(default_factory=_utc_ts)
    pipeline_name: str = "pipeline"
    config: Dict[str, Any] = field(default_factory=dict)
    stages: List[Stage] = field(default_factory=list)

    def init(self, *, allow_exists: bool = False) -> None:
        self.run_dir = self.run_dir.expanduser().resolve()
        if self.run_dir.exists():
            if not allow_exists:
                raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        else:
            self.run_dir.mkdir(parents=True, exist_ok=False)

        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        (self.run_dir / "workspace").mkdir(exist_ok=True)

        self._write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def audit_path(self) -> Path:
        return self.run_dir / "audit.jsonl"

    def _write_manifest(self) -> None:
        obj = {
            "run": {
                "id": self.run_id,
                "created_at": self.created_at,
                "pipeline": self.pipeline_name,
            },
            "config": self.config,
            "stages": [getattr(s, "describe", lambda: {"name": getattr(s, "name", "stage")})() for s in self.stages],
        }
        self.manifest_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def audit(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", _utc_ts())
        record.setdefault("run_id", self.run_id)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _load_completed_stage_names(self) -> List[str]:
        """Deprecated in favor of _load_completed_stage_hashes."""
        return list(self._load_completed_stage_hashes().keys())

    def _load_completed_stage_hashes(self) -> Dict[str, str]:
        """Return mapping: stage_name -> inputs_hash for successful stage_end records."""

        if not self.audit_path.exists():
            return {}
        done: Dict[str, str] = {}
        with self.audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("event") == "stage_end" and obj.get("ok") is True:
                    stage = str(obj.get("stage"))
                    ih = str(obj.get("inputs_hash") or "")
                    # Keep the *latest* successful record for that stage.
                    done[stage] = ih
        return done

    def _compute_stage_inputs_hash(self, stage: Stage, ctx: Dict[str, Any]) -> str:
        """Compute a stable input hash for a stage.

        This is used to decide whether a completed stage can be safely skipped
        when resuming.

        Rules:
          - Always include pipeline config + stage.describe().
          - If the stage provides `fingerprint(context)->dict`, include that.
        """

        sdesc = getattr(stage, "describe", lambda: {"name": getattr(stage, "name", "stage")})()
        fp: Dict[str, Any] = {}
        if hasattr(stage, "fingerprint"):
            try:
                fp = stage.fingerprint(context=ctx)  # type: ignore
            except TypeError:
                # Some user implementations might omit kw-only signature
                fp = stage.fingerprint(ctx)  # type: ignore
        payload = {
            "pipeline": self.pipeline_name,
            "pipeline_config": self.config,
            "stage": getattr(stage, "name", "stage"),
            "stage_desc": sdesc,
            "stage_fingerprint": fp,
        }
        return _sha256_bytes(_canonical_json_bytes(payload))

    def run(self, *, resume: bool = False, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute all stages in order.

        Resume semantics (v1):
          - if resume=True, skip stages that have a successful stage_end record.
        """

        ctx: Dict[str, Any] = dict(context or {})
        ctx.setdefault("run_dir", str(self.run_dir))
        ctx.setdefault("workspace_dir", str(self.run_dir / "workspace"))
        ctx.setdefault("artifacts_dir", str(self.run_dir / "artifacts"))
        ctx.setdefault("run", self)

        completed_hashes = self._load_completed_stage_hashes() if resume else {}

        all_outputs: List[ArtifactRef] = []
        for stage in self.stages:
            sname = getattr(stage, "name", "stage")

            inputs_hash = self._compute_stage_inputs_hash(stage, ctx)
            if resume and (sname in completed_hashes) and (completed_hashes.get(sname) == inputs_hash):
                self.audit({"event": "stage_skip", "stage": sname, "reason": "resume", "inputs_hash": inputs_hash})
                continue

            self.audit({"event": "stage_start", "stage": sname})
            sr: StageResult
            try:
                if hasattr(stage, "_run_with_gates"):
                    # BaseStage helper
                    sr = stage._run_with_gates(context=ctx, fn=lambda: stage.run(context=ctx))  # type: ignore
                else:
                    outs = stage.run(context=ctx)  # type: ignore
                    sr = StageResult(stage=sname, ok=True, outputs=list(outs), gate_results=[], elapsed_s=0.0)
            except Exception as e:
                sr = StageResult(stage=sname, ok=False, outputs=[], gate_results=[], message=str(e), elapsed_s=0.0)

            self.audit(
                {
                    "event": "stage_end",
                    "stage": sname,
                    "ok": sr.ok,
                    "inputs_hash": inputs_hash,
                    "elapsed_s": sr.elapsed_s,
                    "message": sr.message,
                    "outputs": [o.__dict__ for o in sr.outputs],
                    "gates": [gr.__dict__ for gr in sr.gate_results],
                }
            )
            if not sr.ok:
                return {"ok": False, "failed_stage": sname, "message": sr.message, "outputs": [o.__dict__ for o in all_outputs]}

            all_outputs.extend(sr.outputs)

        return {"ok": True, "outputs": [o.__dict__ for o in all_outputs]}

