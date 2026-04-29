from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    """A stable pointer to an Artifact in a run."""

    artifact_id: str
    kind: str
    relpath: str  # relative to run_dir/artifacts


@dataclass
class Artifact:
    """Immutable-ish on-disk object with a manifest.

    Design goals:
    - deterministic directory: run_dir/artifacts/{kind}/{artifact_id}/
    - manifest.json contains metadata + file checksums + parent refs
    - data is JSON-serializable (small)
    """

    kind: str
    data: Dict[str, Any]
    artifact_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = 1
    created_at: str = dataclasses.field(default_factory=_utc_ts)
    parents: List[ArtifactRef] = dataclasses.field(default_factory=list)
    files: List[Dict[str, Any]] = dataclasses.field(default_factory=list)  # [{relpath, sha256, bytes}]

    def to_ref(self, *, relpath: str) -> ArtifactRef:
        return ArtifactRef(artifact_id=self.artifact_id, kind=self.kind, relpath=relpath)

    # -------------------------
    # On-disk layout helpers
    # -------------------------
    def artifact_dir(self, *, run_dir: Path) -> Path:
        return run_dir / "artifacts" / self.kind / self.artifact_id

    def write(
        self,
        *,
        run_dir: Path,
        copy_files: Optional[Sequence[Tuple[Path, str]]] = None,
        allow_exists: bool = False,
    ) -> ArtifactRef:
        """Materialize the artifact to disk and return a ref.

        Parameters
        ----------
        copy_files
            Optional iterable of (src_path, dest_relpath_within_artifact_dir).
            Files are copied and checksummed in manifest.
        allow_exists
            If True, do not error if the artifact directory already exists.
        """

        adir = self.artifact_dir(run_dir=run_dir)
        if adir.exists():
            if not allow_exists:
                raise FileExistsError(f"Artifact directory already exists: {adir}")
        else:
            adir.mkdir(parents=True, exist_ok=False)

        # Copy files
        self.files = []
        if copy_files:
            for src, dest_rel in copy_files:
                src = Path(src)
                if not src.exists():
                    raise FileNotFoundError(str(src))
                dest = adir / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
                self.files.append(
                    {
                        "relpath": dest_rel,
                        "sha256": sha256_file(dest),
                        "bytes": dest.stat().st_size,
                    }
                )

        # Write manifest
        manifest = {
            "artifact": {
                "id": self.artifact_id,
                "kind": self.kind,
                "schema_version": self.schema_version,
                "created_at": self.created_at,
            },
            "parents": [dataclasses.asdict(p) for p in self.parents],
            "data": self.data,
            "files": self.files,
        }
        (adir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        rel = os.path.relpath(adir, run_dir / "artifacts")
        return self.to_ref(relpath=rel)

    @staticmethod
    def load(run_dir: Path, ref: ArtifactRef) -> "Artifact":
        adir = (run_dir / "artifacts" / ref.relpath).resolve()
        man = json.loads((adir / "manifest.json").read_text(encoding="utf-8"))
        art = Artifact(
            kind=man["artifact"]["kind"],
            data=man.get("data") or {},
            artifact_id=man["artifact"]["id"],
            schema_version=man["artifact"].get("schema_version", 1),
            created_at=man["artifact"].get("created_at") or "",
            parents=[ArtifactRef(**p) for p in (man.get("parents") or [])],
            files=man.get("files") or [],
        )
        return art
