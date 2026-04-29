"""Pipeline kernel (luxury v1).

This package provides:
- Artifact: immutable-ish, on-disk objects with manifests and provenance pointers
- Gate: structured QC checks
- Stage: typed transforms from artifacts to artifacts
- PipelineRun: resumable orchestration with an audit log
"""

from .artifacts import Artifact, ArtifactRef
from .gates import Gate, GateResult, GateSeverity
from .stages import Stage, StageResult
from .run import PipelineRun
