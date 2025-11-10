from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class OrcaRecord(BaseModel):
    # Schema for parsed ORCA output (single job)
    schema_version: str = Field(default='0.1.0')
    file: str
    sha256: str
    energy_Eh: Optional[float] = None
    HOMO_Eh: Optional[float] = None
    LUMO_Eh: Optional[float] = None
    gap_eV: Optional[float] = None
    parser: str = Field(default='labtools.orca.parse/0.1.0')

class EnergeticSpanResult(BaseModel):
    # Schema for energetic span calculation result
    schema_version: str = Field(default='0.1.0')
    deltaE_kcal_mol: float
    TDTS: str
    TDI: str
    pair: Dict[str, Any]

class Provenance(BaseModel):
    # Schema for lightweight runtime provenance snapshot
    schema_version: str = Field(default='0.1.0')
    timestamp: str
    project: Optional[str] = None
    host: Optional[str] = None
    system: Dict[str, Any]
    vcs: Dict[str, Any]
    hpc: Dict[str, Any]
    packages: Dict[str, Any]
    env_hash: str
