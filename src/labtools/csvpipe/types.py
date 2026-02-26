from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ColumnRule:
    name: str
    path: str                   # dot/bracket path, e.g. "jobs[].id" or "jobs[].parameters.grid"
    type: str = "str"           # str|int|float|bool|list|json
    required: bool = False
    default: Any = None
    source: str = "csv"         # csv|default_only
    fanout: bool = False
    enum: Optional[List[Any]] = None
    transform: Optional[str] = None
    coalesce: Optional[List[str]] = None

@dataclass
class ComputedRule:
    path: str
    expr: str                   # Jinja expression (without {{ }})

@dataclass
class MappingSpec:
    version: str = "1"
    delimiter: str = ";"
    id_pattern: Optional[str] = None
    columns: List[ColumnRule] = None
    computed: List[ComputedRule] = None
