from __future__ import annotations
from typing import Any, Dict, List, Optional
from .types import ColumnRule, MappingSpec, ComputedRule
import json, re

BRACKET_APPEND = "[]"

def _split_path(path: str) -> List[str]:
    # supports tokens like "jobs[]", "jobs", "parameters.grid"
    return path.split(".")

def _flatten_once(value: Any) -> Any:
    """If value is [list], flatten one level: [[a,b]] -> [a,b]."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value

def _append_or_extend(dst_list: List[Any], value: Any) -> None:
    """Append scalars, extend on list (flatten one level), ignore None."""
    if value is None:
        return
    value = _flatten_once(value)
    if isinstance(value, list):
        for v in value:
            if v is not None:
                dst_list.append(v)
    else:
        dst_list.append(value)

def deep_set_with_append(doc: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set value supporting 'jobs[]' (append semantics) and safe list handling.

    Examples:
      deep_set_with_append(doc, "jobs[].id", "job1")
      deep_set_with_append(doc, "jobs[].depends_on", ["a","b"])      # sets list directly
      deep_set_with_append(doc, "jobs[].depends_on[]", ["a","b"])    # extends list
      deep_set_with_append(doc, "jobs[].parameters.grid", ["G4","G5"])
      deep_set_with_append(doc, "jobs[].parameters.grid[]", ["G4","G5"])
    """
    parts = _split_path(path)
    cur: Any = doc
    for i, tok in enumerate(parts):
        is_last = i == len(parts) - 1
        is_append = tok.endswith(BRACKET_APPEND)
        key = tok[:-2] if is_append else tok

        if is_last:
            if is_append:
                lst = cur.setdefault(key, [])
                _append_or_extend(lst, value)  # flatten one level if needed
            else:
                # direct set (no wrapping). If caller passed list, keep it as is.
                if isinstance(cur, dict):
                    cur[key] = _flatten_once(value)
                else:
                    raise TypeError(f"Cannot set key {key} on non-dict node")
        else:
            if is_append:
                lst = cur.setdefault(key, [])
                # Ensure there is a dict to continue setting deeper fields
                if not lst or not isinstance(lst[-1], dict):
                    lst.append({})
                cur = lst[-1]
            else:
                if not isinstance(cur, dict):
                    raise TypeError(f"Cannot descend into non-dict node at {key}")
                cur = cur.setdefault(key, {})

def deep_get(doc: Dict[str, Any], path: str, default=None):
    cur: Any = doc
    for token in _split_path(path):
        is_append = token.endswith(BRACKET_APPEND)
        key = token[:-2] if is_append else token
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
            if is_append:
                if isinstance(cur, list) and cur:
                    cur = cur[-1]
                else:
                    return default
        else:
            return default
    return cur

def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1","true","t","yes","y","on"}

def cast_value(typ: str, raw: Any):
    if raw is None:
        return None
    if typ == "str":
        return str(raw)
    if typ == "int":
        return int(raw)
    if typ == "float":
        return float(raw)
    if typ == "bool":
        return _to_bool(raw)
    if typ == "json":
        return json.loads(raw) if isinstance(raw, str) else raw
    if typ == "list":
        if isinstance(raw, list):
            return raw
        # Split on ';' and strip; drop empties
        return [s.strip() for s in str(raw).split(";") if s.strip()]
    return raw

def apply_transform(name: Optional[str], v: Any):
    if name is None or v is None:
        return v
    if name == "strip" and isinstance(v, str):
        return v.strip()
    if name == "lower" and isinstance(v, str):
        return v.lower()
    if name == "upper" and isinstance(v, str):
        return v.upper()
    if name == "slug" and isinstance(v, str):
        s = v.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
        return s
    if name == "json" and isinstance(v, str):
        return json.loads(v)
    return v

def normalize_columns(cols: List[dict]) -> List[ColumnRule]:
    out: List[ColumnRule] = []
    for c in cols or []:
        out.append(ColumnRule(
            name=c["name"],
            path=c["path"],
            type=c.get("type","str"),
            required=c.get("required", False),
            default=c.get("default"),
            source=c.get("source","csv"),
            fanout=c.get("fanout", False),
            enum=c.get("enum"),
            transform=c.get("transform"),
            coalesce=c.get("coalesce"),
        ))
    return out

def normalize_computed(cmps: List[dict]) -> List[ComputedRule]:
    return [ComputedRule(path=c["path"], expr=c["expr"]) for c in (cmps or [])]

def build_mapping(spec: dict) -> MappingSpec:
    return MappingSpec(
        version=str(spec.get("version","1")),
        delimiter=spec.get("delimiter",";"),
        id_pattern=spec.get("id_pattern"),
        columns=normalize_columns(spec.get("columns", [])),
        computed=normalize_computed(spec.get("computed", [])),
    )
