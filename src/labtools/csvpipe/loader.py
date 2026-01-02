from __future__ import annotations
from typing import Any, Dict, List, Optional
from copy import deepcopy
import csv as _csv

try:
    from jinja2 import Template
except Exception:
    Template = None

from .types import MappingSpec, ColumnRule
from .mapping import build_mapping, deep_get, deep_set_with_append, cast_value, apply_transform

def _row_cell(row: Dict[str, Any], rule: ColumnRule, delimiter: str):
    present = False
    val = None
    headers = [rule.name] + (rule.coalesce or [])
    for h in headers:
        if h in row:
            present = True
            cell = row[h]
            val = None if cell is None or str(cell).strip() == "" else cell
            break
    if rule.source == "default_only":
        present = True
        val = rule.default
    # fanout split for list columns or rule.fanout
    if val is not None and (rule.fanout or rule.type == "list") and not isinstance(val, list):
        val = [s.strip() for s in str(val).split(delimiter) if s.strip()]
    return val, present

def row_to_job(row: Dict[str, Any], mapping: MappingSpec, defaults: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Build a single job entry doc (shape conforms to one element of jobs[]).

    The mapping must use paths like 'jobs[].id', 'jobs[].type', etc.

    We construct a temporary doc and then extract the last appended 'jobs' element.

    """
    doc: Dict[str, Any] = deepcopy(defaults or {})
    # 1) columns → set values
    for rule in mapping.columns or []:
        val, present = _row_cell(row, rule, mapping.delimiter)
        if not present:
            continue
        if (val is None or val == "") and rule.default is not None:
            val = rule.default
        if val is None or val == "":
            continue
        # cast/transform/enum
        if isinstance(val, list):
            casted = [apply_transform(rule.transform, cast_value(rule.type, v)) for v in val]
        else:
            casted = apply_transform(rule.transform, cast_value(rule.type, val))
        if rule.enum is not None:
            def _ok(x): return x in rule.enum
            if isinstance(casted, list):
                bad = [x for x in casted if not _ok(x)]
                if bad:
                    raise ValueError(f"Row {row.get('_rownum','?')} column {rule.name}: {bad} not in {rule.enum}")
            else:
                if not _ok(casted):
                    raise ValueError(f"Row {row.get('_rownum','?')} column {rule.name}: {casted} not in {rule.enum}")
        deep_set_with_append(doc, rule.path, casted)

    # 2) computed fields
    if mapping.computed and Template is not None:
        ctx = {"row": row, **doc}
        for comp in mapping.computed:
            try:
                val = Template("{{ " + comp.expr + " }}").render(**ctx)
            except Exception:
                continue
            if val not in ("", None, "None"):
                deep_set_with_append(doc, comp.path, val)

    # Extract the last appended jobs[] entry if present; else doc itself
    job = None
    if isinstance(doc.get("jobs"), list) and doc["jobs"]:
        job = doc["jobs"][-1]
    else:
        # If mapping didn't use jobs[].*, the doc itself may already be a single job entry
        job = doc

    # annotate raw row for provenance
    job["_row_raw"] = row
    return job

def read_csv_rows(csv_path) -> List[Dict[str, Any]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader, start=1):
            row["_rownum"] = i
            rows.append(row)
    return rows
