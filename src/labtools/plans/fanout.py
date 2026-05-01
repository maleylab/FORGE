from __future__ import annotations

import itertools
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from labtools.csvpipe.mapping import deep_get, deep_set_with_append


def slugify_value(value: Any) -> str:
    """Return a stable filesystem/job-id friendly representation of a fanout value."""
    if value is None:
        return "none"
    s = str(value).strip()
    if not s:
        return "empty"
    s = s.replace("/", "-")
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-_.")
    return s or "value"


def normalize_axis_path(path: str) -> str:
    """Normalize mapping-style paths for a single extracted job document.

    CSV mappings often use paths like ``jobs[].parameters.method`` while
    ``row_to_job`` returns the extracted job itself. For expansion we therefore
    need ``parameters.method``.
    """
    p = str(path or "").strip()
    for prefix in ("jobs[].", "jobs."):
        if p.startswith(prefix):
            return p[len(prefix):]
    if p == "jobs[]" or p == "jobs":
        return ""
    return p


def axis_key(path: str) -> str:
    p = normalize_axis_path(path)
    if not p:
        return "value"
    return p.split(".")[-1].replace("[]", "")


def _unwrap_axis_value(value: Any) -> Any:
    """Unwrap accidental single-item list wrappers produced by CSV/list coercion."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _axis_values(value: Any, *, delimiter: str = ";") -> List[Any]:
    """Normalize a raw fanout-axis value to a flat list of scalar values.

    Handles all common upstream shapes:
      - ["B3LYP", "PBE0"]
      - "B3LYP;PBE0"
      - [["B3LYP"], ["PBE0"]]
      - ["B3LYP;PBE0"]
    """
    if value is None:
        return []

    raw_items = value if isinstance(value, list) else [value]
    out: List[Any] = []

    for item in raw_items:
        item = _unwrap_axis_value(item)
        if item in (None, ""):
            continue

        if isinstance(item, str) and delimiter in item:
            out.extend([p.strip() for p in item.split(delimiter) if p.strip()])
        else:
            out.append(item)

    return [_unwrap_axis_value(v) for v in out if v not in (None, "")]


def gather_fanout_axes(job: Dict[str, Any], axis_paths: Sequence[str]) -> List[Tuple[str, List[Any]]]:
    """Collect fanout axes from a single job document.

    Scalars are treated as length-one axes. Semicolon-containing strings are
    split defensively so fanout still works even if upstream mapping leaves a
    fanout column as a raw string.
    """
    axes: List[Tuple[str, List[Any]]] = []
    seen: set[str] = set()
    for raw_path in axis_paths or []:
        path = normalize_axis_path(raw_path)
        if not path or path in seen:
            continue
        seen.add(path)
        value = deep_get(job, path, None)
        values = _axis_values(value)
        if values:
            axes.append((path, values))
    return axes


def _fanout_alias_config(fanout_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize fanout alias config.

    Supported YAML:

      fanout:
        aliases:
          basis:
            prefix: BS

    Also accepts shorthand:

      fanout:
        aliases:
          basis: BS
    """
    if not isinstance(fanout_cfg, Mapping):
        return {}
    raw = fanout_cfg.get("aliases") or {}
    if not isinstance(raw, Mapping):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for key, spec in raw.items():
        k = str(key).strip()
        if not k:
            continue
        if isinstance(spec, Mapping):
            prefix = str(spec.get("prefix") or k.upper()).strip()
            start = int(spec.get("start") or 1)
        else:
            prefix = str(spec or k.upper()).strip()
            start = 1
        out[k] = {"prefix": prefix, "start": start}
    return out


def collect_fanout_aliases(
    jobs: Sequence[Dict[str, Any]],
    *,
    axis_paths: Sequence[str],
    fanout_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """Build stable alias maps for selected fanout axes across all jobs.

    Returns maps in alias -> real value form, e.g.
      {"basis": {"BS1": "6-31G(d,p)", "BS2": "def2-SVP"}}
    """
    alias_cfg = _fanout_alias_config(fanout_cfg)
    if not alias_cfg:
        return {}

    # Map axis key -> normalized path(s), allowing either "basis" or "parameters.basis".
    key_to_paths: Dict[str, List[str]] = {}
    for p in axis_paths or []:
        norm = normalize_axis_path(p)
        if not norm:
            continue
        key_to_paths.setdefault(axis_key(norm), []).append(norm)

    alias_maps: Dict[str, Dict[str, str]] = {}

    for key, spec in alias_cfg.items():
        paths = key_to_paths.get(key, [key])
        seen_values: List[str] = []
        seen_set: set[str] = set()

        for job in jobs:
            for path in paths:
                for value in _axis_values(deep_get(job, path, None)):
                    sval = str(value)
                    if sval not in seen_set:
                        seen_set.add(sval)
                        seen_values.append(sval)

        prefix = str(spec.get("prefix") or key.upper())
        start = int(spec.get("start") or 1)
        alias_maps[key] = {
            f"{prefix}{i + start}": value for i, value in enumerate(seen_values)
        }

    return alias_maps


def _alias_for_value(alias_maps: Mapping[str, Mapping[str, str]], key: str, value: Any) -> Optional[str]:
    real = str(value)
    amap = alias_maps.get(key) or {}
    for alias, mapped in amap.items():
        if str(mapped) == real:
            return str(alias)
    return None


def _format_id(
    base_id: str,
    selected: Mapping[str, Any],
    template: str | None,
    *,
    alias_maps: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> str:
    keys: Dict[str, Any] = {"id": base_id, "base_id": base_id}
    suffix_parts: List[str] = []
    alias_maps = alias_maps or {}

    for path, value in selected.items():
        key = axis_key(path)
        slug = slugify_value(value)

        alias = _alias_for_value(alias_maps, key, value)
        alias_slug = slugify_value(alias) if alias else slug

        keys[key] = value
        keys[f"{key}_slug"] = slug
        keys[f"{key}_alias"] = alias or slug
        keys[f"{key}_alias_slug"] = alias_slug

        flat = normalize_axis_path(path).replace(".", "_").replace("[]", "")
        keys[flat] = value
        keys[f"{flat}_slug"] = slug
        keys[f"{flat}_alias"] = alias or slug
        keys[f"{flat}_alias_slug"] = alias_slug

        suffix_parts.append(alias_slug if alias else slug)

    if template:
        try:
            return str(template).format(**keys)
        except Exception as exc:
            raise ValueError(f"Could not render fanout id template '{template}': {exc}") from exc

    if not suffix_parts:
        return base_id
    return base_id + "__" + "__".join(suffix_parts)


def expand_job_fanout(
    job: Dict[str, Any],
    *,
    axis_paths: Sequence[str],
    mode: str = "product",
    id_field: str = "id",
    id_template: str | None = None,
    alias_maps: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Expand one job document over fanout axes."""
    axes = gather_fanout_axes(job, axis_paths)
    if not axes:
        return [deepcopy(job)]

    mode_norm = str(mode or "product").strip().lower()
    if mode_norm not in {"product", "zip"}:
        raise ValueError("fanout mode must be 'product' or 'zip'")

    combos: Iterable[Tuple[Any, ...]]
    if mode_norm == "zip":
        lengths = {len(vals) for _, vals in axes}
        if len(lengths) != 1:
            raise ValueError(f"zip fanout requires equal lengths, got {sorted(lengths)}")
        combos = zip(*[vals for _, vals in axes])
    else:
        combos = itertools.product(*[vals for _, vals in axes])

    base_id = str(job.get(id_field) or job.get("id") or "job")
    out: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    for combo in combos:
        d = deepcopy(job)
        selected: Dict[str, Any] = {}
        selected_aliases: Dict[str, str] = {}

        for (path, _), value in zip(axes, combo):
            value = _unwrap_axis_value(value)
            deep_set_with_append(d, path, value)
            selected[path] = value

            key = axis_key(path)
            alias = _alias_for_value(alias_maps or {}, key, value)
            if alias:
                selected_aliases[key] = alias

        new_id = _format_id(base_id, selected, id_template, alias_maps=alias_maps)
        if new_id in used_ids:
            n = 2
            candidate = f"{new_id}__{n}"
            while candidate in used_ids:
                n += 1
                candidate = f"{new_id}__{n}"
            new_id = candidate

        used_ids.add(new_id)
        d[id_field] = new_id
        d["id"] = new_id
        d["_fanout"] = {
            "mode": mode_norm,
            "axes": {path: selected[path] for path in selected},
            "aliases": selected_aliases,
            "base_id": base_id,
        }
        out.append(d)

    return out


def axis_paths_from_mapping(mapping: Any, mapping_spec: Dict[str, Any]) -> List[str]:
    """Return fanout axes from explicit spec and ``ColumnRule.fanout`` flags."""
    paths: List[str] = []

    fanout_cfg = mapping_spec.get("fanout") or {}
    if not isinstance(fanout_cfg, dict):
        fanout_cfg = {}

    for p in (fanout_cfg.get("axes") or mapping_spec.get("axes") or []):
        if p:
            paths.append(str(p))

    for rule in getattr(mapping, "columns", []) or []:
        if getattr(rule, "fanout", False):
            paths.append(str(getattr(rule, "path")))

    out: List[str] = []
    seen: set[str] = set()
    for p in paths:
        norm = normalize_axis_path(p)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out
