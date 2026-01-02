from typing import Any, Dict, List, Tuple
from copy import deepcopy
from itertools import product
from .mapping import deep_get, deep_set_with_append

def gather_axes(doc: Dict[str, Any], axes_paths: List[str]) -> List[Tuple[str, List[Any]]]:
    axes = []
    for p in axes_paths:
        v = deep_get(doc, p, None)
        if v is None:
            continue
        vals = v if isinstance(v, list) else [v]
        axes.append((p, vals))
    return axes

def expand_product(doc: Dict[str, Any], axes: List[Tuple[str, List[Any]]]) -> List[Dict[str, Any]]:
    if not axes:
        return [doc]
    out = []
    for combo in product(*[vals for _, vals in axes]):
        d = deepcopy(doc)
        for (path, _), value in zip(axes, combo):
            # overwrite the path with the selected value
            deep_set_with_append(d, path, value)
        out.append(d)
    return out

def expand_zip(doc: Dict[str, Any], axes: List[Tuple[str, List[Any]]]) -> List[Dict[str, Any]]:
    if not axes:
        return [doc]
    lengths = {len(vals) for _, vals in axes}
    if len(lengths) != 1:
        raise ValueError(f"zip fanout requires equal lengths, got {sorted(lengths)}")
    out = []
    L = next(iter(lengths))
    for i in range(L):
        d = deepcopy(doc)
        for (path, vals) in axes:
            deep_set_with_append(d, path, vals[i])
        out.append(d)
    return out
