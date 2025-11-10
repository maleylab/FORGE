from typing import Any, Dict
import json

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None

def _load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_doc(doc: Dict[str, Any], schema_path: str) -> None:
    if jsonschema is None:
        return
    schema = _load_schema(schema_path)
    jsonschema.validate(instance=doc, schema=schema)
