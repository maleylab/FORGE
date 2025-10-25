from __future__ import annotations
import json, pathlib
from typing import Dict, Any
import pandas as pd

def jsonl_append(path: str, rec: Dict[str, Any]):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def jsonl_to_parquet(jsonl_path: str, parquet_path: str):
    p = pathlib.Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(jsonl_path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
