"""Append-only JSONL persistence for periodic pipeline checkpoints."""

import json
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: str, records: Iterable[dict[str, Any]]) -> int:
    records = list(records)
    if not records:
        return 0
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
