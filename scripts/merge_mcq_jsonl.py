#!/usr/bin/env python3
"""Merge MCQ JSONL files while making duplicate record IDs unique.

The first occurrence of an ID is preserved.  Each later occurrence receives
the next available numeric ID with the same prefix and zero-padding, e.g.
``PSY-000001`` -> ``PSY-000002``.  A collision with an already assigned ID is
advanced repeatedly until an unused ID is found.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path


ID_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


def next_available_id(record_id: str, used_ids: set[str]) -> str:
    """Return the first unused ID at or above ``record_id``'s numeric suffix."""
    match = ID_PATTERN.fullmatch(record_id)
    if not match:
        raise ValueError(f"Record ID must end in digits: {record_id!r}")
    prefix = match.group("prefix")
    width = len(match.group("number"))
    candidate = int(match.group("number"))
    while True:
        candidate_id = f"{prefix}{candidate:0{width}d}"
        if candidate_id not in used_ids:
            return candidate_id
        candidate += 1


def merge_jsonl(inputs: Iterable[Path], output: Path) -> tuple[int, int]:
    """Write merged records and return ``(records, reassigned_ids)``."""
    used_ids: set[str] = set()
    records = 0
    reassigned = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as destination:
        for input_path in inputs:
            with input_path.open(encoding="utf-8") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {input_path}:{line_number}: {exc.msg}"
                        ) from exc
                    record_id = record.get("id")
                    if not isinstance(record_id, str) or not record_id:
                        raise ValueError(
                            f"Missing/non-string id in {input_path}:{line_number}"
                        )

                    assigned_id = next_available_id(record_id, used_ids)
                    if assigned_id != record_id:
                        record["id"] = assigned_id
                        reassigned += 1
                    used_ids.add(assigned_id)
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records += 1
    return records, reassigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    args = parser.parse_args()

    inputs = sorted(path.resolve() for path in args.inputs)
    output = args.output.resolve()
    if output in inputs:
        raise ValueError("Output path must not also be an input path")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input JSONL file(s) not found: " + ", ".join(missing))

    records, reassigned = merge_jsonl(inputs, output)
    print(f"Merged {records} records into {output}; reassigned {reassigned} duplicate IDs.")


if __name__ == "__main__":
    main()
