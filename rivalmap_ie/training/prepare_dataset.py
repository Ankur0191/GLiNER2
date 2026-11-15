"""Validate and split reviewed GLiNER2 JSONL annotations by article.

Each JSONL input record must follow GLiNER2's native format:
{"input": "...", "output": {"entities": {...}, ...}}

Train/eval split is deterministic and performed by article record, not sentence,
to reduce leakage. This tool does not create synthetic labels or pretend that
unreviewed model predictions are gold annotations.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if "input" not in row or "output" not in row:
            raise ValueError(f"{path}:{line_no}: expected input and output keys")
        if not isinstance(row["input"], str) or not isinstance(row["output"], dict):
            raise ValueError(f"{path}:{line_no}: input must be text and output must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def validate_rows(rows: list[dict[str, Any]], path: Path) -> None:
    for i, row in enumerate(rows, 1):
        output = row["output"]
        if not any(k in output for k in ("entities", "classifications", "json_structures", "relations")):
            raise ValueError(f"{path}: record {i} has no supervised task")
        for label, mentions in output.get("entities", {}).items():
            if not isinstance(mentions, list):
                raise ValueError(f"{path}: record {i}: entity label {label!r} must map to a list")
            for mention in mentions:
                value = mention.get("text") if isinstance(mention, dict) else mention
                if not isinstance(value, str) or value.lower() not in row["input"].lower():
                    raise ValueError(
                        f"{path}: record {i}: entity {value!r} is not present in input text"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.05 <= args.validation_ratio <= 0.4:
        parser.error("--validation-ratio must be between 0.05 and 0.4")

    rows = load_jsonl(args.input)
    validate_rows(rows, args.input)
    random.Random(args.seed).shuffle(rows)
    split_at = max(1, min(len(rows) - 1, round(len(rows) * (1 - args.validation_ratio))))
    train, validation = rows[:split_at], rows[split_at:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("validation.jsonl", validation)):
        target = args.output_dir / name
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data),
            encoding="utf-8",
        )
        print(f"{target}: {len(data)} records")


if __name__ == "__main__":
    main()
