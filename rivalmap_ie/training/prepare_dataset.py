"""Validate and split reviewed GLiNER2 JSONL annotations by article.

Each JSONL input record follows GLiNER2 native format. This script never
creates synthetic labels or treats model predictions as gold annotations.
"""
from __future__ import annotations

import argparse
import hashlib
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
    allowed_tasks = {"entities", "classifications", "json_structures", "relations"}
    for i, row in enumerate(rows, 1):
        output = row["output"]
        if not allowed_tasks.intersection(output):
            raise ValueError(f"{path}: record {i} has no supervised task")
        entities = output.get("entities", {})
        if not isinstance(entities, dict):
            raise ValueError(f"{path}: record {i}: entities must be an object")
        for label, mentions in entities.items():
            if not isinstance(mentions, list):
                raise ValueError(f"{path}: record {i}: entity label {label!r} must map to a list")
            for mention in mentions:
                if isinstance(mention, dict):
                    value = mention.get("text")
                    start, end = mention.get("start"), mention.get("end")
                    if not isinstance(value, str) or not isinstance(start, int) or not isinstance(end, int):
                        raise ValueError(f"{path}: record {i}: invalid span annotation for {label!r}")
                    if start < 0 or end <= start or row["input"][start:end] != value:
                        raise ValueError(
                            f"{path}: record {i}: span {value!r} does not match input offsets"
                        )
                else:
                    value = mention
                    if not isinstance(value, str) or value.casefold() not in row["input"].casefold():
                        raise ValueError(
                            f"{path}: record {i}: entity {value!r} is not present in input text"
                        )


def _record_group(row: dict[str, Any]) -> str:
    """Group duplicate/source-linked article records to prevent split leakage."""
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("source_url", "canonical_url", "article_id", "document_id"):
            value = metadata.get(key)
            if value:
                return str(value).strip().casefold()
    return hashlib.sha256(row["input"].strip().casefold().encode("utf-8")).hexdigest()


def split_rows(rows: list[dict[str, Any]], validation_ratio: float, seed: int):
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_record_group(row), []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two unique article groups are required to avoid train/validation leakage.")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(rows) * validation_ratio))
    validation_keys, validation_count = [], 0
    for key in keys:
        if validation_count >= target and validation_keys:
            break
        validation_keys.append(key)
        validation_count += len(groups[key])
    validation_set = set(validation_keys)
    train = [row for key, values in groups.items() if key not in validation_set for row in values]
    validation = [row for key, values in groups.items() if key in validation_set for row in values]
    if not train or not validation:
        raise ValueError("Could not create non-empty train/validation sets.")
    return train, validation


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
    train, validation = split_rows(rows, args.validation_ratio, args.seed)

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
