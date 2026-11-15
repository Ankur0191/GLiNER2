import json
from pathlib import Path

import pytest

from rivalmap_ie.training.prepare_dataset import load_jsonl, validate_rows


def test_load_and_validate_native_jsonl(tmp_path):
    source = tmp_path / "gold.jsonl"
    source.write_text(json.dumps({
        "input": "Bureau Veritas inspected the plant.",
        "output": {"entities": {"organization": ["Bureau Veritas"], "asset": ["plant"]}}
    }) + "\n", encoding="utf-8")
    rows = load_jsonl(source)
    validate_rows(rows, source)


def test_invalid_entity_span_fails(tmp_path):
    source = tmp_path / "gold.jsonl"
    source.write_text(json.dumps({
        "input": "Bureau Veritas inspected the plant.",
        "output": {"entities": {"organization": ["Fake Company"]}}
    }) + "\n", encoding="utf-8")
    rows = load_jsonl(source)
    with pytest.raises(ValueError):
        validate_rows(rows, source)
