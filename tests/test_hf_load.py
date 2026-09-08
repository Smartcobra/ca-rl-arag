#!/usr/bin/env python3
"""Tests for Hotpot JSON → Hub schema conversion. No network."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hf_load import hotpot_official_row_to_hf
from src.data.loaders import hotpot_to_examples


def test_official_json_row_maps_to_hf_schema() -> None:
    raw = {
        "_id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
        "context": [
            ["Scott Derrickson", ["Scott Derrickson is an American director."]],
            ["Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker."]],
        ],
    }
    row = hotpot_official_row_to_hf(raw)
    assert row["id"] == "5a8b57f25542995d1e6f1371"
    assert row["context"]["title"] == ["Scott Derrickson", "Ed Wood"]
    assert row["supporting_facts"]["title"] == ["Scott Derrickson", "Ed Wood"]
    examples, passages = hotpot_to_examples([row], "eval")
    assert examples[0]["supporting_titles"] == ["Ed Wood", "Scott Derrickson"]
    assert len(passages) == 2


if __name__ == "__main__":
    test_official_json_row_maps_to_hf_schema()
    print("ok")
