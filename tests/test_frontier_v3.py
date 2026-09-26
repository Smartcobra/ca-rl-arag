#!/usr/bin/env python3
"""v3 slice sizes, naive-anchor advantage, and locked eval ids. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.data.id_lock import assert_disjoint, assert_ids_unchanged, load_locked_eval_ids
from src.data.train_valid import split_prefix
from src.utils import read_jsonl


def test_frontier_v3_sizes() -> None:
    cfg = load_yaml(ROOT / "configs" / "frontier_v3.yaml")
    data = cfg["data"]
    assert int(data["train_hotpot"]) + int(data["train_nq"]) == 500
    n_valid = int(data["valid_hotpot"]) + int(data["valid_nq"])
    assert 150 <= n_valid <= 200
    assert int(data["eval_hotpot"]) == 150
    assert int(data["eval_nq"]) == 150
    assert int(cfg["policy"]["learned"]["epochs"]) == 20
    assert "eval" not in Path(data["train_file"]).name
    assert "eval" not in Path(data["valid_file"]).name
    assert data["train_file"] != "data/processed/train_slice.jsonl"
    default = load_yaml(ROOT / "configs" / "default.yaml")
    assert int(default["policy"]["learned"]["epochs"]) == 40
    assert int(default["data"]["train_hotpot"]) == 60


def test_split_prefix_is_the_next_block() -> None:
    rows = [{"id": f"q{i}"} for i in range(12)]
    train, valid = split_prefix(rows, 7, 4)
    assert [row["id"] for row in train] == [f"q{i}" for i in range(7)]
    assert [row["id"] for row in valid] == [f"q{i}" for i in range(7, 11)]
    assert_disjoint(
        ("train", [row["id"] for row in train]),
        ("valid", [row["id"] for row in valid]),
        ("eval", ["locked"]),
    )


def test_advantage_is_sampled_minus_naive() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_policy import advantage_vs_naive

    assert abs(advantage_vs_naive(0.71, 0.58) - 0.13) < 1e-9
    assert advantage_vs_naive(0.4, 0.4) == 0.0


def test_locked_eval_ids_match_the_file() -> None:
    locked = load_locked_eval_ids()
    assert len(locked) == 300
    assert len(set(locked)) == 300
    path = ROOT / "data" / "processed" / "eval_slice.jsonl"
    if not path.exists():
        raise AssertionError(f"missing {path}; the 300 ids cannot be checked")
    got = [str(row["id"]) for row in read_jsonl(path)]
    assert_ids_unchanged(got, locked, "eval_slice.jsonl")


def main() -> None:
    test_frontier_v3_sizes()
    test_split_prefix_is_the_next_block()
    test_advantage_is_sampled_minus_naive()
    test_locked_eval_ids_match_the_file()
    print("FRONTIER V3 OK")


if __name__ == "__main__":
    main()
