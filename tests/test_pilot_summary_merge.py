#!/usr/bin/env python3
"""Unit tests for pilot_summary merge. No GPU, no eval slice."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import merge_pilot_summary_results


def test_learned_only_keeps_rule_and_max() -> None:
    existing = {
        "results": {
            "naive_rag": {"mean_em": 0.333},
            "rule_based": {"mean_em": 0.323},
            "max_tools": {"mean_em": 0.350},
            "learned": {"mean_em": 0.100},
        }
    }
    merged = merge_pilot_summary_results(existing, {"learned": {"mean_em": 0.333}})
    assert list(merged) == ["naive_rag", "rule_based", "max_tools", "learned"]
    assert merged["rule_based"]["mean_em"] == 0.323
    assert merged["max_tools"]["mean_em"] == 0.350
    assert merged["learned"]["mean_em"] == 0.333


def test_four_policy_run_replaces_all_rows() -> None:
    existing = {"results": {"naive_rag": {"mean_em": 0.1}, "learned": {"mean_em": 0.2}}}
    new = {
        "naive_rag": {"mean_em": 0.333},
        "rule_based": {"mean_em": 0.323},
        "max_tools": {"mean_em": 0.350},
        "learned": {"mean_em": 0.333},
    }
    merged = merge_pilot_summary_results(existing, new)
    assert merged == new


def test_no_prior_file_writes_this_run_only() -> None:
    new = {"learned": {"mean_em": 0.333}}
    assert merge_pilot_summary_results(None, new) == new
    assert merge_pilot_summary_results({}, new) == new
    assert merge_pilot_summary_results({"results": "bad"}, new) == new


if __name__ == "__main__":
    test_learned_only_keeps_rule_and_max()
    test_four_policy_run_replaces_all_rows()
    test_no_prior_file_writes_this_run_only()
    print("ok")
