#!/usr/bin/env python3
"""Paired sign test and exact-match bootstrap. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_to_oneshot import (  # noqa: E402
    BOOT_SEED,
    METHOD,
    N_BOOT,
    bootstrap_em,
    exact_sign_pvalue,
    paired_counts,
)
from src.config import load_config


def test_sign_test_edges() -> None:
    assert exact_sign_pvalue(0, 0) == 1.0
    assert exact_sign_pvalue(7, 4) == exact_sign_pvalue(4, 7)
    assert exact_sign_pvalue(10, 0) < exact_sign_pvalue(6, 5)
    assert 0.0 <= exact_sign_pvalue(7, 4) <= 1.0
    # One discordant question is always a two-sided p of 1.
    assert exact_sign_pvalue(1, 0) == 1.0
    # Two repairs and zero breaks: both tails of a fair coin, p = 0.5.
    assert exact_sign_pvalue(2, 0) == 0.5


def test_sign_test_matches_scipy_when_present() -> None:
    try:
        from scipy.stats import binomtest
    except ImportError:
        return
    for repairs, breaks in ((7, 4), (9, 6), (3, 0), (15, 15), (230, 66)):
        got = exact_sign_pvalue(repairs, breaks)
        expected = float(binomtest(repairs, repairs + breaks, 0.5, alternative="two-sided").pvalue)
        assert abs(got - expected) < 1e-9, (repairs, breaks, got, expected)


def test_bootstrap_is_stable() -> None:
    correct = [1, 0, 1, 1, 0, 0, 1, 0]
    first = bootstrap_em(correct)
    second = bootstrap_em(correct)
    assert first == second
    rate, low, high = first
    assert rate == 0.5
    assert 0.0 <= low <= rate <= high <= 1.0
    assert N_BOOT == 10_000
    assert BOOT_SEED == 42


def test_paired_counts_repairs_and_breaks() -> None:
    baseline = [
        {"id": "a", "em": 0, "dataset": "hotpot_qa"},
        {"id": "b", "em": 1, "dataset": "hotpot_qa"},
        {"id": "c", "em": 1, "dataset": "natural_questions"},
        {"id": "d", "em": 0, "dataset": "natural_questions"},
    ]
    learned = [
        {"id": "a", "em": 1, "dataset": "hotpot_qa"},
        {"id": "b", "em": 0, "dataset": "hotpot_qa"},
        {"id": "c", "em": 1, "dataset": "natural_questions"},
        {"id": "d", "em": 0, "dataset": "natural_questions"},
    ]
    row = paired_counts(baseline, learned)
    assert row["repairs"] == 1
    assert row["breaks"] == 1
    assert row["n_correct"] == 2
    assert row["by_dataset"]["hotpot_qa"] == {"repairs": 1, "breaks": 1}
    assert row["by_dataset"]["natural_questions"] == {"repairs": 0, "breaks": 0}
    assert row["sign_test_p"] == 1.0
    assert row["method"] == METHOD


def test_cost_sweep_presets_change_only_lambda() -> None:
    expected = {
        "frontier_lambda0": 0.0,
        "frontier_lambda5": 5.0,
        "frontier_lambda20": 20.0,
        "frontier_lambda80": 80.0,
    }
    for name, lam in expected.items():
        cfg = load_config(reward_preset=name)
        weights = cfg["reward_weights"]
        assert float(weights["lambda_cost"]) == lam
        assert float(weights["act_penalty"]) == 0.0
        assert float(weights["mu_latency"]) == 0.5
    older = load_config(reward_preset="frontier_act02")
    assert float(older["reward_weights"]["lambda_cost"]) == 80.0
    assert float(older["reward_weights"]["act_penalty"]) == 0.02


def main() -> None:
    test_sign_test_edges()
    test_sign_test_matches_scipy_when_present()
    test_bootstrap_is_stable()
    test_paired_counts_repairs_and_breaks()
    test_cost_sweep_presets_change_only_lambda()
    print("PAIRED SIGN OK")


if __name__ == "__main__":
    main()
