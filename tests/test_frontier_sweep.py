#!/usr/bin/env python3
"""Frontier presets straddle break-even; plot labels match yaml. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, load_yaml

EXPECTED = (
    ("frontier_lambda0", 0.0, 0.0),
    ("frontier_lambda20", 20.0, 0.0),
    ("frontier_act02", 80.0, 0.02),
)
RETIRED = ("frontier_act0", "frontier_act005")
# Extra max_tools $ vs ~0.028 quality: λ=20 is still +EV, λ=80 is not.
EXTRA_USD = 0.00058
QUALITY_GAIN = 0.028


def test_frontier_presets_cross_breakeven() -> None:
    rewards = load_yaml(ROOT / "configs" / "reward_weights.yaml")
    presets = rewards["presets"]
    sweep = rewards["frontier_sweep"]
    names = list(sweep["presets"])
    assert names == [p[0] for p in EXPECTED]
    assert sweep["lambda_cost"] == [p[1] for p in EXPECTED]
    assert sweep["act_penalty"] == [p[2] for p in EXPECTED]
    for name in RETIRED:
        assert name not in presets, f"{name} should have been replaced"
    for name, lam, act in EXPECTED:
        w = presets[name]
        assert float(w["lambda_cost"]) == lam
        assert float(w["act_penalty"]) == act
        cfg = load_config(reward_preset=name)
        assert cfg["reward_preset_name"] == name
        assert float(cfg["reward_weights"]["lambda_cost"]) == lam
        assert float(cfg["reward_weights"]["act_penalty"]) == act
    assert EXTRA_USD * 20 < QUALITY_GAIN
    assert EXTRA_USD * 80 > QUALITY_GAIN


def test_plot_results_matches_yaml() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from plot_results import FRONTIER_LEARNED, FRONTIER_PRESET_META

    assert [p for p, _ in FRONTIER_LEARNED] == [p[0] for p in EXPECTED]
    for name, lam, act in EXPECTED:
        meta = FRONTIER_PRESET_META[name]
        assert meta["lambda_cost"] == lam
        assert meta["act_penalty"] == act


def test_default_epochs_are_not_five() -> None:
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    epochs = int(cfg["policy"]["learned"]["epochs"])
    assert 30 <= epochs <= 50, f"ranking trains need 30–50 epochs, got {epochs}"


def test_behavior_counts_one_recipe_and_stop_after_contradiction() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from plot_results import behavior_from_rows, controller_ceilings

    rows = [
        {
            "trajectory": [
                {"action": "verify", "verify": {"label": "contradiction"}},
                {"action": "verify", "verify": {"label": "support"}},
                {"action": "stop"},
            ]
        },
        {
            "trajectory": [
                {"action": "verify", "verify": {"label": "contradiction"}},
                {"action": "stop"},
            ]
        },
    ]
    stats = behavior_from_rows(rows)
    assert stats["n_action_sequences"] == 2
    assert stats["after_contradiction"] == "stop"
    assert behavior_from_rows([{"trajectory": [{"action": "retrieve"}, {"action": "stop"}]}])[
        "after_contradiction"
    ] == "no verify"

    naive = [
        {"id": "h", "dataset": "hotpot_qa", "em": 1, "total_usd": 1.0},
        {"id": "n", "dataset": "natural_questions", "em": 0, "total_usd": 1.0},
        {"id": "x", "dataset": "hotpot_qa", "em": 0, "total_usd": 1.0},
    ]
    lambda0 = [
        {"id": "h", "dataset": "hotpot_qa", "em": 0, "total_usd": 3.0},
        {"id": "n", "dataset": "natural_questions", "em": 1, "total_usd": 3.0},
        {"id": "x", "dataset": "hotpot_qa", "em": 0, "total_usd": 3.0},
    ]
    max_tools = [
        {"id": "h", "dataset": "hotpot_qa", "em": 1, "total_usd": 5.0},
        {"id": "n", "dataset": "natural_questions", "em": 0, "total_usd": 5.0},
        {"id": "x", "dataset": "hotpot_qa", "em": 1, "total_usd": 5.0},
    ]
    points = controller_ceilings(naive, lambda0, max_tools)
    assert [p["n_correct"] for p in points] == [2, 2, 3]
    assert points[0]["label"] == "2"
    assert points[0]["mean_total_usd"] == 5.0 / 3.0
    assert points[2]["n_correct"] == 3


def test_best_artifact_suffix_does_not_rename_other_policies() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_pilot import _artifact_stem

    assert _artifact_stem("learned", "frontier_lambda0", "best") == "learned_frontier_lambda0_best"
    assert _artifact_stem("learned", "frontier_lambda0", None) == "learned_frontier_lambda0"
    assert _artifact_stem("rule_based", "default", "best") == "rule_based_default"


def test_episode_stats_helper() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_policy import _episode_action_stats, _mean

    class _State:
        action_history = ["retrieve", "verify", "stop"]
        counts = {"retrieve": 1, "verify": 1}

    env = type("E", (), {"_state": _State()})()
    stats = _episode_action_stats(env)
    assert stats == {"n_steps": 3.0, "n_retrieve": 1.0, "n_verify": 1.0}
    assert _mean([]) == 0.0
    assert _mean([1.0, 3.0]) == 2.0


def main() -> None:
    test_frontier_presets_cross_breakeven()
    test_plot_results_matches_yaml()
    test_default_epochs_are_not_five()
    test_behavior_counts_one_recipe_and_stop_after_contradiction()
    test_best_artifact_suffix_does_not_rename_other_policies()
    test_episode_stats_helper()
    print("FRONTIER SWEEP OK")


if __name__ == "__main__":
    main()
