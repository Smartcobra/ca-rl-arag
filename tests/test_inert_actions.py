#!/usr/bin/env python3
"""Actions that cannot change the answer written at stop. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from count_inert_actions import (  # noqa: E402
    ONESHOT,
    TRAJ,
    idle_flags,
    kept_usd,
    load_jsonl,
    step_usd,
    summarize,
)


def _row(actions: list[str], queries: dict[int, str] | None = None) -> dict:
    """One episode. Each action costs 0.00005, starting from a budget of 0.05."""
    queries = queries or {}
    budget = 0.05
    spent = 0.0
    steps = []
    for i, action in enumerate(actions):
        step = {
            "action": action,
            "observation_before": {"remaining_usd": budget - spent},
        }
        if action == "rewrite":
            step["rewritten_query"] = queries.get(i, f"q{i}")
        spent += 0.00005
        steps.append(step)
    return {
        "id": "synthetic",
        "question": "original",
        "total_usd": spent,
        "trajectory": steps,
    }


def test_verify_repeat_retrieve_and_orphan_rewrite_are_idle() -> None:
    # retrieve, verify, rewrite, retrieve (new query), retrieve (same), rewrite, stop
    row = _row(
        ["retrieve", "verify", "rewrite", "retrieve", "retrieve", "rewrite", "stop"],
        {2: "fresh"},
    )
    assert idle_flags(row) == [False, True, False, False, True, True, False]
    costs = step_usd(row)
    assert abs(sum(costs) - row["total_usd"]) < 1e-12
    # Two idle retrieves would be wrong: the retrieve after "fresh" is a new query.
    assert sum(idle_flags(row)) == 3


def test_rerank_is_not_idle() -> None:
    row = _row(["retrieve", "rerank", "stop"])
    assert idle_flags(row) == [False, False, False]


def test_lambda5_kept_dollars_match_oneshot_and_seed44_term() -> None:
    oneshot = {row["id"]: row for row in load_jsonl(TRAJ / ONESHOT)}
    lam5 = load_jsonl(TRAJ / "learned_frontier_lambda5_rf5s42.jsonl")
    for row in lam5:
        assert abs(kept_usd(row) - float(oneshot[row["id"]]["total_usd"])) < 1e-9
    summary = summarize(lam5)
    assert summary["idle_1e4"] == 0.88
    assert abs(summary["mean_idle_actions"] - 1.087) < 0.001

    seed42 = summarize(load_jsonl(TRAJ / "learned_frontier_lambda20_rfs42.jsonl"))
    assert seed42["idle_1e4"] == 1.44
    assert abs(seed42["mean_idle_actions"] - 2.34) < 0.01
    assert abs(seed42["mean_tool_actions"] - 5.51) < 0.01

    seed44 = load_jsonl(TRAJ / "learned_frontier_lambda20_rfs44.jsonl")
    summary44 = summarize(seed44)
    assert summary44["idle_1e4"] == 1.93
    extra = sum(
        float(row["total_usd"]) - float(oneshot[row["id"]]["total_usd"]) for row in seed44
    ) / len(seed44)
    assert abs(20 * extra - 0.00781272) < 1e-8


def main() -> None:
    test_verify_repeat_retrieve_and_orphan_rewrite_are_idle()
    test_rerank_is_not_idle()
    test_lambda5_kept_dollars_match_oneshot_and_seed44_term()
    print("INERT ACTIONS OK")


if __name__ == "__main__":
    main()
