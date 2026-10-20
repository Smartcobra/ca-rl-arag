#!/usr/bin/env python3
"""Nominal latency is the reward. The wall clock is only logged."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_rag import AgenticRAG
from src.cost import CostTracker
from src.rewards import RewardComputer


def _card() -> dict:
    return {
        "retrieval": {"bm25_per_call": 0.00005},
        "llm": {"input_per_1m": 0.15, "output_per_1m": 0.60},
        "rerank": {"cross_encoder_per_doc": 0.00001},
        "verify": {"lexical_nli_per_call": 0.00001},
        "latency": {
            "latency_unit_usd_per_sec": 0.001,
            "nominal_ms_per_call": {
                "retrieve": 212.3982,
                "rewrite": 999.2325,
                "rerank": 143.1998,
                "verify": 0.0,
                "generate": 853.9484,
            },
            "nominal_ms_per_completion_token": {"generate": 51.9733},
        },
    }


def _track(clock_retrieve: float, clock_generate: float, completion_tokens: int) -> CostTracker:
    tracker = CostTracker(_card())
    tracker.price_retrieve(clock_retrieve)
    tracker.price_generate(clock_generate, prompt_tokens=700, completion_tokens=completion_tokens, action="stop")
    return tracker


def test_clock_does_not_change_nominal_or_reward() -> None:
    slow = _track(50.0, 4000.0, 2)
    fast = _track(5.0, 900.0, 2)
    assert slow.total_latency_ms != fast.total_latency_ms
    assert slow.nominal_latency_ms == fast.nominal_latency_ms
    expected = 212.3982 + 853.9484 + 2 * 51.9733
    assert abs(slow.nominal_latency_ms - expected) < 1e-6

    weights = {"mu_latency": 0.5, "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "lambda_cost": 20.0}
    computer = RewardComputer(weights, _card())
    kwargs = dict(
        pred="no",
        gold="no",
        evidence=[{"title": "t", "text": "no", "score": 10.0}],
        supporting_titles=None,
        verify_out=None,
        n_actions=2,
        verified=False,
        abstained=False,
        budget={"max_usd": 0.05, "violated": False},
    )
    slow_reward = computer.compute(cost_summary=slow.summary(), **kwargs)
    fast_reward = computer.compute(cost_summary=fast.summary(), **kwargs)
    assert slow_reward.c_lat == fast_reward.c_lat
    assert slow_reward.reward == fast_reward.reward
    # μ = 0.5, unit = 0.001 $/s. Do not drop μ.
    assert abs(slow_reward.c_lat - 0.5 * (expected / 1000.0) * 0.001) < 1e-12


def test_extra_completion_token_is_priced() -> None:
    short = _track(10.0, 1000.0, 2)
    longer = _track(10.0, 1000.0, 3)
    assert abs((longer.nominal_latency_ms - short.nominal_latency_ms) - 51.9733) < 1e-9


def test_rewrite_is_per_call_only() -> None:
    tracker = CostTracker(_card())
    tracker.price_rewrite(12345.0, prompt_tokens=100, completion_tokens=40)
    assert tracker.events[0].nominal_latency_ms == 999.2325
    assert tracker.events[0].latency_ms == 12345.0


def test_budget_uses_the_table() -> None:
    agent = AgenticRAG(
        {"budget": {"max_usd": 1.0, "max_tokens": 10**9, "max_latency_ms": 500.0}, "reward_weights": {}},
        retriever=None,
        reward_computer=RewardComputer({}, _card()),
        generator=object(),
    )
    # Measured clock is over the cap. Nominal retrieve is 212 ms, under 500.
    under = CostTracker(_card())
    under.price_retrieve(10_000.0)
    assert under.total_latency_ms > 500.0
    assert not agent.budget_exhausted(under)
    # Two generates at 2 tokens each: 2 * 957.89 ms, over 500, whatever the clock says.
    over = CostTracker(_card())
    over.price_generate(1.0, 10, 2, action="stop")
    over.price_generate(1.0, 10, 2, action="stop")
    assert over.total_latency_ms < 500.0
    assert agent.budget_exhausted(over)


def test_frontier_lambda20_keeps_mu() -> None:
    text = (ROOT / "configs" / "reward_weights.yaml").read_text()
    block = text.split("frontier_lambda20:", 1)[1].split("frontier_act02:", 1)[0]
    assert "mu_latency: 0.5" in block


def main() -> None:
    test_clock_does_not_change_nominal_or_reward()
    test_extra_completion_token_is_priced()
    test_rewrite_is_per_call_only()
    test_budget_uses_the_table()
    test_frontier_lambda20_keeps_mu()
    print("NOMINAL LATENCY OK")


if __name__ == "__main__":
    main()
