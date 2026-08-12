"""Non-RL policies for Milestone 2 stable baselines."""

from __future__ import annotations

import random
from typing import Any

from ..agentic_rag import ACTIONS, AgentState


def naive_stop_policy(obs: dict[str, Any], state: AgentState) -> str:
    """Retrieve once then stop — mirrors standard RAG inside the agent API."""
    if state.counts.get("retrieve", 0) == 0:
        return "retrieve"
    return "stop"


def rule_based_policy(obs: dict[str, Any], state: AgentState) -> str:
    """Deterministic thresholds for retrieve / rewrite / rerank / verify / stop.

    Designed for a *stable* baseline before RL. Thresholds are documented in
    docs/IMPLEMENTATION_DECISIONS.md and held fixed unless ablated.
    """
    max_retrieve = 3
    mean_score = float(obs.get("mean_score") or 0.0)
    n_ev = int(obs.get("n_evidence") or 0)
    remaining_steps = int(obs.get("remaining_steps") or 0)
    verify = obs.get("verification")

    if remaining_steps <= 1:
        return "stop"

    if state.counts.get("retrieve", 0) == 0:
        return "retrieve"

    # Empty / very weak evidence → rewrite then retrieve again
    if n_ev == 0:
        if state.counts.get("rewrite", 0) < 1:
            return "rewrite"
        if state.counts.get("retrieve", 0) < max_retrieve:
            return "retrieve"
        return "stop"

    # Strong retrieval: optional rerank once, verify, stop (avoid over-retrieve)
    if mean_score >= 3.0:
        if state.counts.get("rerank", 0) < 1 and n_ev >= 3:
            return "rerank"
        if state.counts.get("verify", 0) < 1:
            return "verify"
        return "stop"

    # Medium evidence
    if mean_score >= 1.5:
        if state.counts.get("rerank", 0) < 1:
            return "rerank"
        if state.counts.get("verify", 0) < 1:
            return "verify"
        if verify and verify.get("support", 0.0) < 0.35 and state.counts.get("retrieve", 0) < 2:
            return "retrieve"
        return "stop"

    # Weak evidence
    if state.counts.get("rewrite", 0) < 1:
        return "rewrite"
    if state.counts.get("retrieve", 0) < max_retrieve:
        return "retrieve"
    if state.counts.get("verify", 0) < 1:
        return "verify"
    return "stop"


def always_max_policy(obs: dict[str, Any], state: AgentState) -> str:
    """Upper-cost reference: use every tool up to caps then stop."""
    if state.counts.get("retrieve", 0) < 2:
        return "retrieve"
    if state.counts.get("rewrite", 0) < 1:
        return "rewrite"
    if state.counts.get("retrieve", 0) < 3:
        return "retrieve"
    if state.counts.get("rerank", 0) < 1:
        return "rerank"
    if state.counts.get("verify", 0) < 1:
        return "verify"
    return "stop"


def random_policy(obs: dict[str, Any], state: AgentState, rng: random.Random | None = None) -> str:
    rng = rng or random
    if int(obs.get("remaining_steps") or 0) <= 1:
        return "stop"
    # Bias toward retrieve early
    weights = {
        "retrieve": 0.35,
        "rewrite": 0.15,
        "rerank": 0.15,
        "verify": 0.15,
        "stop": 0.20,
    }
    actions = list(weights.keys())
    probs = [weights[a] for a in actions]
    return rng.choices(actions, weights=probs, k=1)[0]


def get_policy(name: str):
    name = name.lower()
    if name in {"naive_rag", "naive"}:
        return naive_stop_policy, "naive_rag"
    if name in {"rule_based", "rule"}:
        return rule_based_policy, "rule_based"
    if name in {"always_max", "max"}:
        return always_max_policy, "always_max"
    if name == "random":
        return random_policy, "random"
    raise ValueError(f"Unknown policy: {name}")
