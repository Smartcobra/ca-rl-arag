"""Evaluation metrics for quality and efficiency."""

from __future__ import annotations

from typing import Any

from .utils import exact_match, token_f1


def compute_answer_metrics(pred: str, gold: str | list[str], abstained: bool = False) -> dict[str, float]:
    if abstained:
        # Abstention: no EM/F1 credit; calibration handled in reward.
        return {"em": 0.0, "f1": 0.0, "abstained": 1.0}
    return {
        "em": exact_match(pred, gold),
        "f1": token_f1(pred, gold),
        "abstained": 0.0,
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [
        "em",
        "f1",
        "q_ans",
        "q_ground",
        "q_cal",
        "p_hall",
        "reward",
        "total_usd",
        "total_latency_ms",
        "total_tokens",
        "n_steps",
        "n_retrieve",
        "n_rewrite",
        "n_rerank",
        "n_verify",
        "usd_per_correct",
    ]
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[f"mean_{k}"] = sum(vals) / len(vals)
    n = len(rows)
    out["n_examples"] = float(n)
    correct = sum(1 for r in rows if r.get("em", 0) >= 1.0)
    out["n_correct"] = float(correct)
    total_usd = sum(float(r.get("total_usd", 0.0)) for r in rows)
    out["total_usd_all"] = total_usd
    out["usd_per_correct"] = (total_usd / correct) if correct else float("inf")
    return out
