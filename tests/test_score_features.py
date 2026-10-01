#!/usr/bin/env python3
"""Score channels must vary on the 300 eval trajectories. No GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_env import SCORE_FEATURE_INDEX, SCORE_NORM_PATH, vectorize_structured_obs
from src.rewards import WEAK_MEAN_SCORE, calibration_score

# The v3 λ=0 exam. Final evidence is raw BM25 (no rerank). Top-1 runs from
# about 26 to 122, median about 49: the set where tanh(score / 5) was 1.0.
_EVAL_TRAJECTORIES = ROOT / "results" / "trajectories" / "learned_frontier_lambda0_v3.jsonl"
# Same 300 questions after a rerank, where the mean drops but still stays above 3.
_RERANK_TRAJECTORIES = (
    ROOT / "results" / "trajectories" / "rule_based_default.jsonl",
    ROOT / "results" / "trajectories" / "max_tools_default.jsonl",
)
_MIN_STD = 0.05
_CFG = {"agent": {"max_steps": 8}, "budget": {"max_usd": 0.05}}


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 300:
        raise AssertionError(f"{path.name} has {len(rows)} rows; expected 300")
    return rows


def _obs_from_trajectory(row: dict) -> dict:
    scores = [float(doc["score"]) for doc in row["retrieved"]]
    if len(scores) < 2:
        raise AssertionError(f"{row.get('id')} has {len(scores)} retrieved scores")
    stop = row["trajectory"][-1]["observation_before"]
    return {
        "mean_score": sum(scores) / len(scores),
        "n_evidence": len(scores),
        "remaining_steps": stop.get("remaining_steps", 1),
        "remaining_usd": stop.get("remaining_usd", 0.05),
        "top_scores": scores,
        "counts": stop.get("counts") or {},
        "verification": row.get("verify_out"),
    }


def test_score_norm_is_the_train_slice() -> None:
    payload = json.loads(SCORE_NORM_PATH.read_text(encoding="utf-8"))
    assert payload["n_questions"] == 100
    assert "eval" not in Path(payload["source"]).name
    assert payload["n_by_dataset"] == {"hotpot_qa": 60, "natural_questions": 40}
    assert set(payload["features"]) == {"mean", "top1", "gap", "top5_min"}


def test_eval_trajectories_score_features_vary() -> None:
    """Every standardized score channel has to move across the 300 questions.

    Step counters, the budget fraction, and the verify one-hot are constant
    on a fixed action recipe, so they are not part of this check. The bug was
    the score channels: tanh(score / 5) made the mean and the top-1 identical
    on every question.
    """
    rows = _read_jsonl(_EVAL_TRAJECTORIES)
    vecs = np.stack(
        [vectorize_structured_obs(_obs_from_trajectory(row), _CFG) for row in rows],
        axis=0,
    )
    assert vecs.shape == (300, 16)
    std = vecs.std(axis=0)
    for name, idx in SCORE_FEATURE_INDEX.items():
        value = float(std[idx])
        assert value >= _MIN_STD, f"{name} (index {idx}) std={value:.6f} < {_MIN_STD}"


def test_weak_mean_threshold_does_not_fire_on_logged_eval() -> None:
    """+0.6 for mean score < 3 never sees a real retrieved mean on these exams."""
    floors: dict[str, float] = {}
    paths = (_EVAL_TRAJECTORIES, *_RERANK_TRAJECTORIES)
    for path in paths:
        observed = []
        for row in _read_jsonl(path):
            for step in row["trajectory"]:
                obs = step.get("observation_before") or {}
                if int(obs.get("n_evidence") or 0) <= 0:
                    continue
                observed.append(float(obs["mean_score"]))
        floors[path.name] = min(observed)
        assert floors[path.name] >= WEAK_MEAN_SCORE, (
            f"{path.name} mean score {floors[path.name]} is below {WEAK_MEAN_SCORE}"
        )
    # The floor on the raw-BM25 exam is still a lazy abstain, not a justified one.
    evidence = [{"title": "x", "text": "y", "score": floors[_EVAL_TRAJECTORIES.name]}]
    assert calibration_score("ABSTAIN", "Paris", True, evidence) == -0.2


def main() -> None:
    test_score_norm_is_the_train_slice()
    test_eval_trajectories_score_features_vary()
    test_weak_mean_threshold_does_not_fire_on_logged_eval()
    print("SCORE FEATURES OK")


if __name__ == "__main__":
    main()
