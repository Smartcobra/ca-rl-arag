#!/usr/bin/env python3
"""Fast smoke test for Milestone-2 pipeline (no network required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agentic_rag import AgenticRAG
from src.config import load_config
from src.data.loaders import build_synthetic_pilot, save_processed
from src.metrics import aggregate_metrics
from src.policies import get_policy
from src.rag_baseline import RAGBaseline
from src.rag_env import ACTION_TO_IDX, AgenticRAGEnv
from src.retrieval import BM25Retriever
from src.utils import read_jsonl, set_seed


def main() -> None:
    cfg = load_config()
    # Keep smoke offline/fast even if default.yaml points at Qwen.
    cfg.setdefault("generation", {})["backend"] = "extractive"
    set_seed(0)
    train, eval_set, passages = build_synthetic_pilot(
        n_train_hotpot=12, n_train_nq=8, n_eval_hotpot=8, n_eval_nq=6
    )
    out_dir = ROOT / "data" / "processed"
    save_processed(train, eval_set, passages, str(out_dir))

    # Refresh cfg paths relative to package
    corpus = read_jsonl(out_dir / "corpus.jsonl")
    examples = read_jsonl(out_dir / "eval_slice.jsonl")[:10]
    if not examples:
        examples = train[:10]

    retriever = BM25Retriever(corpus)
    baseline = RAGBaseline(cfg, retriever)
    r0 = baseline.run(examples[0])
    assert "reward" in r0 and "prediction" in r0

    mix_rows = [
        {"dataset": "hotpot_qa", "em": 0.0, "f1": 0.0, "reward": 0.1, "abstained": True},
        {"dataset": "hotpot_qa", "em": 1.0, "f1": 1.0, "reward": 0.9, "abstained": False},
        {"dataset": "natural_questions", "em": 1.0, "f1": 1.0, "reward": 0.8, "abstained": False},
        {"dataset": "natural_questions", "em": 1.0, "f1": 1.0, "reward": 0.8, "abstained": False},
    ]
    mix = aggregate_metrics(mix_rows)
    assert mix["mean_em"] == 0.75
    assert mix["n_correct"] == 3.0
    assert mix["n_abstained"] == 1.0
    assert mix["by_dataset"]["hotpot_qa"]["mean_em"] == 0.5
    assert mix["by_dataset"]["hotpot_qa"]["n_correct"] == 1.0
    assert mix["by_dataset"]["hotpot_qa"]["abstain_rate"] == 0.5
    assert mix["by_dataset"]["natural_questions"]["mean_em"] == 1.0
    assert mix["by_dataset"]["natural_questions"]["n_examples"] == 2.0

    agent = AgenticRAG(cfg, retriever)
    policy_fn, name = get_policy("rule_based")
    r1 = agent.run_with_policy(examples[0], policy_fn, policy_name=name)
    assert r1["n_retrieve"] >= 1
    assert r1["trajectory"]

    env = AgenticRAGEnv(cfg, retriever, examples, seed=0)
    obs, info = env.reset(options={"example": examples[0]})
    assert obs.shape == (10,)
    obs, reward, term, trunc, info = env.step(ACTION_TO_IDX["retrieve"])
    assert not term
    obs, reward, term, trunc, info = env.step(ACTION_TO_IDX["stop"])
    assert term
    assert "episode_result" in info

    # Reward ablation presets load
    for preset in ["default", "correctness_only", "lambda_zero", "high_cost_pressure"]:
        c = load_config(reward_preset=preset)
        assert c["reward_preset_name"] == preset

    print("SMOKE OK")
    print("baseline_pred:", r0["prediction"], "em=", r0["em"], "reward=", round(r0["reward"], 4))
    print("agent_pred:", r1["prediction"], "actions=", [t["action"] for t in r1["trajectory"]], "em=", r1["em"])


if __name__ == "__main__":
    main()
