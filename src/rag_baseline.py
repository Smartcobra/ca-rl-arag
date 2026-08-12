"""Standard RAG baseline: retrieve once → generate once (Lewis-style)."""

from __future__ import annotations

from typing import Any

from .cost import CostTracker
from .generation import build_generator
from .rewards import RewardComputer
from .utils import ms_since, timed


class RAGBaseline:
    def __init__(self, cfg: dict[str, Any], retriever, reward_computer: RewardComputer | None = None):
        self.cfg = cfg
        self.retriever = retriever
        self.generator = build_generator(cfg)
        self.reward_computer = reward_computer or RewardComputer(cfg["reward_weights"], cfg.get("price_card"))
        self.top_k = int(cfg.get("retrieval", {}).get("top_k", 5))

    def run(self, example: dict[str, Any]) -> dict[str, Any]:
        tracker = CostTracker(self.cfg["price_card"])
        question = example["question"]
        t0 = timed()
        docs = self.retriever.search(question, top_k=self.top_k)
        tracker.price_retrieve(ms_since(t0), method=self.cfg.get("retrieval", {}).get("method", "bm25"))

        t1 = timed()
        answer, mode, usage = self.generator.generate(question, docs, allow_abstain=True)
        tracker.price_generate(
            ms_since(t1),
            usage.get("prompt_tokens", 200),
            usage.get("completion_tokens", 32),
            action="stop",
        )

        abstained = mode == "abstain" or answer.upper() == "ABSTAIN"
        budget = {
            "max_usd": float(self.cfg.get("budget", {}).get("max_usd", 0.05)),
            "violated": tracker.total_usd > float(self.cfg.get("budget", {}).get("max_usd", 0.05)),
        }
        rb = self.reward_computer.compute(
            pred=answer,
            gold=example.get("answers") or example.get("answer"),
            evidence=docs,
            supporting_titles=example.get("supporting_titles"),
            verify_out=None,
            cost_summary=tracker.summary(),
            n_actions=2,
            verified=False,
            abstained=abstained,
            budget=budget,
        )
        out = {
            "id": example.get("id"),
            "dataset": example.get("dataset"),
            "question": question,
            "prediction": answer,
            "gold": example.get("answers") or example.get("answer"),
            "abstained": abstained,
            "policy": "naive_rag",
            "n_steps": 2,
            "n_retrieve": 1,
            "n_rewrite": 0,
            "n_rerank": 0,
            "n_verify": 0,
            "retrieved": docs,
            "trajectory": [
                {"step": 0, "action": "retrieve", "n_docs": len(docs)},
                {"step": 1, "action": "stop", "answer": answer, "mode": mode},
            ],
            **tracker.summary(),
            **rb.as_dict(),
            "em": rb.components.get("em", 0.0),
            "f1": rb.components.get("f1", 0.0),
        }
        correct = out["em"] >= 1.0
        out["usd_per_correct"] = out["total_usd"] if correct else None
        return out
