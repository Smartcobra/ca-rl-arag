"""Agentic RAG tools + loop: {retrieve, rewrite, rerank, verify, stop}."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .cost import CostTracker
from .generation import build_generator
from .retrieval import LexicalReranker
from .rewards import RewardComputer
from .utils import ms_since, timed, truncate
from .verification import build_verifier

ACTIONS = ("retrieve", "rewrite", "rerank", "verify", "stop")


@dataclass
class AgentState:
    question: str
    current_query: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    candidate_pool: list[dict[str, Any]] = field(default_factory=list)
    draft_answer: str = ""
    verify_out: dict[str, Any] | None = None
    action_history: list[str] = field(default_factory=list)
    step: int = 0
    done: bool = False
    stop_mode: str = "answer"  # answer | abstain | escalate
    last_flags: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: {"retrieve": 0, "rewrite": 0, "rerank": 0, "verify": 0})

    def observation(self, tracker: CostTracker, cfg: dict[str, Any]) -> dict[str, Any]:
        budget = cfg.get("budget", {})
        max_steps = int(cfg.get("agent", {}).get("max_steps", 6))
        return {
            "query": self.question,
            "current_query": self.current_query,
            "n_evidence": len(self.evidence),
            "n_candidates": len(self.candidate_pool),
            "top_scores": [float(d.get("score", 0.0)) for d in self.evidence[:5]],
            "mean_score": float(sum(d.get("score", 0.0) for d in self.evidence) / len(self.evidence)) if self.evidence else 0.0,
            "draft_answer": truncate(self.draft_answer, 200),
            "verification": self.verify_out,
            "action_history": list(self.action_history),
            "counts": dict(self.counts),
            "accumulated_cost_usd": tracker.total_usd,
            "accumulated_latency_ms": tracker.total_latency_ms,
            "accumulated_tokens": tracker.total_tokens,
            "remaining_usd": float(budget.get("max_usd", 0.05)) - tracker.total_usd,
            "remaining_tokens": int(budget.get("max_tokens", 8000)) - tracker.total_tokens,
            "remaining_latency_ms": float(budget.get("max_latency_ms", 30000)) - tracker.total_latency_ms,
            "remaining_steps": max_steps - self.step,
            "last_flags": dict(self.last_flags),
            "done": self.done,
        }


class AgenticRAG:
    def __init__(
        self,
        cfg: dict[str, Any],
        retriever,
        reward_computer: RewardComputer | None = None,
        generator=None,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self._owns_generator = generator is None
        self.generator = generator if generator is not None else build_generator(cfg)
        self.reranker = LexicalReranker()
        self.verifier = build_verifier(cfg)
        self.reward_computer = reward_computer or RewardComputer(cfg["reward_weights"], cfg.get("price_card"))
        self.top_k = int(cfg.get("retrieval", {}).get("top_k", 5))
        self.pool_size = int(cfg.get("retrieval", {}).get("candidate_pool_size", 20))
        self.rerank_n = int(cfg.get("retrieval", {}).get("rerank_top_n", 5))

    def close(self) -> None:
        if self._owns_generator:
            closer = getattr(self.generator, "close", None)
            if callable(closer):
                closer()
        self.generator = None
        verifier_close = getattr(self.verifier, "close", None)
        if callable(verifier_close):
            verifier_close()

    def new_state(self, example: dict[str, Any]) -> AgentState:
        q = example["question"]
        return AgentState(question=q, current_query=q)

    def budget_exhausted(self, tracker: CostTracker) -> bool:
        b = self.cfg.get("budget", {})
        if tracker.total_usd > float(b.get("max_usd", 0.05)):
            return True
        if tracker.total_tokens > int(b.get("max_tokens", 8000)):
            return True
        if tracker.total_latency_ms > float(b.get("max_latency_ms", 30000)):
            return True
        return False

    def step(self, state: AgentState, action: str, tracker: CostTracker) -> tuple[AgentState, dict[str, Any]]:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}; expected one of {ACTIONS}")
        info: dict[str, Any] = {"action": action, "step": state.step}

        if action == "retrieve":
            t0 = timed()
            pool = self.retriever.search(state.current_query, top_k=self.pool_size)
            tracker.price_retrieve(ms_since(t0))
            # Merge into candidate pool by id
            by_id = {d["passage_id"]: d for d in state.candidate_pool}
            for d in pool:
                prev = by_id.get(d["passage_id"])
                if prev is None or float(d.get("score", 0)) > float(prev.get("score", 0)):
                    by_id[d["passage_id"]] = d
            state.candidate_pool = sorted(by_id.values(), key=lambda x: float(x.get("score", 0)), reverse=True)
            state.evidence = state.candidate_pool[: self.top_k]
            state.counts["retrieve"] += 1
            state.last_flags = {"empty_retrieval": len(pool) == 0}
            info.update({"n_docs": len(pool), "top_ids": [d["passage_id"] for d in state.evidence]})

        elif action == "rewrite":
            t0 = timed()
            new_q, usage = self.generator.rewrite(state.current_query, state.evidence)
            tracker.price_rewrite(ms_since(t0), usage["prompt_tokens"], usage["completion_tokens"])
            state.current_query = new_q
            state.counts["rewrite"] += 1
            info.update({"rewritten_query": new_q})

        elif action == "rerank":
            t0 = timed()
            pool = state.candidate_pool or state.evidence
            reranked = self.reranker.rerank(state.current_query, pool, top_n=self.rerank_n)
            tracker.price_rerank(ms_since(t0), n_docs=len(pool) or 1)
            state.evidence = reranked
            state.candidate_pool = reranked + [c for c in state.candidate_pool if c["passage_id"] not in {r["passage_id"] for r in reranked}]
            state.counts["rerank"] += 1
            info.update({"top_ids": [d["passage_id"] for d in state.evidence]})

        elif action == "verify":
            # Ensure draft exists
            if not state.draft_answer:
                t_g = timed()
                ans, mode, usage = self.generator.generate(state.question, state.evidence, allow_abstain=True)
                tracker.price_generate(ms_since(t_g), usage["prompt_tokens"], usage["completion_tokens"], action="stop")
                state.draft_answer = ans
                info["draft_created"] = True
            t0 = timed()
            backend = self.cfg.get("verification", {}).get("backend", "lexical_nli")
            result = self.verifier.verify(state.draft_answer, state.evidence, question=state.question)
            tracker.price_verify(ms_since(t0), backend=backend)
            state.verify_out = result.as_dict()
            state.counts["verify"] += 1
            info.update({"verify": state.verify_out})

        elif action == "stop":
            t0 = timed()
            ans, mode, usage = self.generator.generate(state.question, state.evidence, allow_abstain=True)
            # If verify suggested contradiction / low support, prefer abstain
            if state.verify_out:
                if state.verify_out.get("label") == "contradiction" or (
                    state.verify_out.get("support", 1.0) < 0.25 and state.verify_out.get("uncertainty", 0) > 0.5
                ):
                    if mode != "abstain":
                        # keep answer but mark escalate uncertainty; rule policies may abstain
                        info["verify_warning"] = True
            tracker.price_generate(ms_since(t0), usage["prompt_tokens"], usage["completion_tokens"], action="stop")
            state.draft_answer = ans
            state.stop_mode = mode
            state.done = True
            info.update({"answer": ans, "mode": mode})

        state.action_history.append(action)
        state.step += 1
        return state, info

    def run_with_policy(
        self,
        example: dict[str, Any],
        policy_fn: Callable[[dict[str, Any], AgentState], str],
        policy_name: str = "custom",
    ) -> dict[str, Any]:
        tracker = CostTracker(self.cfg["price_card"])
        state = self.new_state(example)
        trajectory: list[dict[str, Any]] = []
        max_steps = int(self.cfg.get("agent", {}).get("max_steps", 6))
        force = bool(self.cfg.get("budget", {}).get("force_stop_on_exhaustion", True))

        while not state.done and state.step < max_steps:
            obs = state.observation(tracker, self.cfg)
            if force and self.budget_exhausted(tracker):
                action = "stop"
                obs["forced_stop"] = True
            else:
                action = policy_fn(obs, state)
            state, info = self.step(state, action, tracker)
            info["observation_before"] = {k: obs[k] for k in ("n_evidence", "mean_score", "remaining_usd", "remaining_steps", "counts")}
            trajectory.append(info)

        if not state.done:
            state, info = self.step(state, "stop", tracker)
            trajectory.append(info)

        abstained = state.stop_mode == "abstain" or state.draft_answer.upper() == "ABSTAIN"
        budget = {
            "max_usd": float(self.cfg.get("budget", {}).get("max_usd", 0.05)),
            "violated": self.budget_exhausted(tracker),
        }
        rb = self.reward_computer.compute(
            pred=state.draft_answer,
            gold=example.get("answers") or example.get("answer"),
            evidence=state.evidence,
            supporting_titles=example.get("supporting_titles"),
            verify_out=state.verify_out,
            cost_summary=tracker.summary(),
            n_actions=len(state.action_history),
            verified=state.counts.get("verify", 0) > 0,
            abstained=abstained,
            budget=budget,
        )
        out = {
            "id": example.get("id"),
            "dataset": example.get("dataset"),
            "question": example["question"],
            "prediction": state.draft_answer,
            "gold": example.get("answers") or example.get("answer"),
            "abstained": abstained,
            "policy": policy_name,
            "n_steps": len(state.action_history),
            "n_retrieve": state.counts.get("retrieve", 0),
            "n_rewrite": state.counts.get("rewrite", 0),
            "n_rerank": state.counts.get("rerank", 0),
            "n_verify": state.counts.get("verify", 0),
            "retrieved": state.evidence,
            "verify_out": state.verify_out,
            "current_query": state.current_query,
            "trajectory": trajectory,
            **tracker.summary(),
            **rb.as_dict(),
            "em": rb.components.get("em", 0.0),
            "f1": rb.components.get("f1", 0.0),
        }
        correct = out["em"] >= 1.0
        out["usd_per_correct"] = out["total_usd"] if correct else None
        return out
