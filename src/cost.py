"""Cost / latency metering from the FinOps price card."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostEvent:
    action: str
    usd: float
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostTracker:
    price_card: dict[str, Any]
    events: list[CostEvent] = field(default_factory=list)

    def add(self, event: CostEvent) -> CostEvent:
        self.events.append(event)
        return event

    @property
    def total_usd(self) -> float:
        return float(sum(e.usd for e in self.events))

    @property
    def total_latency_ms(self) -> float:
        return float(sum(e.latency_ms for e in self.events))

    @property
    def total_tokens(self) -> int:
        return int(sum(e.prompt_tokens + e.completion_tokens for e in self.events))

    def llm_tokens_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        llm = self.price_card.get("llm", {})
        return (
            prompt_tokens * float(llm.get("input_per_1m", 0.15)) / 1_000_000.0
            + completion_tokens * float(llm.get("output_per_1m", 0.60)) / 1_000_000.0
        )

    def price_retrieve(self, latency_ms: float, method: str = "bm25") -> CostEvent:
        ret = self.price_card.get("retrieval", {})
        usd = float(ret.get("bm25_per_call", 0.00005))
        if method == "dense":
            usd = float(ret.get("dense_per_call", 0.0001))
        return self.add(CostEvent("retrieve", usd=usd, latency_ms=latency_ms))

    def price_rewrite(self, latency_ms: float, prompt_tokens: int, completion_tokens: int) -> CostEvent:
        usd = self.llm_tokens_usd(prompt_tokens, completion_tokens)
        if usd <= 0:
            usd = float(self.price_card.get("actions", {}).get("rewrite", 0.00025))
        return self.add(
            CostEvent(
                "rewrite",
                usd=usd,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    def price_rerank(self, latency_ms: float, n_docs: int) -> CostEvent:
        per = float(self.price_card.get("rerank", {}).get("cross_encoder_per_doc", 0.00001))
        return self.add(CostEvent("rerank", usd=per * max(n_docs, 1), latency_ms=latency_ms, meta={"n_docs": n_docs}))

    def price_verify(self, latency_ms: float, backend: str = "lexical_nli") -> CostEvent:
        v = self.price_card.get("verify", {})
        if backend == "neural_nli":
            usd = float(v.get("neural_nli_per_call", 0.00008))
        else:
            usd = float(v.get("lexical_nli_per_call", 0.00001))
        return self.add(CostEvent("verify", usd=usd, latency_ms=latency_ms, meta={"backend": backend}))

    def price_generate(self, latency_ms: float, prompt_tokens: int, completion_tokens: int, action: str = "stop") -> CostEvent:
        usd = self.llm_tokens_usd(prompt_tokens, completion_tokens)
        if usd <= 0:
            usd = float(self.price_card.get("actions", {}).get(action, 0.0002))
        return self.add(
            CostEvent(
                action,
                usd=usd,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    def latency_cost_usd(self, mu: float) -> float:
        unit = float(self.price_card.get("latency", {}).get("latency_unit_usd_per_sec", 0.001))
        return mu * (self.total_latency_ms / 1000.0) * unit

    def summary(self) -> dict[str, Any]:
        by_action: dict[str, float] = {}
        for e in self.events:
            by_action[e.action] = by_action.get(e.action, 0.0) + e.usd
        return {
            "total_usd": self.total_usd,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "n_events": len(self.events),
            "usd_by_action": by_action,
            "action_counts": {
                a: sum(1 for e in self.events if e.action == a)
                for a in sorted({e.action for e in self.events})
            },
        }
