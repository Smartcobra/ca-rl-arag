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
    # Table value used by the reward and the latency budget.
    # latency_ms stays the measured wall clock, for logging only.
    nominal_latency_ms: float = 0.0


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
        """Measured wall clock. Logged, and not an input to the reward."""
        return float(sum(e.latency_ms for e in self.events))

    @property
    def nominal_latency_ms(self) -> float:
        """Fixed table latency. This is what the reward and the budget see."""
        return float(sum(e.nominal_latency_ms for e in self.events))

    def _nominal_latency_ms(self, action: str, measured_ms: float, completion_tokens: int = 0) -> float:
        lat = self.price_card.get("latency") or {}
        per_call = lat.get("nominal_ms_per_call")
        if not isinstance(per_call, dict):
            return float(measured_ms)
        key = "generate" if action in {"stop", "generate"} else action
        base = float(per_call.get(key, 0.0))
        extra = 0.0
        if key == "generate":
            per_tok = lat.get("nominal_ms_per_completion_token") or {}
            extra = float(per_tok.get("generate", 0.0)) * max(int(completion_tokens), 0)
        return base + extra

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
        return self.add(
            CostEvent(
                "retrieve",
                usd=usd,
                latency_ms=latency_ms,
                nominal_latency_ms=self._nominal_latency_ms("retrieve", latency_ms),
            )
        )

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
                nominal_latency_ms=self._nominal_latency_ms("rewrite", latency_ms, completion_tokens),
            )
        )

    def price_rerank(self, latency_ms: float, n_docs: int) -> CostEvent:
        per = float(self.price_card.get("rerank", {}).get("cross_encoder_per_doc", 0.00001))
        return self.add(
            CostEvent(
                "rerank",
                usd=per * max(n_docs, 1),
                latency_ms=latency_ms,
                meta={"n_docs": n_docs},
                nominal_latency_ms=self._nominal_latency_ms("rerank", latency_ms),
            )
        )

    def price_verify(self, latency_ms: float, backend: str = "lexical_nli") -> CostEvent:
        v = self.price_card.get("verify", {})
        if backend == "neural_nli":
            usd = float(v.get("neural_nli_per_call", 0.00008))
        else:
            usd = float(v.get("lexical_nli_per_call", 0.00001))
        return self.add(
            CostEvent(
                "verify",
                usd=usd,
                latency_ms=latency_ms,
                meta={"backend": backend},
                nominal_latency_ms=self._nominal_latency_ms("verify", latency_ms),
            )
        )

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
                nominal_latency_ms=self._nominal_latency_ms(action, latency_ms, completion_tokens),
            )
        )

    def latency_cost_usd(self, mu: float) -> float:
        unit = float(self.price_card.get("latency", {}).get("latency_unit_usd_per_sec", 0.001))
        return mu * (self.nominal_latency_ms / 1000.0) * unit

    def summary(self) -> dict[str, Any]:
        by_action: dict[str, float] = {}
        for e in self.events:
            by_action[e.action] = by_action.get(e.action, 0.0) + e.usd
        return {
            "total_usd": self.total_usd,
            "total_latency_ms": self.total_latency_ms,
            "nominal_latency_ms": self.nominal_latency_ms,
            "total_tokens": self.total_tokens,
            "n_events": len(self.events),
            "usd_by_action": by_action,
            "action_counts": {
                a: sum(1 for e in self.events if e.action == a)
                for a in sorted({e.action for e in self.events})
            },
        }
