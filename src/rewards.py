"""Multi-component reward for CA-RL-ARAG (Scope Memo V2 §6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metrics import compute_answer_metrics
from .utils import tokenize


@dataclass
class RewardBreakdown:
    q_ans: float = 0.0
    q_ground: float = 0.0
    q_cal: float = 0.0
    c_tok: float = 0.0
    c_ret: float = 0.0
    c_lat: float = 0.0
    p_hall: float = 0.0
    p_act: float = 0.0
    p_bud: float = 0.0
    unused_budget_bonus: float = 0.0
    reward: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        d = {
            "q_ans": self.q_ans,
            "q_ground": self.q_ground,
            "q_cal": self.q_cal,
            "c_tok": self.c_tok,
            "c_ret": self.c_ret,
            "c_lat": self.c_lat,
            "p_hall": self.p_hall,
            "p_act": self.p_act,
            "p_bud": self.p_bud,
            "unused_budget_bonus": self.unused_budget_bonus,
            "reward": self.reward,
        }
        d.update(self.components)
        return d


def grounded_reading_score(
    evidence: list[dict[str, Any]],
    supporting_titles: list[str] | None,
) -> float:
    """GRASP-style grounded reading when gold supporting titles exist."""
    if not supporting_titles:
        return 0.0
    if not evidence:
        return 0.0
    gold = {t.lower() for t in supporting_titles}
    retrieved = {str(e.get("title", "")).lower() for e in evidence}
    if not gold:
        return 0.0
    return len(gold & retrieved) / len(gold)


def claim_support_score(answer: str, evidence: list[dict[str, Any]], verify_out: dict[str, Any] | None) -> float:
    if verify_out is not None:
        return float(verify_out.get("support", 0.0))
    if not answer or answer.upper() == "ABSTAIN":
        return 0.0
    e_toks = set()
    for e in evidence:
        e_toks |= set(tokenize(f"{e.get('title', '')} {e.get('text', '')}"))
    a_toks = set(tokenize(answer))
    if not a_toks:
        return 0.0
    return len(a_toks & e_toks) / len(a_toks)


def calibration_score(pred: str, gold: str | list[str], abstained: bool, evidence: list[dict[str, Any]]) -> float:
    """Reward justified abstain; penalize confident wrong / unjustified refuse."""
    from .utils import exact_match

    has_evidence = len(evidence) > 0
    correct = exact_match(pred, gold) >= 1.0
    if abstained:
        if correct:
            return -0.5  # refused a solvable answer
        # Justified if little evidence or answer would be wrong
        return 0.6 if (not has_evidence or not correct) else -0.2
    if correct:
        return 0.3
    # Confident wrong
    return -0.4 if has_evidence else -0.2


class RewardComputer:
    def __init__(self, weights: dict[str, Any], price_card: dict[str, Any] | None = None):
        self.w = weights
        self.price_card = price_card or {}

    def compute(
        self,
        *,
        pred: str,
        gold: str | list[str],
        evidence: list[dict[str, Any]],
        supporting_titles: list[str] | None,
        verify_out: dict[str, Any] | None,
        cost_summary: dict[str, Any],
        n_actions: int,
        verified: bool,
        abstained: bool,
        budget: dict[str, Any],
    ) -> RewardBreakdown:
        ans_metrics = compute_answer_metrics(pred, gold, abstained=abstained)
        # Primary answer signal: blend EM and F1 (EM strict, F1 partial) like GRASP R_A ~ F1
        q_ans = 0.5 * ans_metrics["em"] + 0.5 * ans_metrics["f1"]

        gold_ground = grounded_reading_score(evidence, supporting_titles)
        claim_ground = claim_support_score(pred, evidence, verify_out)
        q_ground = 0.5 * claim_ground + 0.5 * gold_ground if supporting_titles else claim_ground

        q_cal = calibration_score(pred, gold, abstained, evidence)

        # Costs
        usd_by = cost_summary.get("usd_by_action", {})
        c_tok = float(
            usd_by.get("rewrite", 0.0)
            + usd_by.get("stop", 0.0)
            + usd_by.get("generate", 0.0)
        )
        # Attribute remaining LLM-ish events already folded; also include token total scaled
        c_ret = float(usd_by.get("retrieve", 0.0) + usd_by.get("rerank", 0.0) + usd_by.get("verify", 0.0))
        # Prefer tracker totals if action split incomplete
        total_usd = float(cost_summary.get("total_usd", c_tok + c_ret))
        if c_tok + c_ret <= 0:
            c_tok = 0.6 * total_usd
            c_ret = 0.4 * total_usd

        mu = float(self.w.get("mu_latency", 0.5))
        unit = float(self.price_card.get("latency", {}).get("latency_unit_usd_per_sec", 0.001))
        c_lat = mu * (float(cost_summary.get("total_latency_ms", 0.0)) / 1000.0) * unit

        # Hallucination penalty
        unsupported = max(0.0, 1.0 - claim_ground) if not abstained else 0.0
        contrad = float((verify_out or {}).get("contradiction", 0.0))
        hall = unsupported * 0.7 + contrad * 0.3
        hall_w = float(self.w.get("hall_weight", 0.5))
        if (not verified) and n_actions > 1:
            hall_w += float(self.w.get("verify_ignored_extra", 0.15))
        p_hall = hall_w * hall

        p_act = float(self.w.get("act_penalty", 0.02)) * max(0, n_actions - 1)

        # Budget violation
        p_bud = 0.0
        if budget.get("violated"):
            p_bud = float(self.w.get("bud_penalty", 1.0))

        alpha = float(self.w.get("alpha", 1.0))
        beta = float(self.w.get("beta", 0.4))
        gamma = float(self.w.get("gamma", 0.15))
        lam = float(self.w.get("lambda_cost", 2.0))

        quality = alpha * q_ans + beta * q_ground + gamma * q_cal
        cost_term = lam * (c_tok + c_ret) + c_lat

        # Quality-gated unused budget bonus (GRASP R_E pattern)
        unused_bonus = 0.0
        gate = float(self.w.get("quality_gate", 0.5))
        if q_ans >= gate:
            max_usd = float(budget.get("max_usd", 0.05)) or 0.05
            frac_left = max(0.0, 1.0 - (total_usd / max_usd))
            unused_bonus = float(self.w.get("unused_budget_bonus", 0.1)) * frac_left

        reward = quality - cost_term - p_hall - p_act - p_bud + unused_bonus

        rb = RewardBreakdown(
            q_ans=q_ans,
            q_ground=q_ground,
            q_cal=q_cal,
            c_tok=c_tok,
            c_ret=c_ret,
            c_lat=c_lat,
            p_hall=p_hall,
            p_act=p_act,
            p_bud=p_bud,
            unused_budget_bonus=unused_bonus,
            reward=reward,
            components={"em": ans_metrics["em"], "f1": ans_metrics["f1"]},
        )
        return rb
