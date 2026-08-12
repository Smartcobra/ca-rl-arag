"""Verification: NLI-style claim support (LOCKED for V1).

Decision (Scope Memo review comment):
  Use an NLI approach for verification — NOT an LLM-as-judge verifier — so the
  implementation stays consistent across all experiments.

Backends:
  - lexical_nli (default): reproducible, no GPU; approximates entailment via
    claim–evidence token support / contradiction cues.
  - neural_nli (optional): same interface using a cross-encoder NLI model when
    transformers + the model weights are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils import tokenize


CONTRADICTION_CUES = {
    "not",
    "never",
    "no",
    "unlike",
    "incorrect",
    "false",
    "rather",
    "instead",
}


@dataclass
class VerifyResult:
    support: float
    contradiction: float
    uncertainty: float
    label: str  # support | contradiction | neutral
    claims: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "contradiction": self.contradiction,
            "uncertainty": self.uncertainty,
            "label": self.label,
            "claims": self.claims,
        }


def _split_claims(answer: str) -> list[str]:
    answer = answer.strip()
    if not answer or answer.upper() == "ABSTAIN":
        return []
    # Short answers are a single claim; longer answers split on sentence boundaries.
    parts = [p.strip() for p in answer.replace(";", ".").split(".") if p.strip()]
    return parts or [answer]


class LexicalNLIVerifier:
    def __init__(self, support_threshold: float = 0.45, contradiction_threshold: float = 0.55):
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold

    def verify(self, answer: str, evidence: list[dict[str, Any]], question: str = "") -> VerifyResult:
        claims = _split_claims(answer)
        if not claims:
            return VerifyResult(
                support=0.0,
                contradiction=0.0,
                uncertainty=1.0,
                label="neutral",
                claims=[],
            )

        evidence_text = " ".join(f"{e.get('title', '')} {e.get('text', '')}" for e in evidence)
        e_toks = set(tokenize(evidence_text))
        claim_rows = []
        supports = []
        contradictions = []

        for claim in claims:
            c_toks = set(tokenize(claim))
            if not c_toks:
                claim_rows.append({"claim": claim, "support": 0.0, "contradiction": 0.0, "label": "neutral"})
                supports.append(0.0)
                contradictions.append(0.0)
                continue
            overlap = len(c_toks & e_toks) / len(c_toks)
            cue_hit = 1.0 if (c_toks & CONTRADICTION_CUES) and overlap < 0.5 else 0.0
            # Weak contradiction if evidence contains opposite capital-city style conflict is hard;
            # use low-overlap + negation cues.
            contra = min(1.0, cue_hit * 0.8 + (0.3 if overlap < 0.15 and len(evidence) > 0 else 0.0))
            if overlap >= self.support_threshold:
                label = "support"
            elif contra >= self.contradiction_threshold:
                label = "contradiction"
            else:
                label = "neutral"
            claim_rows.append({"claim": claim, "support": overlap, "contradiction": contra, "label": label})
            supports.append(overlap)
            contradictions.append(contra)

        support = sum(supports) / len(supports)
        contradiction = max(contradictions) if contradictions else 0.0
        uncertainty = max(0.0, 1.0 - support - 0.5 * contradiction)
        if support >= self.support_threshold and support >= contradiction:
            label = "support"
        elif contradiction >= self.contradiction_threshold:
            label = "contradiction"
        else:
            label = "neutral"
        return VerifyResult(support, contradiction, uncertainty, label, claim_rows)


class NeuralNLIVerifier:
    """Optional transformers-backed NLI. Falls back to lexical if unavailable."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        support_threshold: float = 0.45,
        contradiction_threshold: float = 0.55,
    ):
        self.model_name = model_name
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self._fallback = LexicalNLIVerifier(support_threshold, contradiction_threshold)
        self._model = None
        self._available = False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(model_name)
            self._available = True
        except Exception:
            self._available = False

    def verify(self, answer: str, evidence: list[dict[str, Any]], question: str = "") -> VerifyResult:
        if not self._available or self._model is None:
            return self._fallback.verify(answer, evidence, question=question)

        claims = _split_claims(answer)
        if not claims:
            return VerifyResult(0.0, 0.0, 1.0, "neutral", [])

        premises = [f"{e.get('title', '')}. {e.get('text', '')}" for e in evidence] or [""]
        claim_rows = []
        supports = []
        contradictions = []
        # Labels often: contradiction / entailment / neutral — map by model output
        for claim in claims:
            pairs = [(p, claim) for p in premises[:5]]
            scores = self._model.predict(pairs)
            # Handle both 3-logit and single-score models loosely
            best_entail = 0.0
            best_contra = 0.0
            import numpy as np

            arr = np.array(scores)
            if arr.ndim == 1:
                best_entail = float(max(arr))
                best_contra = float(max(0.0, 1.0 - best_entail))
            else:
                # assume [contradiction, entailment, neutral] or similar — take max entail-ish
                # DeBERTa NLI often: 0=contradiction, 1=entailment, 2=neutral
                if arr.shape[1] >= 3:
                    best_contra = float(arr[:, 0].max())
                    best_entail = float(arr[:, 1].max())
                else:
                    best_entail = float(arr.max())
            if best_entail >= self.support_threshold:
                label = "support"
            elif best_contra >= self.contradiction_threshold:
                label = "contradiction"
            else:
                label = "neutral"
            claim_rows.append(
                {"claim": claim, "support": best_entail, "contradiction": best_contra, "label": label}
            )
            supports.append(best_entail)
            contradictions.append(best_contra)

        support = sum(supports) / len(supports)
        contradiction = max(contradictions) if contradictions else 0.0
        uncertainty = max(0.0, 1.0 - support - 0.5 * contradiction)
        if support >= self.support_threshold and support >= contradiction:
            label = "support"
        elif contradiction >= self.contradiction_threshold:
            label = "contradiction"
        else:
            label = "neutral"
        return VerifyResult(support, contradiction, uncertainty, label, claim_rows)


def build_verifier(cfg: dict[str, Any]):
    vcfg = cfg.get("verification", {})
    backend = vcfg.get("backend", "lexical_nli")
    st = float(vcfg.get("support_threshold", 0.45))
    ct = float(vcfg.get("contradiction_threshold", 0.55))
    if backend == "lexical_nli":
        return LexicalNLIVerifier(st, ct)
    if backend == "neural_nli":
        return NeuralNLIVerifier(vcfg.get("neural_model", "cross-encoder/nli-deberta-v3-base"), st, ct)
    raise ValueError(f"Unknown verification backend: {backend}. V1 locks NLI (lexical_nli|neural_nli).")
