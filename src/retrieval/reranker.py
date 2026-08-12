"""Lightweight lexical reranker (V1-stable; swap for cross-encoder later)."""

from __future__ import annotations

from typing import Any

from ..utils import tokenize


def _overlap_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    d = tokenize(text)
    if not q or not d:
        return 0.0
    # soft TF overlap
    hits = sum(1 for t in d if t in q)
    return hits / (len(d) ** 0.5) * (len(q & set(d)) / len(q))


class LexicalReranker:
    """Re-scores a candidate pool with query–document lexical overlap.

    Kept intentionally simple for Milestone 2 stability. Interface matches a
    future CrossEncoderReranker so experiments stay consistent.
    """

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
        scored = []
        for c in candidates:
            text = f"{c.get('title', '')}. {c.get('text', '')}"
            lexical = _overlap_score(query, text)
            base = float(c.get("score", 0.0))
            # Blend retrieval score with lexical re-score
            new_score = 0.35 * base + 0.65 * lexical
            row = dict(c)
            row["score"] = float(new_score)
            row["rerank_score"] = float(lexical)
            scored.append(row)
        scored.sort(key=lambda x: x["score"], reverse=True)
        out = scored[:top_n]
        for i, row in enumerate(out):
            row["rank"] = i
        return out
