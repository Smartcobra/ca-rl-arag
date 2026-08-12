"""BM25 retriever over a passage corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from ..utils import tokenize


@dataclass
class Passage:
    passage_id: str
    title: str
    text: str
    source: str = ""
    meta: dict[str, Any] | None = None

    @property
    def content(self) -> str:
        return f"{self.title}. {self.text}".strip()


class BM25Retriever:
    def __init__(self, passages: list[dict[str, Any]]):
        self.passages = [
            Passage(
                passage_id=p["passage_id"],
                title=p.get("title", ""),
                text=p.get("text", ""),
                source=p.get("source", ""),
                meta={k: v for k, v in p.items() if k not in {"passage_id", "title", "text", "source"}},
            )
            for p in passages
        ]
        self._tokens = [tokenize(p.content) for p in self.passages]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        self._id_to_idx = {p.passage_id: i for i, p in enumerate(self.passages)}

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._bm25 or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        k = min(top_k, len(self.passages))
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for rank, i in enumerate(idxs):
            p = self.passages[i]
            results.append(
                {
                    "passage_id": p.passage_id,
                    "title": p.title,
                    "text": p.text,
                    "score": float(scores[i]),
                    "rank": rank,
                    "source": p.source,
                    "meta": p.meta or {},
                }
            )
        return results

    def get(self, passage_id: str) -> Passage | None:
        i = self._id_to_idx.get(passage_id)
        return self.passages[i] if i is not None else None
