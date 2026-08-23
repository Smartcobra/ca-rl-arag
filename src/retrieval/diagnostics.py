"""Retrieval diagnostics: gold recall@k on the shared BM25 index."""

from __future__ import annotations

from typing import Any, Iterable

from .bm25 import BM25Retriever


def gold_passage_ids(example: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    """Passage IDs that are gold for this example (not the whole local context)."""
    titles = {t for t in (example.get("supporting_titles") or []) if t}
    ids: list[str] = []
    seen: set[str] = set()
    for pid in example.get("local_passage_ids") or []:
        if pid in seen:
            continue
        passage = by_id.get(pid)
        if passage is None:
            continue
        if passage.get("is_gold_support") or passage.get("title") in titles:
            seen.add(pid)
            ids.append(pid)
    return ids


def _recall_with_retriever(
    retriever: BM25Retriever,
    by_id: dict[str, dict[str, Any]],
    examples: Iterable[dict[str, Any]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    scored: list[tuple[dict[str, Any], set[str]]] = []
    for ex in examples:
        gids = gold_passage_ids(ex, by_id)
        if gids:
            scored.append((ex, set(gids)))
    n = len(scored)
    out: dict[str, Any] = {f"recall@{k}": None for k in ks}
    out.update({"n_examples": n, "n_miss_at_5": 0, "n_corpus": len(retriever)})
    if n == 0:
        return out

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    miss_at_5 = 0
    for ex, golds in scored:
        ranked_ids = [r["passage_id"] for r in retriever.search(ex["question"], top_k=max_k)]
        for k in ks:
            if golds & set(ranked_ids[:k]):
                hits[k] += 1
        if 5 in ks and not (golds & set(ranked_ids[:5])):
            miss_at_5 += 1

    out["n_miss_at_5"] = miss_at_5
    for k in ks:
        out[f"recall@{k}"] = hits[k] / n
    return out


def gold_recall(
    passages: list[dict[str, Any]],
    examples: Iterable[dict[str, Any]],
    ks: tuple[int, ...] = (1, 5, 20),
    retriever: BM25Retriever | None = None,
) -> dict[str, Any]:
    """Fraction of examples with at least one gold passage in BM25 top-k."""
    by_id = {p["passage_id"]: p for p in passages}
    index = retriever or BM25Retriever(passages)
    return _recall_with_retriever(index, by_id, examples, ks)


def gold_recall_by_dataset(
    passages: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 5, 20),
) -> dict[str, Any]:
    retriever = BM25Retriever(passages)
    overall = gold_recall(passages, examples, ks=ks, retriever=retriever)
    by_ds: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        grouped.setdefault(str(ex.get("dataset") or "unknown"), []).append(ex)
    for ds, rows in grouped.items():
        by_ds[ds] = gold_recall(passages, rows, ks=ks, retriever=retriever)
    return {"overall": overall, "by_dataset": by_ds}
