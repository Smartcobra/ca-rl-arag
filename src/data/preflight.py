"""Refuse GPU ranking runs when the eval slice or corpus is still the tiny V1 index."""

from __future__ import annotations

from typing import Any

from ..metrics import counts_by_dataset
from .loaders import NQ_ANCHOR_SOURCE
from .wiki_passages import (
    SINGLE_HOP_DATASETS,
    distinct_single_hop_gold_articles,
    min_distinct_gold_articles,
)


def ranking_data_errors(
    cfg: dict[str, Any],
    examples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    *,
    split: str = "eval",
) -> list[str]:
    """Return human-readable failures. Empty means the ranking files are ready.

    Checks the on-disk file (call before ``stratified_limit``). ``--limit`` must
    not hide a 2k-passage corpus or a non-300 eval file.
    """
    data = cfg.get("data") or {}
    min_passages = int(data.get("min_corpus_passages") or 0)
    require_slice = bool(data.get("require_ranking_slice", min_passages > 0))
    if min_passages <= 0 and not require_slice:
        return []

    errors: list[str] = []
    want_h = int(data.get(f"{split}_hotpot") or 0)
    want_n = int(data.get(f"{split}_nq") or 0)
    want_total = want_h + want_n
    counts = counts_by_dataset(examples)
    got_h = int(counts.get("hotpot_qa") or 0)
    got_n = sum(int(counts.get(name) or 0) for name in SINGLE_HOP_DATASETS)
    hop_present = [name for name in SINGLE_HOP_DATASETS if counts.get(name)]

    if require_slice and want_total:
        if len(examples) != want_total:
            errors.append(
                f"{split} file has {len(examples)} examples; expected {want_total} "
                f"({want_h} Hotpot + {want_n} single-hop). Re-run: python scripts/prepare_data.py --hf"
            )
        if want_h and got_h != want_h:
            errors.append(f"{split} Hotpot count is {got_h}, expected {want_h}.")
        if want_n and got_n != want_n:
            errors.append(
                f"{split} single-hop count is {got_n}, expected {want_n} "
                f"(natural_questions / trivia_qa / squad)."
            )
        if want_n and len(hop_present) > 1:
            errors.append(
                f"{split} has mixed single-hop datasets {hop_present}; expected one of {SINGLE_HOP_DATASETS}."
            )
        if want_n and got_n == want_n:
            n_titles = distinct_single_hop_gold_articles(examples)
            need_titles = min_distinct_gold_articles(want_n)
            if n_titles < need_titles:
                errors.append(
                    f"{split} single-hop has {n_titles} distinct gold articles; "
                    f"need >= {need_titles} for {want_n} questions "
                    f"(~30 per 150). A SQuAD prefix is a single-topic slice. "
                    "Re-run: python scripts/prepare_data.py --hf"
                )

    n_passages = len(corpus)
    if min_passages > 0 and n_passages < min_passages:
        errors.append(
            f"corpus has {n_passages} passages; ranking runs need >= {min_passages}. "
            "The small index saturates BM25 (gold almost always in one shot). "
            "Re-run: python scripts/prepare_data.py --hf"
        )

    n_anchor = sum(
        1
        for p in corpus
        if str(p.get("source") or "") == NQ_ANCHOR_SOURCE
        or str(p.get("title") or "").startswith("NQ anchor")
    )
    if n_anchor:
        errors.append(
            f"corpus still has {n_anchor} NQ answer-anchor passages (label leakage). "
            "Re-run: python scripts/prepare_data.py --hf  "
            "(uses Tevatron/wikipedia-nq DPR Wikipedia passages; never plants "
            "'The answer is {gold}')."
        )
    return errors


def assert_ranking_data(
    cfg: dict[str, Any],
    examples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    *,
    split: str = "eval",
) -> None:
    errors = ranking_data_errors(cfg, examples, corpus, split=split)
    if errors:
        raise SystemExit("Ranking data check failed:\n- " + "\n- ".join(errors))
    data = cfg.get("data") or {}
    min_passages = int(data.get("min_corpus_passages") or 0)
    counts = counts_by_dataset(examples)
    print(
        f"Ranking data check OK: {split}={len(examples)} {counts} "
        f"corpus={len(corpus)}"
        + (f" (>= {min_passages})" if min_passages else "")
    )
