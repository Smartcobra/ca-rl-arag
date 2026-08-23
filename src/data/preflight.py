"""Refuse GPU ranking runs when the eval slice or corpus is still the tiny V1 index."""

from __future__ import annotations

from typing import Any

from ..metrics import counts_by_dataset


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
    got_n = int(counts.get("natural_questions") or 0)

    if require_slice and want_total:
        if len(examples) != want_total:
            errors.append(
                f"{split} file has {len(examples)} examples; expected {want_total} "
                f"({want_h} Hotpot + {want_n} NQ). Re-run: python scripts/prepare_data.py --hf"
            )
        if want_h and got_h != want_h:
            errors.append(f"{split} Hotpot count is {got_h}, expected {want_h}.")
        if want_n and got_n != want_n:
            errors.append(f"{split} NQ count is {got_n}, expected {want_n}.")

    n_passages = len(corpus)
    if min_passages > 0 and n_passages < min_passages:
        errors.append(
            f"corpus has {n_passages} passages; ranking runs need >= {min_passages}. "
            "The small index saturates BM25 (gold almost always in one shot). "
            "Re-run: python scripts/prepare_data.py --hf"
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
