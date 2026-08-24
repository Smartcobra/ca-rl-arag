#!/usr/bin/env python3
"""Prepare NQ + HotpotQA slices and a shared passage corpus.

NQ evidence is DPR Wikipedia 100-word passages (Tevatron/wikipedia-nq),
not answer-anchor leakage. TriviaQA / SQuAD are fallbacks if NQ is too heavy.

Usage:
  python scripts/prepare_data.py                 # tries HuggingFace, falls back to synthetic
  python scripts/prepare_data.py --synthetic     # offline synthetic pilot corpus
  python scripts/prepare_data.py --hf            # require HuggingFace download
  python scripts/prepare_data.py --hf --distractor-pool 80000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.data.loaders import (
    add_hotpot_distractor_pool,
    build_synthetic_pilot,
    corpus_source_counts,
    hotpot_to_examples,
    save_processed,
)
from src.data.wiki_passages import (
    count_leaky_anchors,
    load_single_hop_with_real_passages,
    merge_passages,
)
from src.metrics import counts_by_dataset
from src.utils import set_seed


def _iter_dataset_range(ds: Any, start: int, end: int, chunk: int = 1000) -> Iterator[dict[str, Any]]:
    start = max(0, int(start))
    end = min(int(end), len(ds))
    for i in range(start, end, chunk):
        j = min(i + chunk, end)
        batch = ds.select(range(i, j))
        for row in batch:
            yield row


def iter_unused_hotpot_rows(
    hotpot: Any,
    n_train_used: int,
    n_eval_used: int,
    chunk: int = 1000,
) -> Iterator[dict[str, Any]]:
    """Yield Hotpot rows that are not in the locked train/eval prefixes.

    Slice uses train[:n_train_used] and validation[:n_eval_used]. The pool
    starts after those prefixes so golds stay attached only to locked questions.
    """
    train = hotpot["train"]
    val = hotpot["validation"]
    yield from _iter_dataset_range(train, n_train_used, len(train), chunk)
    yield from _iter_dataset_range(val, n_eval_used, len(val), chunk)


def load_hf_slices(cfg: dict) -> tuple[list, list, list, dict[str, Any]]:
    from datasets import load_dataset

    data_cfg = cfg["data"]
    hotpot_cfg = data_cfg.get("hotpot_config", "distractor")
    pool_target = int(data_cfg.get("distractor_pool_target", 0) or 0)

    print("Loading HotpotQA (distractor) from HuggingFace...")
    hotpot = load_dataset("hotpotqa/hotpot_qa", hotpot_cfg)

    n_train_h = int(data_cfg["train_hotpot"])
    n_train_n = int(data_cfg["train_nq"])
    n_eval_h = int(data_cfg["eval_hotpot"])
    n_eval_n = int(data_cfg["eval_nq"])
    nq_negs = int(data_cfg.get("nq_negatives_per_query", 8) or 0)
    preferred_nq = data_cfg.get("nq_passage_dataset") or "Tevatron/wikipedia-nq"

    hotpot_train_raw = list(hotpot["train"].select(range(min(n_train_h, len(hotpot["train"])))))
    hotpot_eval_raw = list(hotpot["validation"].select(range(min(n_eval_h, len(hotpot["validation"])))))

    h_train, h_passages = hotpot_to_examples(hotpot_train_raw, "train")
    h_eval, h_passages_e = hotpot_to_examples(hotpot_eval_raw, "eval")

    n_train, n_eval, n_passages, nq_stats = load_single_hop_with_real_passages(
        n_train_n,
        n_eval_n,
        negatives_per_query=nq_negs,
        preferred_hf_id=str(preferred_nq),
    )

    passages = merge_passages(h_passages, h_passages_e, n_passages)
    n_leaky = count_leaky_anchors(passages)
    if n_leaky:
        raise RuntimeError(
            f"Refusing to write {n_leaky} NQ answer-anchor passages. "
            "Single-hop evidence must be real Wikipedia text."
        )

    pool_stats: dict[str, Any] = {
        "n_passages": len(passages),
        "n_slice_passages": len(passages),
        "n_distractors_added": 0,
        "n_distractor_rows_scanned": 0,
        "n_distractor_dups_skipped": 0,
        "target": pool_target,
        "hit_target": False,
        "note": "unused Hotpot contexts plus DPR Wikipedia golds/negatives",
        "single_hop": nq_stats,
    }
    if pool_target > len(passages):
        print(
            f"Adding unused Hotpot distractors toward {pool_target} passages "
            f"(slice currently {len(passages)}; NQ uses {nq_stats.get('label')})..."
        )
        last_print = 0

        def _progress_rows() -> Iterator[dict[str, Any]]:
            nonlocal last_print
            for n_rows, row in enumerate(iter_unused_hotpot_rows(hotpot, n_train_h, n_eval_h), start=1):
                if n_rows - last_print >= 2000:
                    print(f"  scanned {n_rows} unused Hotpot rows...", flush=True)
                    last_print = n_rows
                yield row

        passages, pool_stats = add_hotpot_distractor_pool(passages, _progress_rows(), pool_target)
        pool_stats["note"] = "unused Hotpot contexts plus DPR Wikipedia golds/negatives"
        pool_stats["single_hop"] = nq_stats
        print(
            f"Distractor pool: +{pool_stats['n_distractors_added']} passages "
            f"from {pool_stats['n_distractor_rows_scanned']} unused rows "
            f"(corpus={pool_stats['n_passages']}, target={pool_target})."
        )

    train = h_train + n_train
    eval_set = h_eval + n_eval
    return train, eval_set, passages, pool_stats


def _maybe_recall_diag(passages: list, eval_set: list, enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    print(f"Computing BM25 gold recall@k on {len(passages)} passages / {len(eval_set)} eval...")
    from src.retrieval.diagnostics import gold_recall_by_dataset

    return gold_recall_by_dataset(passages, eval_set)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use offline synthetic pilot data")
    parser.add_argument("--hf", action="store_true", help="Require HuggingFace datasets")
    parser.add_argument(
        "--distractor-pool",
        type=int,
        default=None,
        help="Target corpus size including golds (default: data.distractor_pool_target). 0 disables.",
    )
    parser.add_argument(
        "--skip-recall-diag",
        action="store_true",
        help="Skip BM25 gold recall@k after writing the corpus.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.distractor_pool is not None:
        cfg.setdefault("data", {})["distractor_pool_target"] = int(args.distractor_pool)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    out_dir = resolve_path(cfg, cfg["data"]["processed_dir"])

    train = eval_set = passages = None
    source = "synthetic"
    pool_stats: dict[str, Any] = {}

    if not args.synthetic:
        try:
            train, eval_set, passages, pool_stats = load_hf_slices(cfg)
            source = "huggingface_nq_hotpot"
        except Exception as e:
            if args.hf:
                raise
            print(f"HuggingFace load failed ({e}); falling back to synthetic pilot corpus.")

    if train is None:
        data_cfg = cfg["data"]
        train, eval_set, passages = build_synthetic_pilot(
            n_train_hotpot=int(data_cfg["train_hotpot"]),
            n_train_nq=int(data_cfg["train_nq"]),
            n_eval_hotpot=int(data_cfg["eval_hotpot"]),
            n_eval_nq=int(data_cfg["eval_nq"]),
        )
        source = "synthetic"
        pool_stats = {
            "n_passages": len(passages),
            "n_slice_passages": len(passages),
            "n_distractors_added": 0,
            "target": 0,
            "note": "synthetic mode does not add an HF distractor pool",
        }

    paths = save_processed(train, eval_set, passages, str(out_dir))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    data_cfg = cfg["data"]
    counts = corpus_source_counts(passages)
    if source != "synthetic" and counts.get("n_nq_anchor", 0):
        raise RuntimeError(
            f"HF corpus still contains {counts['n_nq_anchor']} answer-anchor passages. "
            "This is label leakage; fix wiki_passages loading."
        )
    run_diag = source != "synthetic" and not args.skip_recall_diag
    recall = _maybe_recall_diag(passages, eval_set, run_diag)
    single_hop = (pool_stats or {}).get("single_hop") or {}
    hop_name = str(single_hop.get("dataset") or "natural_questions")
    meta = {
        "source": source,
        "seed": seed,
        "n_train": len(train),
        "n_eval": len(eval_set),
        "n_passages": len(passages),
        "n_gold_support": counts["n_gold_support"],
        "n_hotpot_slice": counts["n_hotpot_slice"],
        "n_hotpot_distractor": counts["n_hotpot_distractor"],
        "n_nq_wiki": counts.get("n_nq_wiki", 0),
        "n_nq_wiki_neg": counts.get("n_nq_wiki_neg", 0),
        "n_nq_anchor": counts.get("n_nq_anchor", 0),
        "distractor_pool_target": int(data_cfg.get("distractor_pool_target", 0) or 0),
        "distractor_pool": pool_stats,
        "train_target": {
            "hotpot_qa": int(data_cfg["train_hotpot"]),
            hop_name: int(data_cfg["train_nq"]),
        },
        "eval_target": {
            "hotpot_qa": int(data_cfg["eval_hotpot"]),
            hop_name: int(data_cfg["eval_nq"]),
        },
        "train_by_dataset": counts_by_dataset(train),
        "eval_by_dataset": counts_by_dataset(eval_set),
        "single_hop_dataset": hop_name,
        "nq_corpus": single_hop.get("nq_corpus")
        if source != "synthetic"
        else "synthetic",
        "nq_hf_dataset": single_hop.get("hf_id"),
        "nq_passage_note": single_hop.get("label")
        or (
            "DPR Wikipedia 100-word passages; answer-anchors are forbidden"
            if source != "synthetic"
            else "synthetic closed corpus"
        ),
        "ranking": "hotpot_and_nq_after_real_wiki_passages",
        "retrieval_diag": recall,
        "paths": paths,
    }
    (out_dir / "slice_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote processed slices to {out_dir}")


if __name__ == "__main__":
    main()
