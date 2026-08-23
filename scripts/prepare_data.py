#!/usr/bin/env python3
"""Prepare NQ + HotpotQA slices and a shared passage corpus.

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
    NQ_ANCHOR_SOURCE,
    add_hotpot_distractor_pool,
    build_synthetic_pilot,
    corpus_source_counts,
    hotpot_to_examples,
    nq_to_examples,
    save_processed,
)
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
    print("Loading NQ Open from HuggingFace...")
    nq = load_dataset("google-research-datasets/nq_open")

    n_train_h = int(data_cfg["train_hotpot"])
    n_train_n = int(data_cfg["train_nq"])
    n_eval_h = int(data_cfg["eval_hotpot"])
    n_eval_n = int(data_cfg["eval_nq"])

    hotpot_train_raw = list(hotpot["train"].select(range(min(n_train_h, len(hotpot["train"])))))
    hotpot_eval_raw = list(hotpot["validation"].select(range(min(n_eval_h, len(hotpot["validation"])))))

    nq_train_raw = list(nq["train"].select(range(min(n_train_n, len(nq["train"])))))
    # nq_open may use validation
    nq_val_split = "validation" if "validation" in nq else "train"
    nq_eval_raw = list(nq[nq_val_split].select(range(min(n_eval_n, len(nq[nq_val_split])))))

    h_train, h_passages = hotpot_to_examples(hotpot_train_raw, "train")
    h_eval, h_passages_e = hotpot_to_examples(hotpot_eval_raw, "eval")
    n_train = nq_to_examples(nq_train_raw, "train")
    n_eval = nq_to_examples(nq_eval_raw, "eval")

    # Deduplicate passages
    by_id = {p["passage_id"]: p for p in h_passages + h_passages_e}
    passages = list(by_id.values())

    # For NQ without Wikipedia dump in V1: inject answer-bearing synthetic passages
    # so single-hop questions remain solvable on the shared index (documented limitation).
    # Do not invent anchors for unused NQ rows — that would grow a trivial ceiling.
    for ex in n_train + n_eval:
        ans = ex["answer"]
        pid = f"nq_anchor_{ex['id']}"
        if pid not in by_id:
            text = f"According to reference sources, the answer is {ans}."
            # Also include a lightly natural sentence from the question
            text = f"{ex['question']} The answer is {ans}. {text}"
            p = {
                "passage_id": pid,
                "title": f"NQ anchor for {ex['id']}",
                "text": text,
                "source": NQ_ANCHOR_SOURCE,
                "is_gold_support": True,
            }
            by_id[pid] = p
            passages.append(p)
            ex["local_passage_ids"] = [pid]
            ex["supporting_titles"] = [p["title"]]

    pool_stats: dict[str, Any] = {
        "n_passages": len(passages),
        "n_slice_passages": len(passages),
        "n_distractors_added": 0,
        "n_distractor_rows_scanned": 0,
        "n_distractor_dups_skipped": 0,
        "target": pool_target,
        "hit_target": False,
        "note": "nq_open has no passages; pool is unused Hotpot contexts only",
    }
    if pool_target > len(passages):
        print(
            f"Adding unused Hotpot distractors toward {pool_target} passages "
            f"(slice currently {len(passages)}; nq_open contributes none)..."
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
        pool_stats["note"] = "nq_open has no passages; pool is unused Hotpot contexts only"
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
    run_diag = source != "synthetic" and not args.skip_recall_diag
    recall = _maybe_recall_diag(passages, eval_set, run_diag)
    meta = {
        "source": source,
        "seed": seed,
        "n_train": len(train),
        "n_eval": len(eval_set),
        "n_passages": len(passages),
        "n_gold_support": counts["n_gold_support"],
        "n_hotpot_slice": counts["n_hotpot_slice"],
        "n_hotpot_distractor": counts["n_hotpot_distractor"],
        "n_nq_anchor": counts["n_nq_anchor"],
        "distractor_pool_target": int(data_cfg.get("distractor_pool_target", 0) or 0),
        "distractor_pool": pool_stats,
        "train_target": {
            "hotpot_qa": int(data_cfg["train_hotpot"]),
            "natural_questions": int(data_cfg["train_nq"]),
        },
        "eval_target": {
            "hotpot_qa": int(data_cfg["eval_hotpot"]),
            "natural_questions": int(data_cfg["eval_nq"]),
        },
        "train_by_dataset": {
            "hotpot_qa": sum(1 for e in train if e["dataset"] == "hotpot_qa"),
            "natural_questions": sum(1 for e in train if e["dataset"] == "natural_questions"),
        },
        "eval_by_dataset": {
            "hotpot_qa": sum(1 for e in eval_set if e["dataset"] == "hotpot_qa"),
            "natural_questions": sum(1 for e in eval_set if e["dataset"] == "natural_questions"),
        },
        "nq_corpus": "answer_anchor_passages",
        "nq_ceiling_note": "NQ EM near 1.0 is the answer-anchor ceiling, not a policy result",
        "nq_pool_note": "nq_open has no passages; unused Hotpot contexts are the distractor pool",
        "ranking": "deferred_until_hotpot_n_approx_150_read_separately",
        "retrieval_diag": recall,
        "paths": paths,
    }
    (out_dir / "slice_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote processed slices to {out_dir}")


if __name__ == "__main__":
    main()
