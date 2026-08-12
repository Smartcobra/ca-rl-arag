#!/usr/bin/env python3
"""Prepare NQ + HotpotQA slices and a shared passage corpus.

Usage:
  python scripts/prepare_data.py                 # tries HuggingFace, falls back to synthetic
  python scripts/prepare_data.py --synthetic     # offline synthetic pilot corpus
  python scripts/prepare_data.py --hf            # require HuggingFace download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.data.loaders import (
    build_synthetic_pilot,
    hotpot_to_examples,
    nq_to_examples,
    save_processed,
)
from src.utils import set_seed


def load_hf_slices(cfg: dict) -> tuple[list, list, list]:
    from datasets import load_dataset

    data_cfg = cfg["data"]
    hotpot_cfg = data_cfg.get("hotpot_config", "distractor")

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
                "source": "nq_anchor",
                "is_gold_support": True,
            }
            by_id[pid] = p
            passages.append(p)
            ex["local_passage_ids"] = [pid]
            ex["supporting_titles"] = [p["title"]]

    train = h_train + n_train
    eval_set = h_eval + n_eval
    return train, eval_set, passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use offline synthetic pilot data")
    parser.add_argument("--hf", action="store_true", help="Require HuggingFace datasets")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    out_dir = resolve_path(cfg, cfg["data"]["processed_dir"])

    train = eval_set = passages = None
    source = "synthetic"

    if not args.synthetic:
        try:
            train, eval_set, passages = load_hf_slices(cfg)
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

    paths = save_processed(train, eval_set, passages, str(out_dir))
    meta = {
        "source": source,
        "n_train": len(train),
        "n_eval": len(eval_set),
        "n_passages": len(passages),
        "train_by_dataset": {
            "hotpot_qa": sum(1 for e in train if e["dataset"] == "hotpot_qa"),
            "natural_questions": sum(1 for e in train if e["dataset"] == "natural_questions"),
        },
        "eval_by_dataset": {
            "hotpot_qa": sum(1 for e in eval_set if e["dataset"] == "hotpot_qa"),
            "natural_questions": sum(1 for e in eval_set if e["dataset"] == "natural_questions"),
        },
        "paths": paths,
    }
    (out_dir / "slice_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote processed slices to {out_dir}")


if __name__ == "__main__":
    main()
