#!/usr/bin/env python3
"""Build the v3 train and validation slices. Does not rewrite the locked 300.

Train and validation are prefixes of the HuggingFace *train* split:
  Hotpot train[:train_hotpot], then the next valid_hotpot rows
  NQ train[:train_nq], then the next valid_nq rows

New gold passages are merged into the existing corpus. Gold wins on a
duplicate passage id. eval_slice.jsonl is read only to check that its ids
still match tests/fixtures/locked_eval_ids.json.

Usage:
  python scripts/build_train_valid.py --config configs/frontier_v3.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.data.id_lock import assert_disjoint, assert_ids_unchanged, load_locked_eval_ids
from src.data.loaders import hotpot_to_examples
from src.data.preflight import ranking_data_errors
from src.data.train_valid import split_prefix, tag_split
from src.data.wiki_passages import count_leaky_anchors, load_single_hop_train_prefix, merge_passages
from src.metrics import counts_by_dataset
from src.utils import read_jsonl, set_seed, write_jsonl


def _counts(data_cfg: dict, split: str) -> tuple[int, int]:
    return int(data_cfg[f"{split}_hotpot"]), int(data_cfg[f"{split}_nq"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/frontier_v3.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))

    train_h, train_n = _counts(data_cfg, "train")
    valid_h, valid_n = _counts(data_cfg, "valid")
    if not 150 <= valid_h + valid_n <= 200:
        raise SystemExit(
            f"valid slice is {valid_h + valid_n} questions; keep it between 150 and 200."
        )

    eval_path = resolve_path(cfg, data_cfg["eval_file"])
    corpus_path = resolve_path(cfg, data_cfg["corpus_file"])
    train_path = resolve_path(cfg, data_cfg["train_file"])
    valid_rel = data_cfg.get("valid_file")
    if not valid_rel:
        raise SystemExit("config data.valid_file is required")
    valid_path = resolve_path(cfg, valid_rel)
    if "eval" in valid_path.name.lower() or "eval" in train_path.name.lower():
        raise SystemExit("train and valid filenames must not contain 'eval'")
    if not corpus_path.exists():
        raise SystemExit(
            f"Missing corpus {corpus_path}. Build it once with prepare_data.py --hf. "
            "This script only merges new gold passages into that file."
        )
    if not eval_path.exists():
        raise SystemExit(f"Missing locked eval file {eval_path}")

    locked = load_locked_eval_ids()
    eval_ids = [str(row.get("id")) for row in read_jsonl(eval_path)]
    assert_ids_unchanged(eval_ids, locked, eval_path.name)

    from datasets import load_dataset

    hotpot_cfg = data_cfg.get("hotpot_config", "distractor")
    print("Loading HotpotQA train split (validation split is the locked eval; not read)...")
    hotpot = load_dataset("hotpotqa/hotpot_qa", hotpot_cfg, trust_remote_code=True)
    need_h = train_h + valid_h
    if len(hotpot["train"]) < need_h:
        raise SystemExit(f"Hotpot train split has {len(hotpot['train'])} rows; need {need_h}")
    hotpot_raw = list(hotpot["train"].select(range(need_h)))
    hotpot_examples, hotpot_passages = hotpot_to_examples(hotpot_raw, "train")
    hotpot_train, hotpot_valid = split_prefix(hotpot_examples, train_h, valid_h)
    # Passages were built for the whole prefix. merge_passages dedups by id.

    nq_negs = int(data_cfg.get("nq_negatives_per_query", 8) or 0)
    preferred = str(data_cfg.get("nq_passage_dataset") or "Tevatron/wikipedia-nq")
    nq_examples, nq_passages, nq_stats = load_single_hop_train_prefix(
        train_n + valid_n,
        negatives_per_query=nq_negs,
        preferred_hf_id=preferred,
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    nq_train, nq_valid = split_prefix(nq_examples, train_n, valid_n)

    train_rows = tag_split(hotpot_train + nq_train, "train")
    valid_rows = tag_split(hotpot_valid + nq_valid, "valid")
    assert_disjoint(
        ("train", [str(row["id"]) for row in train_rows]),
        ("valid", [str(row["id"]) for row in valid_rows]),
        ("eval", locked),
    )

    corpus = merge_passages(read_jsonl(corpus_path), hotpot_passages, nq_passages)
    n_leaky = count_leaky_anchors(corpus)
    if n_leaky:
        raise SystemExit(f"Refusing to write {n_leaky} answer-anchor passages.")

    train_errors = ranking_data_errors(cfg, train_rows, corpus, split="train")
    valid_errors = ranking_data_errors(cfg, valid_rows, corpus, split="valid")
    if train_errors or valid_errors:
        raise SystemExit(
            "\n".join(train_errors + valid_errors)
            + "\nDo not run prepare_data.py --hf to fix this. It rewrites eval_slice.jsonl."
        )

    # Re-read eval ids immediately before writing, and do not open eval for write.
    assert_ids_unchanged(
        [str(row.get("id")) for row in read_jsonl(eval_path)],
        locked,
        eval_path.name,
    )
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    write_jsonl(corpus_path, corpus)
    meta = {
        "train_file": str(data_cfg["train_file"]),
        "valid_file": str(valid_rel),
        "eval_file_unchanged": str(data_cfg["eval_file"]),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "train_by_dataset": counts_by_dataset(train_rows),
        "valid_by_dataset": counts_by_dataset(valid_rows),
        "n_corpus": len(corpus),
        "single_hop": nq_stats,
        "note": "Validation is the next HF train-split prefix. Eval ids were checked, not rewritten.",
    }
    meta_path = train_path.parent / "train_valid_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {train_path} n={len(train_rows)} {counts_by_dataset(train_rows)}")
    print(f"Wrote {valid_path} n={len(valid_rows)} {counts_by_dataset(valid_rows)}")
    print(f"Updated {corpus_path} n={len(corpus)} (eval file not rewritten)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
