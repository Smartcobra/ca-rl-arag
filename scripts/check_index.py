#!/usr/bin/env python3
"""Check which BM25 index a trajectory file was scored against. No GPU.

Merging new gold passages into `corpus.jsonl` changes BM25 for every question,
so rows from two different corpora are not comparable. This re-runs the first
retrieve for the locked 300 and compares the top ids with what a trajectory
file logged at step 0.

Usage:
  python scripts/check_index.py --config configs/frontier_v3.yaml \
      --against results/trajectories/learned_frontier_act02_v3.jsonl \
      --against results/trajectories/max_tools_default.jsonl
  # add --require-match to exit non-zero unless every file matches 300/300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.data.preflight import corpus_fingerprint
from src.retrieval import BM25Retriever
from src.utils import read_jsonl


def first_retrieve_top_ids(row: dict) -> list[str] | None:
    """Step-0 top ids are the raw BM25 hit, before any rerank rewrites scores."""
    for step in row.get("trajectory") or []:
        if step.get("action") == "retrieve":
            ids = step.get("top_ids")
            return [str(i) for i in ids] if ids else None
        # A non-retrieve first action means the file cannot pin the index.
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/frontier_v3.yaml")
    parser.add_argument(
        "--against",
        action="append",
        default=[],
        help="Trajectory JSONL to compare. Repeatable.",
    )
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="Exit non-zero unless every compared file matches on every question.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus = read_jsonl(resolve_path(cfg, cfg["data"]["corpus_file"]))
    examples = read_jsonl(resolve_path(cfg, cfg["data"]["eval_file"]))
    fingerprint = corpus_fingerprint(corpus)
    print(f"corpus: {fingerprint['n_passages']} passages, id sha1 {fingerprint['sha1']}")
    print(f"eval:   {len(examples)} questions")

    top_k = int(cfg.get("retrieval", {}).get("top_k", 5))
    retriever = BM25Retriever(corpus)
    live = {
        str(row["id"]): [d["passage_id"] for d in retriever.search(row["question"], top_k=top_k)]
        for row in examples
    }
    print(f"re-ran the first retrieve for {len(live)} questions\n")

    failures = []
    for rel in args.against:
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT / rel
        if not path.exists():
            print(f"{path.name}: MISSING")
            failures.append(path.name)
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        compared = top1 = top5 = 0
        skipped = 0
        for row in rows:
            logged = first_retrieve_top_ids(row)
            got = live.get(str(row.get("id")))
            if logged is None or got is None:
                skipped += 1
                continue
            compared += 1
            top1 += int(logged[:1] == got[:1])
            top5 += int(logged == got)
        verdict = "same index" if compared and top5 == compared else "DIFFERENT INDEX"
        print(
            f"{path.name}: {verdict} — top-1 {top1}/{compared}, top-{top_k} {top5}/{compared}"
            + (f", skipped {skipped}" if skipped else "")
        )
        if not compared or top5 != compared:
            failures.append(path.name)

    if args.require_match and failures:
        raise SystemExit(f"\nIndex mismatch: {', '.join(failures)}")


if __name__ == "__main__":
    main()
