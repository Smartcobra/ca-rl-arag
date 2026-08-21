#!/usr/bin/env python3
"""Sweep reward-weight ablation presets on a fixed eval slice."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agentic_rag import AgenticRAG
from src.config import load_config, resolve_path
from src.data.loaders import stratified_limit
from src.evaluate import evaluate_agent, save_metrics
from src.generation import build_generator
from src.gpu import cleanup_gpu_resources, log_gpu_memory
from src.metrics import compact_ablation_row, counts_by_dataset, format_eval_summary, ordered_dataset_names
from src.policies import get_policy
from src.retrieval import BM25Retriever
from src.utils import ensure_dir, read_jsonl, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Stratified cap from the eval file (default 100 = 50/50 on a 150+150 slice). "
        "Pass 0 to use the full eval. Not the ranking table; that is the full 300-example pilot.",
    )
    parser.add_argument(
        "--presets",
        default="default,correctness_only,correctness_grounding,correctness_faithfulness_cost,lambda_zero,high_cost_pressure",
    )
    parser.add_argument("--policy", default="rule_based")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    set_seed(int(base_cfg["experiment"]["seed"]))
    corpus = read_jsonl(resolve_path(base_cfg, base_cfg["data"]["corpus_file"]))
    examples = read_jsonl(resolve_path(base_cfg, base_cfg["data"]["eval_file"]))
    n_file = counts_by_dataset(examples)
    limit = None if args.limit == 0 else args.limit
    examples = stratified_limit(examples, limit, seed=int(base_cfg["experiment"]["seed"]))
    n_run = counts_by_dataset(examples)
    print(f"Ablation slice: n={len(examples)} by_dataset={n_run} (eval file was {n_file}; not a ranking table)")
    retriever = BM25Retriever(corpus)
    policy_fn, policy_name = get_policy(args.policy)

    metrics_dir = ensure_dir(resolve_path(base_cfg, base_cfg["logging"]["metrics_dir"]))
    table = {}
    generator = None
    agent = None

    log_gpu_memory("before model creation")
    generator = build_generator(base_cfg)
    log_gpu_memory("after model creation")

    try:
        for preset in [p.strip() for p in args.presets.split(",") if p.strip()]:
            log_gpu_memory(f"before {preset}")
            cfg = load_config(args.config, reward_preset=preset)
            agent = AgenticRAG(cfg, retriever, generator=generator)
            out = evaluate_agent(agent, examples, policy_fn, policy_name)
            save_metrics(metrics_dir / f"ablation_{policy_name}_{preset}.json", out["summary"], {"preset": preset})
            table[preset] = compact_ablation_row(out["summary"])
            print(format_eval_summary(preset, out["summary"]))
            del out
            # Agent does not own the shared generator; drop the wrapper between presets.
            agent.close()
            agent = None
            log_gpu_memory(f"after {preset}")

        out_path = metrics_dir / "reward_ablation_table.json"
        out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        by_ds_table: dict[str, dict] = {}
        for preset, row in table.items():
            for ds in ordered_dataset_names(row.get("by_dataset") or {}):
                by_ds_table.setdefault(ds, {})[preset] = row["by_dataset"][ds]
        by_ds_path = metrics_dir / "reward_ablation_by_dataset.json"
        by_ds_path.write_text(json.dumps(by_ds_table, indent=2), encoding="utf-8")
        print(f"Wrote {by_ds_path}")
    finally:
        log_gpu_memory("before cleanup")
        cleanup_gpu_resources(agent, generator)
        agent = generator = None
        log_gpu_memory("after cleanup")


if __name__ == "__main__":
    main()
