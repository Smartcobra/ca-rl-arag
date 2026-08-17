#!/usr/bin/env python3
"""Run Milestone-2 pilot: baseline RAG + agentic policies + optional env rollouts."""

from __future__ import annotations

# Secondary CUDA allocator hint. Must be set before torch is imported.
# This does not free referenced tensors; lifecycle cleanup below is the primary fix.
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
from src.evaluate import evaluate_agent, evaluate_baseline, save_metrics
from src.generation import build_generator
from src.gpu import cleanup_gpu_resources, log_gpu_memory
from src.policies import get_policy
from src.rag_baseline import RAGBaseline
from src.rag_env import ACTION_TO_IDX, AgenticRAGEnv
from src.retrieval import BM25Retriever
from src.utils import ensure_dir, read_jsonl, set_seed, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reward-preset", default=None)
    parser.add_argument("--split", choices=["eval", "train"], default="eval")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--policies", default="naive_rag,rule_based,max_tools")
    parser.add_argument("--run-env-check", action="store_true", help="Roll a few Gymnasium episodes")
    args = parser.parse_args()

    cfg = load_config(args.config, reward_preset=args.reward_preset)
    set_seed(int(cfg["experiment"]["seed"]))

    corpus = read_jsonl(resolve_path(cfg, cfg["data"]["corpus_file"]))
    examples = read_jsonl(resolve_path(cfg, cfg["data"][f"{args.split}_file"]))
    if args.limit:
        examples = examples[: args.limit]

    retriever = BM25Retriever(corpus)
    print(f"Corpus={len(retriever)} examples={len(examples)} preset={cfg['reward_preset_name']}")

    traj_dir = ensure_dir(resolve_path(cfg, cfg["logging"]["trajectory_dir"]))
    metrics_dir = ensure_dir(resolve_path(cfg, cfg["logging"]["metrics_dir"]))

    results = {}
    generator = None
    baseline = None
    agent = None
    env = None

    # Strategy B: one immutable inference model shared across policies.
    # Isolation is at the policy/reward layer, not by loading a new Qwen copy.
    log_gpu_memory("before model creation")
    generator = build_generator(cfg)
    log_gpu_memory("after model creation")

    try:
        # 1) Standard RAG baseline
        log_gpu_memory("before naive_rag")
        baseline = RAGBaseline(cfg, retriever, generator=generator)
        base_path = traj_dir / f"baseline_{cfg['reward_preset_name']}.jsonl"
        if base_path.exists():
            base_path.unlink()
        base_out = evaluate_baseline(baseline, examples, out_path=base_path)
        save_metrics(metrics_dir / f"baseline_{cfg['reward_preset_name']}.json", base_out["summary"], {"policy": "naive_rag"})
        results["naive_rag"] = base_out["summary"]
        print("naive_rag:", json.dumps(base_out["summary"], indent=2))
        del base_out
        log_gpu_memory("after naive_rag")

        # 2) Agentic policies — same generator, different action policies
        agent = AgenticRAG(cfg, retriever, generator=generator)
        for name in [p.strip() for p in args.policies.split(",") if p.strip()]:
            if name == "naive_rag":
                continue
            policy_fn, policy_name = get_policy(name)
            log_gpu_memory(f"before {policy_name}")
            out_path = traj_dir / f"{policy_name}_{cfg['reward_preset_name']}.jsonl"
            if out_path.exists():
                out_path.unlink()
            out = evaluate_agent(agent, examples, policy_fn, policy_name, out_path=out_path)
            save_metrics(metrics_dir / f"{policy_name}_{cfg['reward_preset_name']}.json", out["summary"], {"policy": policy_name})
            results[policy_name] = out["summary"]
            print(f"{policy_name}:", json.dumps(out["summary"], indent=2))
            del out
            log_gpu_memory(f"after {policy_name}")

        # 3) Optional Gymnasium env sanity — reuse the agent (do not load a third model)
        if args.run_env_check and examples:
            log_gpu_memory("before env_check")
            env = AgenticRAGEnv(cfg, retriever, examples, seed=cfg["experiment"]["seed"], agent=agent)
            env_rows = []
            for i, ex in enumerate(examples[: min(5, len(examples))]):
                # rule-like action sequence via env API
                obs, info = env.reset(options={"example": ex})
                done = False
                total_r = 0.0
                steps = 0
                policy_fn, _ = get_policy("rule_based")
                # Use structured obs from info for policy; env also exposes vector obs
                state = env._state
                while not done and steps < int(cfg["agent"]["max_steps"]):
                    structured = info.get("structured_obs") or state.observation(env._tracker, cfg)
                    action_name = policy_fn(structured, state)
                    obs, r, term, trunc, info = env.step(ACTION_TO_IDX[action_name])
                    total_r += float(r)
                    done = term or trunc
                    steps += 1
                    state = env._state
                env_rows.append(
                    {
                        "id": ex.get("id"),
                        "reward": total_r,
                        "em": (info.get("episode_result") or {}).get("em"),
                        "f1": (info.get("episode_result") or {}).get("f1"),
                    }
                )
            write_jsonl(traj_dir / "env_rollouts.jsonl", env_rows)
            print("env_rollouts:", json.dumps(env_rows, indent=2))
            log_gpu_memory("after env_check")

        summary_path = metrics_dir / f"pilot_summary_{cfg['reward_preset_name']}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "reward_preset": cfg["reward_preset_name"],
                    "n_examples": len(examples),
                    "split": args.split,
                    "results": results,
                    "ablation_presets_available": cfg["reward_ablation_presets"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {summary_path}")
    finally:
        log_gpu_memory("before cleanup")
        cleanup_gpu_resources(env, agent, baseline, generator)
        env = agent = baseline = generator = None
        log_gpu_memory("after cleanup")


if __name__ == "__main__":
    main()
