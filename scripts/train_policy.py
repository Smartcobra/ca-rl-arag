#!/usr/bin/env python3
"""REINFORCE trainer for the tiny learned policy.

Train split only: reads ``train_slice.jsonl`` (100 examples). Never opens
``eval_slice.jsonl``. The 300-example ranking file is a later, separate eval.

Each epoch logs sampled ``mean_reward`` and a deterministic (argmax)
``eval_reward`` on the same train examples. Frozen ``run_pilot`` scoring
also uses argmax; without this pass the sample-vs-greedy gap is invisible.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agentic_rag import AgenticRAG
from src.config import load_config, resolve_path
from src.data.loaders import stratified_limit
from src.data.preflight import assert_ranking_data
from src.generation import build_generator
from src.gpu import cleanup_gpu_resources, log_gpu_memory
from src.metrics import counts_by_dataset
from src.policies.learned import MAX_HIDDEN, LearnedPolicy, assert_train_only_path
from src.rag_env import ACTION_TO_IDX, AgenticRAGEnv
from src.retrieval import BM25Retriever
from src.utils import ensure_dir, read_jsonl, set_seed


def _learned_cfg(cfg: dict) -> dict:
    return dict((cfg.get("policy") or {}).get("learned") or {})


def _load_train_examples(cfg: dict, limit: int | None) -> tuple[list[dict], Path]:
    train_rel = cfg["data"]["train_file"]
    train_path = assert_train_only_path(resolve_path(cfg, train_rel))
    examples = read_jsonl(train_path)
    examples = stratified_limit(examples, limit, seed=int(cfg["experiment"]["seed"]))
    return examples, train_path


def _reinforce_loss(
    log_probs: list[torch.Tensor],
    entropies: list[torch.Tensor],
    advantage: float,
    entropy_coef: float,
) -> torch.Tensor:
    logp_sum = torch.stack(log_probs).sum()
    ent_sum = torch.stack(entropies).sum()
    return -(logp_sum * float(advantage)) - float(entropy_coef) * ent_sum


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _episode_action_stats(env: AgenticRAGEnv) -> dict[str, float]:
    state = env._state
    if state is None:
        return {"n_steps": 0.0, "n_retrieve": 0.0, "n_verify": 0.0}
    return {
        "n_steps": float(len(state.action_history)),
        "n_retrieve": float(state.counts.get("retrieve", 0)),
        "n_verify": float(state.counts.get("verify", 0)),
    }


def _rollout(
    env: AgenticRAGEnv,
    policy: LearnedPolicy,
    cfg: dict,
    example: dict,
    *,
    deterministic: bool,
) -> tuple[float, float, dict[str, float], list[torch.Tensor], list[torch.Tensor]]:
    """One episode. Training samples; greedy eval uses argmax (deterministic=True)."""
    obs, _info = env.reset(options={"example": example})
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    done = False
    last_info: dict = {}
    reward = 0.0
    while not done:
        obs_before = obs
        action_idx, logp, ent = policy.act(obs, cfg=cfg, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action_idx)
        executed_name = str(info.get("action") or "")
        executed_idx = ACTION_TO_IDX.get(executed_name, action_idx)
        if executed_idx != action_idx:
            logp, ent = policy.log_prob_action(obs_before, executed_idx, cfg=cfg)
        if not deterministic:
            log_probs.append(logp)
            entropies.append(ent)
        reward = float(r)
        last_info = info
        done = bool(term or trunc)
    ep = last_info.get("episode_result") or {}
    return reward, float(ep.get("em") or 0.0), _episode_action_stats(env), log_probs, entropies


def greedy_eval_epoch(
    env: AgenticRAGEnv,
    policy: LearnedPolicy,
    cfg: dict,
    examples: list[dict],
) -> dict[str, float]:
    """Argmax pass on the TRAIN examples. Never opens eval_slice.jsonl.

    Frozen run_pilot scoring also uses argmax. Logging this as eval_reward is
    how we see the sample-vs-greedy gap that previously erased tool use.
    """
    was_training = policy.mlp.training
    policy.mlp.eval()
    rewards: list[float] = []
    ems: list[float] = []
    steps: list[float] = []
    retrieves: list[float] = []
    verifies: list[float] = []
    try:
        with torch.no_grad():
            for ex in examples:
                reward, em, stats, _, _ = _rollout(env, policy, cfg, ex, deterministic=True)
                rewards.append(reward)
                ems.append(em)
                steps.append(stats["n_steps"])
                retrieves.append(stats["n_retrieve"])
                verifies.append(stats["n_verify"])
    finally:
        if was_training:
            policy.mlp.train()
    return {
        "eval_reward": _mean(rewards),
        "eval_em": _mean(ems),
        "eval_n_steps": _mean(steps),
        "eval_n_retrieve": _mean(retrieves),
        "eval_n_verify": _mean(verifies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny REINFORCE policy on the 100-example train split only.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reward-preset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional stratified cap on the TRAIN file only. Default: full train slice (100).",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip train-size + corpus-size preflight (synthetic / extractive debug only).",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--curve",
        default=None,
        help="Learning-curve JSON. Default: train_policy_curve.json for the default "
        "checkpoint, else train_policy_curve_<checkpoint-suffix>.json.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, reward_preset=args.reward_preset)
    learned = _learned_cfg(cfg)
    epochs = int(args.epochs if args.epochs is not None else learned.get("epochs", 40))
    lr = float(args.lr if args.lr is not None else learned.get("lr", 0.003))
    hidden = int(args.hidden if args.hidden is not None else learned.get("hidden", 16))
    entropy_coef = float(args.entropy_coef if args.entropy_coef is not None else learned.get("entropy_coef", 0.01))
    ckpt_rel = args.checkpoint or learned.get("checkpoint", "results/checkpoints/learned_policy.pt")
    if hidden > MAX_HIDDEN:
        raise SystemExit(f"--hidden {hidden} exceeds cap {MAX_HIDDEN} (100 train examples).")

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    examples, train_path = _load_train_examples(cfg, args.limit)
    corpus = read_jsonl(resolve_path(cfg, cfg["data"]["corpus_file"]))
    if not args.skip_data_check:
        # Preflight the on-disk train file (not a limited subset, not eval).
        full_train = read_jsonl(train_path)
        assert_ranking_data(cfg, full_train, corpus, split="train")
    if args.limit:
        n_file = counts_by_dataset(read_jsonl(train_path))
        n_run = counts_by_dataset(examples)
        print(f"Stratified --limit {args.limit} on TRAIN: {n_file} -> {n_run}")

    n_run = counts_by_dataset(examples)
    print(
        f"TRAIN ONLY path={train_path} n={len(examples)} by_dataset={n_run} "
        f"preset={cfg['reward_preset_name']} hidden={hidden} epochs={epochs} lr={lr}"
    )

    retriever = BM25Retriever(corpus)
    metrics_dir = ensure_dir(resolve_path(cfg, cfg["logging"]["metrics_dir"]))
    ckpt_path = resolve_path(cfg, ckpt_rel)
    best_path = ckpt_path.with_name(ckpt_path.stem + "_best" + ckpt_path.suffix)
    if args.curve:
        curve_path = resolve_path(cfg, args.curve)
    else:
        default_ckpt = Path(str(learned.get("checkpoint", "results/checkpoints/learned_policy.pt")))
        if Path(ckpt_rel).as_posix() == default_ckpt.as_posix() or ckpt_path.stem == "learned_policy":
            curve_path = metrics_dir / "train_policy_curve.json"
        else:
            suffix = ckpt_path.stem.removeprefix("learned_policy_").removeprefix("learned_policy")
            curve_path = metrics_dir / f"train_policy_curve_{suffix or ckpt_path.stem}.json"

    generator = None
    agent = None
    env = None
    curve: list[dict] = []
    baseline = 0.0
    best_eval_reward = float("-inf")

    log_gpu_memory("before model creation")
    generator = build_generator(cfg)
    log_gpu_memory("after model creation")

    try:
        agent = AgenticRAG(cfg, retriever, generator=generator)
        env = AgenticRAGEnv(cfg, retriever, examples, seed=seed, agent=agent)
        policy = LearnedPolicy(hidden=hidden, seed=seed, device="cpu")
        opt = torch.optim.Adam(policy.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            order = list(examples)
            random.Random(seed + epoch).shuffle(order)
            epoch_rewards: list[float] = []
            epoch_em: list[float] = []
            epoch_steps: list[float] = []
            epoch_retrieve: list[float] = []
            epoch_verify: list[float] = []

            for ex in order:
                reward, em, stats, log_probs, entropies = _rollout(
                    env, policy, cfg, ex, deterministic=False
                )
                epoch_rewards.append(reward)
                epoch_em.append(em)
                epoch_steps.append(stats["n_steps"])
                epoch_retrieve.append(stats["n_retrieve"])
                epoch_verify.append(stats["n_verify"])

                if log_probs:
                    advantage = reward - baseline
                    loss = _reinforce_loss(log_probs, entropies, advantage, entropy_coef)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                baseline = 0.95 * baseline + 0.05 * reward

            greedy = greedy_eval_epoch(env, policy, cfg, examples)
            row = {
                "epoch": epoch,
                "n": len(order),
                "mean_reward": _mean(epoch_rewards),
                "mean_em": _mean(epoch_em),
                "mean_n_steps": _mean(epoch_steps),
                "mean_n_retrieve": _mean(epoch_retrieve),
                "mean_n_verify": _mean(epoch_verify),
                "eval_reward": greedy["eval_reward"],
                "eval_em": greedy["eval_em"],
                "eval_n_steps": greedy["eval_n_steps"],
                "eval_n_retrieve": greedy["eval_n_retrieve"],
                "eval_n_verify": greedy["eval_n_verify"],
                "eval_split": "train_greedy",
                "split": "train",
                "hidden": hidden,
                "lr": lr,
                "reward_preset": cfg["reward_preset_name"],
            }
            curve.append(row)
            curve_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")
            print(
                f"epoch {epoch}/{epochs}  n={row['n']}  "
                f"mean_reward={row['mean_reward']:.4f}  eval_reward={row['eval_reward']:.4f}  "
                f"mean_em={row['mean_em']:.3f}  eval_em={row['eval_em']:.3f}  "
                f"mean_steps={row['mean_n_steps']:.2f}  eval_steps={row['eval_n_steps']:.2f}  "
                f"mean_retrieve={row['mean_n_retrieve']:.2f}  eval_retrieve={row['eval_n_retrieve']:.2f}  "
                f"mean_verify={row['mean_n_verify']:.2f}  eval_verify={row['eval_n_verify']:.2f}"
            )
            extra = {
                "epoch": epoch,
                "mean_reward": row["mean_reward"],
                "eval_reward": row["eval_reward"],
            }
            policy.save(ckpt_path, extra=extra)
            if row["eval_reward"] > best_eval_reward:
                best_eval_reward = row["eval_reward"]
                policy.save(best_path, extra=extra)

        print(f"Wrote {curve_path}")
        print(f"Wrote {ckpt_path}")
        if best_path.exists():
            print(f"Wrote {best_path} (best eval_reward={best_eval_reward:.4f})")
    finally:
        log_gpu_memory("before cleanup")
        cleanup_gpu_resources(env, agent, generator)
        env = agent = generator = None
        log_gpu_memory("after cleanup")


if __name__ == "__main__":
    main()
