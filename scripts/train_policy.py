#!/usr/bin/env python3
"""REINFORCE trainer for the tiny learned policy.

Reads the train file in the config. Never opens ``eval_slice.jsonl``.
The 300-example ranking file is a later, separate eval.

Advantage is the sampled trajectory reward minus the naive reward on that
same question (cached per reward preset). A group of K samples plus this
anchor is Milestone 4. This trainer uses one sample.

When ``data.valid_file`` exists, ``_best.pt`` is the epoch with the highest
greedy reward on that validation slice. Otherwise greedy stays on the train
file (smoke runs and the older 100-example config).
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
from src.data.id_lock import assert_disjoint, load_locked_eval_ids
from src.data.loaders import stratified_limit
from src.data.preflight import assert_ranking_data
from src.generation import build_generator
from src.gpu import cleanup_gpu_resources, log_gpu_memory
from src.metrics import counts_by_dataset
from src.policies import naive_stop_policy
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


def advantage_vs_naive(sampled_reward: float, naive_reward: float) -> float:
    """One sampled trajectory minus the naive reward on the same question.

    K sampled trajectories plus this anchor is Milestone 4 (GRPO). K=1 here.
    """
    return float(sampled_reward) - float(naive_reward)


def _naive_reward(env: AgenticRAGEnv, cfg: dict, example: dict) -> float:
    """Deterministic retrieve-then-stop reward. Cached because λ changes the scalar."""
    _obs, info = env.reset(options={"example": example})
    state = env._state
    done = False
    reward = 0.0
    while not done and state is not None:
        structured = info.get("structured_obs") or state.observation(env._tracker, cfg)
        action_name = naive_stop_policy(structured, state)
        _obs, reward, term, trunc, info = env.step(ACTION_TO_IDX[action_name])
        state = env._state
        done = bool(term or trunc)
    return float(reward)


def _naive_cache_path(metrics_dir: Path, preset: str) -> Path:
    return Path(metrics_dir) / f"naive_reward_cache_{preset}.json"


def ensure_naive_reward_cache(
    env: AgenticRAGEnv,
    cfg: dict,
    examples: list[dict],
    path: Path,
) -> dict[str, float]:
    preset = str(cfg["reward_preset_name"])
    stored: dict[str, float] = {}
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob, dict) and blob.get("reward_preset") == preset:
            stored = {str(k): float(v) for k, v in (blob.get("rewards") or {}).items()}
    missing = [ex for ex in examples if str(ex.get("id")) not in stored]
    if missing:
        print(f"Caching naive reward for {len(missing)} questions (preset={preset})")
    for i, ex in enumerate(missing, start=1):
        stored[str(ex.get("id"))] = _naive_reward(env, cfg, ex)
        if i % 25 == 0 or i == len(missing):
            print(f"  naive cache {i}/{len(missing)}")
            _write_naive_cache(path, preset, stored)
    if missing:
        _write_naive_cache(path, preset, stored)
    return stored


def _write_naive_cache(path: Path, preset: str, rewards: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"reward_preset": preset, "n": len(rewards), "rewards": rewards}, indent=2),
        encoding="utf-8",
    )


def _load_valid_examples(cfg: dict) -> tuple[list[dict], Path] | None:
    rel = (cfg.get("data") or {}).get("valid_file")
    if not rel:
        return None
    path = assert_train_only_path(resolve_path(cfg, rel))
    if not path.exists():
        return None
    return read_jsonl(path), path


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


def _trainer_state_path(ckpt_path: Path) -> Path:
    return Path(ckpt_path).with_name(Path(ckpt_path).stem + "_trainer" + Path(ckpt_path).suffix)


def _load_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"curve file must be a JSON list: {path}")
    return data


def _last_finished_epoch(curve: list[dict]) -> int:
    epochs = [int(row["epoch"]) for row in curve if row.get("epoch") is not None]
    return max(epochs) if epochs else 0


def save_trainer_state(
    path: Path,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    baseline: float,
    best_eval_reward: float,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "optimizer": optimizer.state_dict(),
            "baseline": float(baseline),
            "best_eval_reward": float(best_eval_reward),
        },
        path,
    )
    return path


def load_trainer_state(path: Path) -> dict:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    for key in ("epoch", "optimizer", "baseline", "best_eval_reward"):
        if key not in payload:
            raise ValueError(f"trainer state {path} missing {key}")
    return payload


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load last checkpoint, trainer state (epoch / optimizer / baseline / "
        "best_eval_reward), and the existing curve, then continue from the next epoch.",
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
        raise SystemExit(f"--hidden {hidden} exceeds cap {MAX_HIDDEN}.")

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
    valid_pair = _load_valid_examples(cfg)
    if valid_pair is None and (cfg.get("data") or {}).get("valid_file") and not args.skip_data_check:
        raise SystemExit(
            f"Missing validation file {cfg['data']['valid_file']}. "
            "Run: python scripts/build_train_valid.py --config configs/frontier_v3.yaml"
        )
    if valid_pair is None:
        select_examples = examples
        eval_split = "train_greedy"
        valid_path = None
    else:
        select_examples, valid_path = valid_pair
        train_ids = [str(ex.get("id")) for ex in read_jsonl(train_path)]
        valid_ids = [str(ex.get("id")) for ex in select_examples]
        locked_ids = load_locked_eval_ids()
        assert_disjoint(("train", train_ids), ("valid", valid_ids), ("eval", locked_ids))
        if not args.skip_data_check:
            assert_ranking_data(cfg, select_examples, corpus, split="valid")
        eval_split = "valid_greedy"
    print(
        f"TRAIN ONLY path={train_path} n={len(examples)} by_dataset={n_run} "
        f"preset={cfg['reward_preset_name']} hidden={hidden} epochs={epochs} lr={lr}"
    )
    if valid_path is not None:
        print(f"SELECT on validation path={valid_path} n={len(select_examples)} split={eval_split}")
    else:
        print(f"SELECT on train greedy n={len(select_examples)} (no valid_file)")
    print("advantage = sampled reward - naive reward on the same question")

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
    start_epoch = 1
    trainer_path = _trainer_state_path(ckpt_path)

    log_gpu_memory("before model creation")
    generator = build_generator(cfg)
    log_gpu_memory("after model creation")

    try:
        agent = AgenticRAG(cfg, retriever, generator=generator)
        env = AgenticRAGEnv(cfg, retriever, examples, seed=seed, agent=agent)
        naive_cache = ensure_naive_reward_cache(
            env,
            cfg,
            examples,
            _naive_cache_path(metrics_dir, str(cfg["reward_preset_name"])),
        )
        policy = LearnedPolicy(hidden=hidden, seed=seed, device="cpu")
        opt = torch.optim.Adam(policy.parameters(), lr=lr)

        if args.resume:
            if not ckpt_path.exists():
                raise SystemExit(f"--resume needs checkpoint {ckpt_path}")
            if not trainer_path.exists():
                raise SystemExit(
                    f"--resume needs trainer state {trainer_path} "
                    "(epoch, optimizer, baseline, best_eval_reward)"
                )
            if not curve_path.exists():
                raise SystemExit(f"--resume needs existing curve {curve_path}")
            loaded = LearnedPolicy.load(ckpt_path, device="cpu")
            if loaded.hidden != hidden:
                raise SystemExit(
                    f"--resume hidden mismatch: checkpoint={loaded.hidden} requested={hidden}"
                )
            policy = loaded
            opt = torch.optim.Adam(policy.parameters(), lr=lr)
            state = load_trainer_state(trainer_path)
            opt.load_state_dict(state["optimizer"])
            baseline = float(state["baseline"])
            best_eval_reward = float(state["best_eval_reward"])
            curve = _load_curve(curve_path)
            finished = int(state["epoch"])
            curve_finished = _last_finished_epoch(curve)
            if curve_finished != finished:
                print(
                    f"RESUME warning: trainer epoch={finished} curve last={curve_finished}. "
                    "Continuing from trainer epoch + 1."
                )
            start_epoch = finished + 1
            print(
                f"RESUME finished_epoch={finished} next={start_epoch}/{epochs} "
                f"baseline={baseline:.4f} best_eval_reward={best_eval_reward:.4f} "
                f"curve_rows={len(curve)}"
            )
            if start_epoch > epochs:
                print(f"Already finished {finished} of {epochs} epochs. Nothing to do.")
                return

        for epoch in range(start_epoch, epochs + 1):
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
                    ex_id = str(ex.get("id"))
                    if ex_id not in naive_cache:
                        naive_cache[ex_id] = _naive_reward(env, cfg, ex)
                    advantage = advantage_vs_naive(reward, naive_cache[ex_id])
                    loss = _reinforce_loss(log_probs, entropies, advantage, entropy_coef)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                # EMA baseline stays in the trainer file so --resume keeps its schema.
                # The advantage above does not use it.

            greedy = greedy_eval_epoch(env, policy, cfg, select_examples)
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
                "eval_split": eval_split,
                "advantage": "sampled_minus_naive",
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
                "baseline": baseline,
                "best_eval_reward": best_eval_reward,
            }
            if row["eval_reward"] > best_eval_reward:
                best_eval_reward = row["eval_reward"]
                extra["best_eval_reward"] = best_eval_reward
                extra["selection_split"] = eval_split
                policy.save(best_path, extra=extra)
            policy.save(ckpt_path, extra=extra)
            save_trainer_state(
                trainer_path,
                epoch=epoch,
                optimizer=opt,
                baseline=baseline,
                best_eval_reward=best_eval_reward,
            )

        print(f"Wrote {curve_path}")
        print(f"Wrote {ckpt_path}")
        print(f"Wrote {trainer_path}")
        if best_path.exists():
            print(f"Wrote {best_path} (best eval_reward={best_eval_reward:.4f})")
    finally:
        log_gpu_memory("before cleanup")
        cleanup_gpu_resources(env, agent, generator)
        env = agent = generator = None
        log_gpu_memory("after cleanup")


if __name__ == "__main__":
    main()
