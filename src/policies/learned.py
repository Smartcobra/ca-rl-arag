"""Tiny MLP action head for Milestone-3 REINFORCE (train-only).

Input is the 15-d vector from ``AgenticRAGEnv._vector_obs``. Output is 5
logits in ``ACTIONS`` order. Hidden width is capped: the train slice is
too small for a wide net.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..agentic_rag import ACTIONS, AgentState
from ..rag_env import IDX_TO_ACTION, OBS_DIM, vectorize_structured_obs

N_ACTIONS = len(ACTIONS)
MAX_HIDDEN = 32
# 15→16→5 = 341. 15→32→5 = 677. Tests refuse anything larger.
MAX_PARAM_COUNT = 700

# Vector layout must match AgenticRAGEnv._vector_obs. Append-only:
# [mean_score, n_evidence/10, remaining_steps/max, remaining_usd/max_usd,
#  verify_support, verify_contra, retrieve/3, rewrite/2, rerank/2, verify/2,
#  top1_bm25, top1_top2_gap, verify_one_hot × 3]
_IDX_N_EVIDENCE = 1
_IDX_REMAINING_STEPS = 2
_IDX_RETRIEVE = 6
_IDX_REWRITE = 7
_IDX_RERANK = 8
_IDX_VERIFY = 9


def assert_train_only_path(path: str | Path) -> Path:
    """Refuse any path whose filename looks like the locked eval split."""
    p = Path(path)
    if "eval" in p.name.lower():
        raise ValueError(
            f"Refusing to load eval data for training: {p}. "
            "Use train_slice.jsonl or valid_slice.jsonl. Never eval_slice.jsonl."
        )
    return p


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def legal_mask(obs_vec: np.ndarray, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Boolean mask over ACTIONS. Cuts illegal / wasted tools on a tiny dataset."""
    cfg = cfg or {}
    agent = cfg.get("agent") or {}
    max_steps = float(agent.get("max_steps", 8))
    max_retrieve = float(agent.get("max_retrieve", 3))
    max_rewrite = float(agent.get("max_rewrite", 2))
    max_rerank = float(agent.get("max_rerank", 2))
    max_verify = float(agent.get("max_verify", 2))

    vec = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
    if vec.size != OBS_DIM:
        raise ValueError(f"expected obs dim {OBS_DIM}, got {vec.size}")

    n_ev = float(vec[_IDX_N_EVIDENCE]) * 10.0
    remaining_steps = float(vec[_IDX_REMAINING_STEPS]) * max_steps
    n_retrieve = float(vec[_IDX_RETRIEVE]) * 3.0
    n_rewrite = float(vec[_IDX_REWRITE]) * 2.0
    n_rerank = float(vec[_IDX_RERANK]) * 2.0
    n_verify = float(vec[_IDX_VERIFY]) * 2.0

    allow = np.ones(N_ACTIONS, dtype=bool)
    if remaining_steps <= 1.0 + 1e-5:
        allow[:] = False
        allow[ACTIONS.index("stop")] = True
        return allow

    if n_ev < 0.5:
        allow[ACTIONS.index("rerank")] = False
        allow[ACTIONS.index("verify")] = False
        allow[ACTIONS.index("stop")] = False
        if n_retrieve >= max_retrieve - 1e-5:
            allow[ACTIONS.index("retrieve")] = False
        if n_rewrite >= max_rewrite - 1e-5:
            allow[ACTIONS.index("rewrite")] = False
        if not allow.any():
            allow[ACTIONS.index("stop")] = True
        return allow

    if n_retrieve >= max_retrieve - 1e-5:
        allow[ACTIONS.index("retrieve")] = False
    if n_rewrite >= max_rewrite - 1e-5:
        allow[ACTIONS.index("rewrite")] = False
    if n_rerank >= max_rerank - 1e-5:
        allow[ACTIONS.index("rerank")] = False
    if n_verify >= max_verify - 1e-5:
        allow[ACTIONS.index("verify")] = False
    if not allow.any():
        allow[ACTIONS.index("stop")] = True
    return allow


class PolicyMLP(nn.Module):
    """15 → hidden → 5. Default hidden=16 is 341 parameters."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        hidden = int(hidden)
        if hidden < 1 or hidden > MAX_HIDDEN:
            raise ValueError(f"hidden must be in 1..{MAX_HIDDEN} (got {hidden}); the train slice is not much data")
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
        if parameter_count(self) > MAX_PARAM_COUNT:
            raise ValueError(f"policy has {parameter_count(self)} params; cap is {MAX_PARAM_COUNT}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class LearnedPolicy:
    """CPU Categorical policy over the env action ids."""

    def __init__(self, hidden: int = 16, seed: int = 42, device: str = "cpu"):
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.device = torch.device(device)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        self.mlp = PolicyMLP(self.hidden).to(self.device)
        self.mlp.train()

    def parameters(self):
        return self.mlp.parameters()

    def _dist(self, obs_np: np.ndarray, cfg: dict[str, Any] | None) -> Categorical:
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32).reshape(1, OBS_DIM), device=self.device)
        logits = self.mlp(obs)
        if cfg is not None:
            mask = torch.as_tensor(legal_mask(obs_np, cfg), device=self.device)
            logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        return Categorical(logits=logits)

    def act(
        self,
        obs_np: np.ndarray,
        cfg: dict[str, Any] | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs_np, cfg)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).squeeze()
        entropy = dist.entropy().squeeze()
        return int(action.item()), log_prob, entropy

    def log_prob_action(
        self,
        obs_np: np.ndarray,
        action_idx: int,
        cfg: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs_np, cfg)
        action = torch.as_tensor([int(action_idx)], device=self.device)
        return dist.log_prob(action).squeeze(), dist.entropy().squeeze()

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.mlp.state_dict(),
            "hidden": self.hidden,
            "seed": self.seed,
            "obs_dim": OBS_DIM,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "LearnedPolicy":
        path = Path(path)
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=device)
        saved_dim = payload.get("obs_dim")
        if saved_dim is not None and int(saved_dim) != OBS_DIM:
            raise ValueError(
                f"{path} was trained with obs_dim={saved_dim}; this code expects {OBS_DIM}. "
                "v2 checkpoints stay on disk. Train a v3 checkpoint."
            )
        pol = cls(hidden=int(payload["hidden"]), seed=int(payload.get("seed") or 0), device=device)
        try:
            pol.mlp.load_state_dict(payload["state_dict"])
        except RuntimeError as exc:
            raise ValueError(
                f"{path} does not match OBS_DIM={OBS_DIM}. "
                "A 10-d v2 checkpoint cannot load into the 15-d head."
            ) from exc
        return pol

    def as_callable(self, cfg: dict[str, Any], *, deterministic: bool = True):
        """``(structured_obs, state) -> action_name`` for ``evaluate_agent`` / ``run_pilot``."""
        self.mlp.eval()

        def policy_fn(obs: dict[str, Any], state: AgentState | None = None) -> str:
            del state
            vec = vectorize_structured_obs(obs, cfg)
            with torch.no_grad():
                action_idx, _, _ = self.act(vec, cfg=cfg, deterministic=deterministic)
            return IDX_TO_ACTION[action_idx]

        return policy_fn


def learned_checkpoint_path(cfg: dict[str, Any], checkpoint: str | Path | None = None) -> Path:
    from ..config import resolve_path

    learned = (cfg.get("policy") or {}).get("learned") or {}
    rel = checkpoint if checkpoint is not None else learned.get("checkpoint", "results/checkpoints/learned_policy.pt")
    if cfg.get("root"):
        return resolve_path(cfg, rel)
    return Path(rel)


def load_learned_policy_fn(
    cfg: dict[str, Any] | None,
    checkpoint: str | Path | None = None,
    *,
    deterministic: bool = True,
):
    if cfg is None:
        raise ValueError("learned policy requires cfg to resolve the checkpoint")
    path = learned_checkpoint_path(cfg, checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"learned checkpoint missing: {path}. Train first: python scripts/train_policy.py"
        )
    return LearnedPolicy.load(path).as_callable(cfg, deterministic=deterministic)
