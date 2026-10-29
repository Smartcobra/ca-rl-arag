"""Gymnasium RL environment wrapping Agentic RAG.

Episode = one QA example from initial query → stop (or budget/step exhaustion).
Action space (V1 frozen): retrieve | rewrite | rerank | verify | stop
Observation: compact state features (Scope Memo V2 §5).
Reward: trajectory reward assigned at episode end (sparse); optional step shaping=0 for V1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agentic_rag import ACTIONS, AgenticRAG, AgentState
from .cost import CostTracker
from .rewards import RewardComputer


ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

# Layout is append-only. legal_mask reads indices 1, 2, 6–9. New features go at the end.
# No dataset id: that would let the policy copy a Hotpot/NQ switch without reading the situation.
# 16 = the old 15, plus the minimum of the top-5 scores.
OBS_DIM = 16

# z-scored BM25 channels. The variance test checks every one of these.
SCORE_FEATURE_INDEX = {
    "mean_score": 0,
    "top1_bm25": 10,
    "top1_top2_gap": 11,
    "top5_min": 15,
}
SCORE_CLIP = 3.0
SCORE_NORM_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "bm25_score_norm.json"
_SCORE_NORM_KEYS = ("mean", "top1", "gap", "top5_min")


def _load_score_norm(path: Path = SCORE_NORM_PATH) -> dict[str, dict[str, float]]:
    """Mean and population std of the first BM25 top-5 on train_slice.jsonl."""
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. The observation standardizes BM25 with the training-slice "
            "mean and std; those constants live in that file. Frozen policies "
            "(naive_rag / rule_based / max_tools) do not need it."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload["features"]
    out: dict[str, dict[str, float]] = {}
    for name in _SCORE_NORM_KEYS:
        mean = float(features[name]["mean"])
        std = float(features[name]["std"])
        if std <= 0.0:
            raise ValueError(f"{path} has non-positive std for {name}")
        out[name] = {"mean": mean, "std": std}
    return out


# Loaded on first use, not at import: a frozen-baseline pilot never builds the
# vector, so a missing constants file must not stop it.
_SCORE_NORM: dict[str, dict[str, float]] | None = None


def score_norm() -> dict[str, dict[str, float]]:
    global _SCORE_NORM
    if _SCORE_NORM is None:
        _SCORE_NORM = _load_score_norm()
    return _SCORE_NORM


def standardize_score(value: float, feature: str) -> float:
    """(x - train_mean) / train_std, clipped to [-3, 3].

    ``feature`` is one of mean, top1, gap, top5_min. The constants are the
    100-question training slice, not the 300 eval questions.
    """
    stats = score_norm()[feature]
    z = (float(value) - stats["mean"]) / stats["std"]
    return float(np.clip(z, -SCORE_CLIP, SCORE_CLIP))


def _bm25_top_features(obs: dict[str, Any]) -> tuple[float, float, float]:
    """Standardized top-1, top-1−top-2 gap, and minimum of the top-5.

    An empty list stays 0. A single score has no gap, so that channel stays 0
    instead of looking like a real tie (a real gap of 0 is standardized).
    """
    scores = [float(s) for s in (obs.get("top_scores") or [])]
    if not scores:
        return 0.0, 0.0, 0.0
    top1_norm = standardize_score(scores[0], "top1")
    top5_min = standardize_score(min(scores[:5]), "top5_min")
    if len(scores) < 2:
        return top1_norm, 0.0, top5_min
    gap = max(scores[0] - scores[1], 0.0)
    return top1_norm, standardize_score(gap, "gap"), top5_min


def _verify_one_hot(verify: dict[str, Any] | None) -> tuple[float, float, float]:
    """support / contradiction / neutral. All zero before verify has run."""
    label = str((verify or {}).get("label") or "").strip().lower()
    if label == "support":
        return 1.0, 0.0, 0.0
    if label == "contradiction":
        return 0.0, 1.0, 0.0
    if label == "neutral":
        return 0.0, 0.0, 1.0
    return 0.0, 0.0, 0.0


def vectorize_structured_obs(obs: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    """16-d vector used by the env and the learned policy head.

    Score channels (mean, top-1, top-1−top-2 gap, top-5 minimum) are
    standardized with the training-slice mean and standard deviation and
    clipped to [-3, 3]. Before any retrieval those channels stay 0.

    The first 10 dims are the original snapshot, with the mean score on the
    new scale. Then top-1, the gap, a verify one-hot (support, contradiction,
    neutral), and the top-5 minimum.
    """
    max_steps = float(cfg.get("agent", {}).get("max_steps", 6))
    max_usd = float(cfg.get("budget", {}).get("max_usd", 0.05)) or 0.05
    verify = obs.get("verification") or {}
    mean_score = float(obs.get("mean_score") or 0.0)
    n_evidence = float(obs.get("n_evidence") or 0)
    # No passages yet: leave the channel at 0. Do not z-score a missing score.
    if n_evidence <= 0.0 and mean_score == 0.0:
        mean_norm = 0.0
    else:
        mean_norm = standardize_score(mean_score, "mean")
    counts = obs.get("counts") or {}
    top1_norm, gap_norm, top5_min_norm = _bm25_top_features(obs)
    oh_support, oh_contra, oh_neutral = _verify_one_hot(verify if obs.get("verification") else None)
    vec = np.array(
        [
            mean_norm,
            min(n_evidence / 10.0, 1.0),
            max(float(obs.get("remaining_steps") or 0) / max_steps, 0.0),
            min(max(float(obs.get("remaining_usd") or 0) / max_usd, 0.0), 1.0),
            float(verify.get("support") or 0.0),
            float(verify.get("contradiction") or 0.0),
            min(float(counts.get("retrieve", 0)) / 3.0, 1.0),
            min(float(counts.get("rewrite", 0)) / 2.0, 1.0),
            min(float(counts.get("rerank", 0)) / 2.0, 1.0),
            min(float(counts.get("verify", 0)) / 2.0, 1.0),
            top1_norm,
            gap_norm,
            oh_support,
            oh_contra,
            oh_neutral,
            top5_min_norm,
        ],
        dtype=np.float32,
    )
    if vec.size != OBS_DIM:
        raise ValueError(f"expected obs dim {OBS_DIM}, got {vec.size}")
    return vec


class AgenticRAGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict[str, Any],
        retriever,
        examples: list[dict[str, Any]],
        reward_computer: RewardComputer | None = None,
        seed: int | None = 42,
        agent: AgenticRAG | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.examples = examples
        self._owns_agent = agent is None
        self.agent = agent if agent is not None else AgenticRAG(cfg, retriever, reward_computer)
        self.reward_computer = self.agent.reward_computer
        self.action_space = spaces.Discrete(len(ACTIONS))
        # Compact vector observation for bandit/RL heads (Milestone 3+)
        # First 10: standardized mean_score, n_evidence/10, remaining_steps/max,
        # remaining_usd/max_usd, verify_support, verify_contra, retrieve/3,
        # rewrite/2, rerank/2, verify/2.
        # Then standardized top-1, standardized gap, verify one-hot, standardized top-5 min.
        # Score channels use [-3, 3]. The other channels sit inside [0, 1].
        self.observation_space = spaces.Box(
            low=-SCORE_CLIP, high=SCORE_CLIP, shape=(OBS_DIM,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._example: dict[str, Any] | None = None
        self._state: AgentState | None = None
        self._tracker: CostTracker | None = None
        self._trajectory: list[dict[str, Any]] = []
        self._example_index = 0

    def _vector_obs(self) -> np.ndarray:
        assert self._state is not None and self._tracker is not None
        return vectorize_structured_obs(self._state.observation(self._tracker, self.cfg), self.cfg)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if "example" in options:
            self._example = options["example"]
        elif "index" in options:
            self._example = self.examples[int(options["index"]) % len(self.examples)]
        else:
            idx = int(self._rng.integers(0, len(self.examples)))
            self._example = self.examples[idx]
            self._example_index = idx

        self._state = self.agent.new_state(self._example)
        self._tracker = CostTracker(self.cfg["price_card"])
        self._trajectory = []
        info = {
            "example_id": self._example.get("id"),
            "question": self._example.get("question"),
            "structured_obs": self._state.observation(self._tracker, self.cfg),
        }
        return self._vector_obs(), info

    def step(self, action: int | np.integer) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._state is not None and self._tracker is not None and self._example is not None
        action_i = int(action)
        action_name = IDX_TO_ACTION[action_i]
        max_steps = int(self.cfg.get("agent", {}).get("max_steps", 6))

        if self.cfg.get("budget", {}).get("force_stop_on_exhaustion", True) and self.agent.budget_exhausted(self._tracker):
            action_name = "stop"

        if self._state.step >= max_steps - 1 and action_name != "stop":
            action_name = "stop"

        self._state, step_info = self.agent.step(self._state, action_name, self._tracker)
        self._trajectory.append(step_info)

        terminated = bool(self._state.done)
        truncated = False
        reward = 0.0
        info: dict[str, Any] = {"action": action_name, "step_info": step_info}

        if terminated:
            abstained = self._state.stop_mode == "abstain"
            budget = {
                "max_usd": float(self.cfg.get("budget", {}).get("max_usd", 0.05)),
                "violated": self.agent.budget_exhausted(self._tracker),
            }
            rb = self.reward_computer.compute(
                pred=self._state.draft_answer,
                gold=self._example.get("answers") or self._example.get("answer"),
                evidence=self._state.evidence,
                supporting_titles=self._example.get("supporting_titles"),
                verify_out=self._state.verify_out,
                cost_summary=self._tracker.summary(),
                n_actions=len(self._state.action_history),
                verified=self._state.counts.get("verify", 0) > 0,
                abstained=abstained,
                budget=budget,
            )
            reward = float(rb.reward)
            info["episode_result"] = {
                "prediction": self._state.draft_answer,
                "gold": self._example.get("answers") or self._example.get("answer"),
                "trajectory": self._trajectory,
                "cost": self._tracker.summary(),
                "reward_breakdown": rb.as_dict(),
                "em": rb.components.get("em", 0.0),
                "f1": rb.components.get("f1", 0.0),
            }

        info["structured_obs"] = self._state.observation(self._tracker, self.cfg)
        return self._vector_obs(), reward, terminated, truncated, info

    def run_episode_with_action_sequence(self, example: dict[str, Any], actions: list[str]) -> dict[str, Any]:
        """Helper for deterministic pilots / unit tests."""
        obs, _ = self.reset(options={"example": example})
        total_reward = 0.0
        last_info: dict[str, Any] = {}
        for a in actions:
            obs, r, term, trunc, info = self.step(ACTION_TO_IDX[a])
            total_reward += float(r)
            last_info = info
            if term or trunc:
                break
        if not last_info.get("episode_result"):
            obs, r, term, trunc, info = self.step(ACTION_TO_IDX["stop"])
            total_reward += float(r)
            last_info = info
        result = last_info.get("episode_result", {})
        result["reward"] = total_reward
        return result

    def close(self) -> None:
        if self._owns_agent:
            closer = getattr(self.agent, "close", None)
            if callable(closer):
                closer()
        self.agent = None
