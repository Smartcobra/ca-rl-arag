"""Gymnasium RL environment wrapping Agentic RAG.

Episode = one QA example from initial query → stop (or budget/step exhaustion).
Action space (V1 frozen): retrieve | rewrite | rerank | verify | stop
Observation: compact state features (Scope Memo V2 §5).
Reward: trajectory reward assigned at episode end (sparse); optional step shaping=0 for V1.
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agentic_rag import ACTIONS, AgenticRAG, AgentState
from .cost import CostTracker
from .rewards import RewardComputer


ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}


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
        # [mean_score, n_evidence/10, remaining_steps/max, remaining_usd/max_usd,
        #  verify_support, verify_contra, retrieve_count/3, rewrite/2, rerank/2, verify/2]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._example: dict[str, Any] | None = None
        self._state: AgentState | None = None
        self._tracker: CostTracker | None = None
        self._trajectory: list[dict[str, Any]] = []
        self._example_index = 0

    def _vector_obs(self) -> np.ndarray:
        assert self._state is not None and self._tracker is not None
        obs = self._state.observation(self._tracker, self.cfg)
        max_steps = float(self.cfg.get("agent", {}).get("max_steps", 6))
        max_usd = float(self.cfg.get("budget", {}).get("max_usd", 0.05)) or 0.05
        verify = obs.get("verification") or {}
        # Normalize mean_score roughly (BM25 scores vary); squash with tanh-like
        mean_score = float(obs.get("mean_score") or 0.0)
        mean_norm = float(np.tanh(mean_score / 5.0))
        vec = np.array(
            [
                mean_norm,
                min(float(obs.get("n_evidence") or 0) / 10.0, 1.0),
                max(float(obs.get("remaining_steps") or 0) / max_steps, 0.0),
                min(max(float(obs.get("remaining_usd") or 0) / max_usd, 0.0), 1.0),
                float(verify.get("support") or 0.0),
                float(verify.get("contradiction") or 0.0),
                min(float(obs.get("counts", {}).get("retrieve", 0)) / 3.0, 1.0),
                min(float(obs.get("counts", {}).get("rewrite", 0)) / 2.0, 1.0),
                min(float(obs.get("counts", {}).get("rerank", 0)) / 2.0, 1.0),
                min(float(obs.get("counts", {}).get("verify", 0)) / 2.0, 1.0),
            ],
            dtype=np.float32,
        )
        return vec

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
