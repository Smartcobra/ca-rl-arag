#!/usr/bin/env python3
"""Unit tests for the tiny REINFORCE policy. No GPU, no eval slice."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_rag import ACTIONS
from src.policies.learned import (
    MAX_HIDDEN,
    MAX_PARAM_COUNT,
    OBS_DIM,
    LearnedPolicy,
    PolicyMLP,
    assert_train_only_path,
    legal_mask,
    parameter_count,
)
from src.rag_env import ACTION_TO_IDX

_AGENT = {
    "max_steps": 8,
    "max_retrieve": 3,
    "max_rewrite": 2,
    "max_rerank": 2,
    "max_verify": 2,
}
_CFG = {"agent": _AGENT}


def _empty_evidence_obs(*, remaining_frac: float = 1.0) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[2] = remaining_frac
    return obs


def test_mlp_shape_and_size() -> None:
    mlp = PolicyMLP(16)
    n = parameter_count(mlp)
    assert n == 261
    assert n <= MAX_PARAM_COUNT
    out = mlp(torch.randn(4, OBS_DIM))
    assert tuple(out.shape) == (4, len(ACTIONS))


def test_hidden_cap() -> None:
    try:
        PolicyMLP(MAX_HIDDEN + 1)
    except ValueError as exc:
        assert str(MAX_HIDDEN) in str(exc)
        return
    raise AssertionError("expected ValueError for oversized hidden")


def test_legal_mask_empty_evidence() -> None:
    mask = legal_mask(_empty_evidence_obs(), _CFG)
    assert mask[ACTION_TO_IDX["retrieve"]]
    assert mask[ACTION_TO_IDX["rewrite"]]
    assert not mask[ACTION_TO_IDX["rerank"]]
    assert not mask[ACTION_TO_IDX["verify"]]
    assert not mask[ACTION_TO_IDX["stop"]]


def test_legal_mask_last_step() -> None:
    mask = legal_mask(_empty_evidence_obs(remaining_frac=1.0 / 8.0), _CFG)
    assert mask[ACTION_TO_IDX["stop"]]
    assert mask.sum() == 1


def test_act_respects_mask() -> None:
    pol = LearnedPolicy(hidden=16, seed=0)
    obs = _empty_evidence_obs()
    for _ in range(20):
        action, logp, ent = pol.act(obs, cfg=_CFG)
        assert action in {ACTION_TO_IDX["retrieve"], ACTION_TO_IDX["rewrite"]}
        assert torch.isfinite(logp)
        assert torch.isfinite(ent)


def test_reinforce_step_finite() -> None:
    pol = LearnedPolicy(hidden=16, seed=0)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-2)
    obs = _empty_evidence_obs()
    _action, logp, ent = pol.act(obs, cfg=_CFG)
    loss = -(logp * 1.0) - 0.01 * ent
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()


def test_assert_train_only_path() -> None:
    p = assert_train_only_path("data/processed/train_slice.jsonl")
    assert p.name == "train_slice.jsonl"
    try:
        assert_train_only_path("data/processed/eval_slice.jsonl")
    except ValueError as exc:
        assert "eval" in str(exc).lower()
        return
    raise AssertionError("expected ValueError for eval_slice.jsonl")


def test_save_load_roundtrip(tmp_path=None) -> None:
    if tmp_path is None:
        dest = ROOT / "results" / "checkpoints" / "_test_learned.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cleanup = True
    else:
        dest = Path(tmp_path) / "learned.pt"
        cleanup = False
    pol = LearnedPolicy(hidden=8, seed=7)
    pol.save(dest, extra={"epoch": 1, "mean_reward": 0.1})
    loaded = LearnedPolicy.load(dest)
    assert loaded.hidden == 8
    x = torch.zeros(1, OBS_DIM)
    assert torch.allclose(pol.mlp(x), loaded.mlp(x))
    if cleanup and dest.exists():
        dest.unlink()


def main() -> None:
    test_mlp_shape_and_size()
    test_hidden_cap()
    test_legal_mask_empty_evidence()
    test_legal_mask_last_step()
    test_act_respects_mask()
    test_reinforce_step_finite()
    test_assert_train_only_path()
    test_save_load_roundtrip()
    print("LEARNED POLICY OK")


if __name__ == "__main__":
    main()
