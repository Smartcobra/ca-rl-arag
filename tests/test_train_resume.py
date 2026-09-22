#!/usr/bin/env python3
"""Resume helpers for train_policy.py. No GPU."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.policies.learned import LearnedPolicy
from train_policy import (
    _last_finished_epoch,
    _load_curve,
    _trainer_state_path,
    load_trainer_state,
    save_trainer_state,
)


def test_resume_saves_and_loads_trainer_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = tmp_path / "learned_policy.pt"
        trainer = _trainer_state_path(ckpt)
        assert trainer.name == "learned_policy_trainer.pt"

        curve_path = tmp_path / "train_policy_curve.json"
        curve_path.write_text(
            json.dumps([{"epoch": 1, "mean_reward": 0.1}, {"epoch": 2, "mean_reward": 0.2}]),
            encoding="utf-8",
        )
        assert _last_finished_epoch(_load_curve(curve_path)) == 2

        policy = LearnedPolicy(hidden=8, seed=0)
        opt = torch.optim.Adam(policy.parameters(), lr=0.003)
        dummy = torch.zeros(1, 10)
        loss = policy.mlp(dummy).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        before = {k: v.clone() for k, v in opt.state_dict()["state"][0].items() if torch.is_tensor(v)}

        save_trainer_state(
            trainer,
            epoch=2,
            optimizer=opt,
            baseline=0.42,
            best_eval_reward=0.55,
        )
        policy.save(ckpt, extra={"epoch": 2})

        # Change optimizer so a bad load would be visible.
        loss = policy.mlp(dummy).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        loaded_policy = LearnedPolicy.load(ckpt)
        loaded_opt = torch.optim.Adam(loaded_policy.parameters(), lr=0.003)
        state = load_trainer_state(trainer)
        loaded_opt.load_state_dict(state["optimizer"])

        assert int(state["epoch"]) == 2
        assert float(state["baseline"]) == 0.42
        assert float(state["best_eval_reward"]) == 0.55
        after = loaded_opt.state_dict()["state"][0]
        assert torch.allclose(before["exp_avg"], after["exp_avg"])
        assert _last_finished_epoch(_load_curve(curve_path)) + 1 == int(state["epoch"]) + 1


def main() -> None:
    test_resume_saves_and_loads_trainer_state()
    print("TRAIN RESUME OK")


if __name__ == "__main__":
    main()
