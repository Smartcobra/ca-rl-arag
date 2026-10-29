#!/usr/bin/env python3
"""Reproduce the verify-sensitivity softmax probe. No GPU, no retrieval.

Behind the paper sentence "the verify label moves the probabilities by at most
0.0068" is a single forward pass: take one real post-verify observation, set the
verify label to support, then to contradiction, and read the two softmaxes off
the frozen policy. This regenerates that number so it can be checked instead of
trusted.

The probe point is `learned_policy_frontier_lambda0.pt` (epoch 40) on the first
post-verify decision in `learned_frontier_lambda0.jsonl`. That checkpoint is a
**10-d** policy from the v2 family, so it predates the 15-d and 16-d
observations and `LearnedPolicy.load` refuses it on purpose. The 10-d vectorizer
is replayed here rather than imported; see `_legacy_vectorize_10d`.

Usage:
  python scripts/verify_softmax_probe.py            # check the committed JSON
  python scripts/verify_softmax_probe.py --write    # regenerate it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agentic_rag import ACTIONS
from src.config import load_config
from src.policies.learned import legal_mask
from src.rag_env import OBS_DIM

DEFAULT_CHECKPOINT = "results/checkpoints/learned_policy_frontier_lambda0.pt"
DEFAULT_TRAJECTORIES = "results/trajectories/learned_frontier_lambda0.jsonl"
DEFAULT_OUT = "results/metrics/verify_sensitivity_lambda0.json"
LEGACY_OBS_DIM = 10
# Probabilities are compared and stored at this many decimals. The published
# figure is 4-decimal, and raw float32 would make the committed file churn
# across torch builds.
ROUND_TO = 4


def _legacy_vectorize_10d(obs: dict[str, Any], label: str, cfg: dict[str, Any]) -> np.ndarray:
    """The 10-d observation as it stood at commit 23dc131, with the label forced.

    Deliberately a copy, not a call into `src.rag_env`. The live vectorizer is
    16-d and z-scores the BM25 channels against the training slice; running the
    v2 checkpoint through it would be a different experiment, and reshaping the
    live one to stay 10-d-compatible would hold today's code hostage to a frozen
    number. The cost of the copy is that this function must not be "fixed" to
    track `vectorize_structured_obs`.
    """
    if label not in ("support", "contradiction"):
        raise ValueError(f"label must be support or contradiction, got {label!r}")
    max_steps = float(cfg.get("agent", {}).get("max_steps", 6))
    max_usd = float(cfg.get("budget", {}).get("max_usd", 0.05)) or 0.05
    counts = obs.get("counts") or {}
    return np.array(
        [
            float(np.tanh(float(obs.get("mean_score") or 0.0) / 5.0)),
            min(float(obs.get("n_evidence") or 0) / 10.0, 1.0),
            max(float(obs.get("remaining_steps") or 0) / max_steps, 0.0),
            min(max(float(obs.get("remaining_usd") or 0) / max_usd, 0.0), 1.0),
            1.0 if label == "support" else 0.0,
            1.0 if label == "contradiction" else 0.0,
            min(float(counts.get("retrieve", 0)) / 3.0, 1.0),
            min(float(counts.get("rewrite", 0)) / 2.0, 1.0),
            min(float(counts.get("rerank", 0)) / 2.0, 1.0),
            min(float(counts.get("verify", 0)) / 2.0, 1.0),
        ],
        dtype=np.float32,
    )


def load_legacy_policy(path: Path) -> nn.Sequential:
    """Load a 10-d checkpoint straight into a bare net.

    `LearnedPolicy.load` rejects this file because the live head is 16-d. That
    guard is right for training and eval and wrong here, so this reads the
    state dict directly and checks the width itself.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("net."): v for k, v in payload["state_dict"].items()}
    in_dim = int(state["0.weight"].shape[1])
    if in_dim != LEGACY_OBS_DIM:
        raise SystemExit(
            f"{path} takes a {in_dim}-d observation; this probe replays the {LEGACY_OBS_DIM}-d one. "
            "Point --checkpoint at a v2-family checkpoint."
        )
    hidden = int(state["0.weight"].shape[0])
    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, len(ACTIONS)))
    net.load_state_dict(state)
    net.eval()
    return net


def find_probe_point(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str, int]:
    """First decision in the file the policy made with a verify result in hand.

    "Post-verify" is read off the observation (`counts.verify >= 1`), not off the
    action, so the step is the one where the label could have changed the choice.
    """
    for row in rows:
        for i, step in enumerate(row.get("trajectory") or []):
            obs = step.get("observation_before") or {}
            if int((obs.get("counts") or {}).get("verify", 0)) >= 1:
                return obs, str(row.get("id")), i
    raise SystemExit("no post-verify decision in the trajectory file; nothing to probe")


def _softmax_block(net: nn.Sequential, obs: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Raw and legal-mask-restricted probabilities for both verify labels."""
    raw: dict[str, np.ndarray] = {}
    masked: dict[str, np.ndarray] = {}
    for label in ("support", "contradiction"):
        vec = _legacy_vectorize_10d(obs, label, cfg)
        with torch.no_grad():
            logits = net(torch.as_tensor(vec).reshape(1, LEGACY_OBS_DIM))
        raw[label] = torch.softmax(logits, dim=-1).numpy().ravel()
        # legal_mask reads indices 1, 2 and 6-9 only, and the layout is
        # append-only, so a zero-padded 10-d vector masks like the real thing.
        padded = np.zeros(OBS_DIM, dtype=np.float32)
        padded[:LEGACY_OBS_DIM] = vec
        allow = torch.as_tensor(legal_mask(padded, cfg))
        masked[label] = torch.softmax(
            logits.masked_fill(~allow.unsqueeze(0), -1e9), dim=-1
        ).numpy().ravel()

    out = {}
    for name, block in (("raw", raw), ("masked", masked)):
        support, contradiction = block["support"], block["contradiction"]
        rounded = {
            lab: {a: round(float(p), ROUND_TO) for a, p in zip(ACTIONS, probs)}
            for lab, probs in block.items()
        }
        out[name] = {
            "support": rounded["support"],
            "contradiction": rounded["contradiction"],
            "max_abs_diff": round(float(np.max(np.abs(support - contradiction))), ROUND_TO),
            "same_to_two_decimals": all(
                round(rounded["support"][a], 2) == round(rounded["contradiction"][a], 2)
                for a in ACTIONS
            ),
            "argmax_support": ACTIONS[int(np.argmax(support))],
            "argmax_contradiction": ACTIONS[int(np.argmax(contradiction))],
        }
    return out


def build_payload(checkpoint: Path, trajectories: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in trajectories.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    obs, question_id, step_index = find_probe_point(rows)
    net = load_legacy_policy(checkpoint)
    epoch = torch.load(checkpoint, map_location="cpu", weights_only=True).get("epoch")

    payload: dict[str, Any] = {
        "checkpoint": checkpoint.name,
        "epoch": epoch,
        "actions": list(ACTIONS),
        "observation_before": {
            "n_evidence": obs.get("n_evidence"),
            "mean_score": obs.get("mean_score"),
            "remaining_usd": obs.get("remaining_usd"),
            "remaining_steps": obs.get("remaining_steps"),
            "counts": obs.get("counts"),
        },
    }
    payload.update(_softmax_block(net, obs, cfg))
    payload["provenance"] = {
        "script": "scripts/verify_softmax_probe.py",
        "trajectories": trajectories.name,
        "question_id": question_id,
        "step_index": step_index,
        "selection": "first decision whose observation has counts.verify >= 1",
        "obs_dim": LEGACY_OBS_DIM,
        "vectorizer": "src/rag_env.py at commit 23dc131 (pre-15-d), replayed in this script",
        "note": (
            "v2-family checkpoint. The live observation is 16-d with z-scored BM25 "
            "channels, so this number cannot be regenerated through src.rag_env."
        ),
    }
    return payload


def _diff(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            out += _diff(expected.get(key), actual.get(key), f"{path}.{key}" if path else key)
        return out
    return [] if expected == actual else [f"  {path}: committed {expected!r}, recomputed {actual!r}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trajectories", default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--write", action="store_true", help="Write the JSON instead of checking it.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint, trajectories, out = (
        p if (p := Path(x)).is_absolute() else ROOT / x
        for x in (args.checkpoint, args.trajectories, args.out)
    )
    for path in (checkpoint, trajectories):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    payload = build_payload(checkpoint, trajectories, cfg)
    prov = payload["provenance"]
    print(f"checkpoint: {payload['checkpoint']} (epoch {payload['epoch']}, {prov['obs_dim']}-d)")
    print(f"probe:      {prov['trajectories']} {prov['question_id']} step {prov['step_index']}")
    for name in ("raw", "masked"):
        block = payload[name]
        print(
            f"{name:7s} max |P(support) - P(contradiction)| = {block['max_abs_diff']:.4f}"
            f"   argmax {block['argmax_support']} / {block['argmax_contradiction']}"
        )

    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out.relative_to(ROOT)}")
        return

    if not out.exists():
        raise SystemExit(f"\n{out} does not exist. Run with --write to create it.")
    committed = json.loads(out.read_text(encoding="utf-8"))
    differences = _diff(committed, payload)
    if differences:
        raise SystemExit(
            f"\n{out.relative_to(ROOT)} does not match this run:\n"
            + "\n".join(differences)
            + "\n\nRerun with --write if the change is intended."
        )
    print(f"\n{out.relative_to(ROOT)} matches this run.")


if __name__ == "__main__":
    main()
