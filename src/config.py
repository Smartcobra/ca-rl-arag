"""Shared config loading for CA-RL-ARAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: str | Path = "configs/default.yaml",
    reward_path: str | Path = "configs/reward_weights.yaml",
    price_path: str | Path = "configs/price_card.yaml",
    reward_preset: str | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    rewards = load_yaml(reward_path)
    prices = load_yaml(price_path)

    preset_name = reward_preset or cfg.get("reward", {}).get("preset", "default")
    presets = rewards.get("presets", {})
    if preset_name not in presets:
        raise KeyError(f"Unknown reward preset '{preset_name}'. Options: {list(presets)}")

    cfg["reward_weights"] = dict(presets[preset_name])
    cfg["reward_preset_name"] = preset_name
    cfg["reward_ablation_presets"] = list(presets.keys())
    cfg["pareto_sweep"] = rewards.get("pareto_sweep", {})
    cfg["price_card"] = prices
    cfg["root"] = str(ROOT)
    return cfg


def resolve_path(cfg: dict[str, Any], rel: str | Path) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return Path(cfg["root"]) / p
