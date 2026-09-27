#!/usr/bin/env python3
"""Convert pilot / ablation metric runs into figures under results/figs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import DATASET_LABELS, ordered_dataset_names

POLICY_ORDER = ("naive_rag", "rule_based", "max_tools", "learned")
POLICY_LABELS = {
    "naive_rag": "naive RAG",
    "rule_based": "rule-based",
    "max_tools": "max-tools",
    "always_max": "max-tools",  # legacy key in older metric dumps
    "learned": "learned",
}
ABLATION_ORDER = (
    "correctness_only",
    "correctness_grounding",
    "correctness_faithfulness_cost",
    "default",
    "lambda_zero",
    "high_cost_pressure",
)
COLORS = {
    "naive_rag": "#2A6F97",
    "rule_based": "#E07A3D",
    "max_tools": "#3A5A40",
    "always_max": "#3A5A40",
    "learned": "#7B2D8E",
    "frontier_lambda0": "#7B2D8E",
    "frontier_lambda20": "#9B4DB0",
    "frontier_act02": "#C77DFF",
    # Historical 2026-09-13 λ=80 family (failed sweep; kept so old JSON still plots).
    "frontier_act0": "#7B2D8E",
    "frontier_act005": "#9B4DB0",
}

# Frozen ranking anchors + learned family for the headline EM-vs-$ figure.
# Order is free → mid → expensive so the line, if it exists, reads left-to-right.
FRONTIER_FROZEN = ("naive_rag", "rule_based", "max_tools")
FRONTIER_LEARNED = (
    ("frontier_lambda0", "learned λ=0"),
    ("frontier_lambda20", "learned λ=20"),
    ("frontier_act02", "learned λ=80 act=0.02"),
)
FRONTIER_PRESET_META = {
    "frontier_lambda0": {"lambda_cost": 0.0, "act_penalty": 0.0},
    "frontier_lambda20": {"lambda_cost": 20.0, "act_penalty": 0.0},
    "frontier_act02": {"lambda_cost": 80.0, "act_penalty": 0.02},
}
FRONTIER_FROZEN_FILES = {
    "naive_rag": "baseline_default.json",
    "rule_based": "rule_based_default.json",
    "max_tools": "max_tools_default.json",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _normalize_results(results: dict) -> dict:
    """Map legacy always_max key → max_tools for plotting."""
    out = dict(results)
    if "always_max" in out and "max_tools" not in out:
        out["max_tools"] = out.pop("always_max")
    elif "always_max" in out and "max_tools" in out:
        out.pop("always_max")
    return out


def _ordered_policies(results: dict) -> list[str]:
    known = [p for p in POLICY_ORDER if p in results]
    extra = [p for p in results if p not in known]
    return known + sorted(extra)


def plot_policy_quality_reward(results: dict, out_dir: Path) -> None:
    policies = _ordered_policies(results)
    labels = [POLICY_LABELS.get(p, p) for p in policies]
    x = np.arange(len(policies))
    width = 0.25
    em = [results[p]["mean_em"] for p in policies]
    f1 = [results[p]["mean_f1"] for p in policies]
    reward = [results[p]["mean_reward"] for p in policies]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width, em, width, label="mean EM", color="#4C6EF5")
    ax.bar(x, f1, width, label="mean F1", color="#12B886")
    ax.bar(x + width, reward, width, label="mean reward", color="#F08C00")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score")
    ax.set_title("Policy comparison — quality & reward (overall, mix-weighted)")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=min(0.0, min(reward) - 0.05))
    ax.axhline(0.0, color="#888", linewidth=0.6)
    fig.tight_layout()
    _save(fig, out_dir / "policy_quality_reward.png")


def plot_policy_cost(results: dict, out_dir: Path) -> None:
    policies = _ordered_policies(results)
    labels = [POLICY_LABELS.get(p, p) for p in policies]
    usd = [results[p]["mean_total_usd"] for p in policies]
    tokens = [results[p]["mean_total_tokens"] for p in policies]
    colors = [COLORS.get(p, "#666") for p in policies]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].bar(labels, usd, color=colors)
    axes[0].set_ylabel("Mean USD / example")
    axes[0].set_title("Cost ($)")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, tokens, color=colors)
    axes[1].set_ylabel("Mean tokens / example")
    axes[1].set_title("Tokens")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Policy comparison — efficiency (overall, mix-weighted)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "policy_cost.png")


def plot_action_mix(results: dict, out_dir: Path) -> None:
    policies = _ordered_policies(results)
    labels = [POLICY_LABELS.get(p, p) for p in policies]
    actions = ("mean_n_retrieve", "mean_n_rewrite", "mean_n_rerank", "mean_n_verify")
    action_labels = ("retrieve", "rewrite", "rerank", "verify")
    palette = ("#2A6F97", "#90BE6D", "#F9C74F", "#F94144")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bottom = np.zeros(len(policies))
    for key, name, color in zip(actions, action_labels, palette):
        vals = np.array([results[p].get(key, 0.0) for p in policies], dtype=float)
        ax.bar(labels, vals, bottom=bottom, label=name, color=color)
        bottom += vals
    ax.set_ylabel("Mean actions / example")
    ax.set_title("Action mix by policy (overall, mix-weighted)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir / "policy_action_mix.png")


def plot_reward_components(results: dict, out_dir: Path) -> None:
    policies = _ordered_policies(results)
    labels = [POLICY_LABELS.get(p, p) for p in policies]
    keys = ("mean_q_ans", "mean_q_ground", "mean_q_cal", "mean_p_hall")
    names = ("Q_ans", "Q_ground", "Q_cal", "P_hall")
    x = np.arange(len(policies))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, (key, name) in enumerate(zip(keys, names)):
        vals = [results[p].get(key, 0.0) for p in policies]
        ax.bar(x + (i - 1.5) * width, vals, width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean component")
    ax.set_title("Reward components by policy (overall, mix-weighted)")
    ax.axhline(0.0, color="#888", linewidth=0.6)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    _save(fig, out_dir / "policy_reward_components.png")


def _summary_block(blob: dict) -> dict:
    if isinstance(blob, dict) and "summary" in blob and isinstance(blob["summary"], dict):
        return blob["summary"]
    return blob


def _action_sequence(row: dict) -> tuple:
    return tuple(step.get("action") for step in (row.get("trajectory") or []))


def behavior_from_rows(rows: list[dict]) -> dict:
    """How many action recipes a policy used, and what followed a contradiction."""
    sequences: dict[tuple, int] = {}
    next_actions: dict[str, int] = {}
    n_contradiction = 0
    saw_verify = False
    for row in rows:
        seq = _action_sequence(row)
        sequences[seq] = sequences.get(seq, 0) + 1
        traj = row.get("trajectory") or []
        for i, step in enumerate(traj):
            if step.get("action") == "verify" or step.get("verify"):
                saw_verify = True
            label = (step.get("verify") or {}).get("label")
            if label != "contradiction":
                continue
            n_contradiction += 1
            nxt = traj[i + 1]["action"] if i + 1 < len(traj) else "end"
            next_actions[nxt] = next_actions.get(nxt, 0) + 1
    if not saw_verify:
        after = "no verify"
    elif n_contradiction == 0:
        after = "no contradiction"
    elif set(next_actions) <= {"stop"}:
        after = "stop"
    elif set(next_actions) <= {"verify", "stop"}:
        # Scripted extra verify, then stop. Never retrieve or rewrite.
        after = "stop"
    else:
        after = ",".join(f"{name}:{count}" for name, count in sorted(next_actions.items()))
    return {
        "n_action_sequences": len(sequences),
        "after_contradiction": after,
        "n_contradiction": n_contradiction,
        "contradiction_next": next_actions,
    }


def _row_em(row: dict) -> int:
    return int(row.get("em") or 0)


def _row_usd(row: dict) -> float:
    return float(row.get("total_usd") or 0.0)


def _ceiling_point(name: str, label: str, n_correct: int, mean_usd: float, n_examples: int) -> dict:
    return {
        "name": name,
        "label": label,
        "n_correct": n_correct,
        "n_examples": n_examples,
        "mean_em": (n_correct / n_examples) if n_examples else 0.0,
        "mean_total_usd": mean_usd,
        "kind": "ceiling",
    }


def controller_ceilings(naive_rows: list[dict], lambda0_rows: list[dict], max_rows: list[dict]) -> list[dict]:
    """Headroom if a controller could pick among fixed recipes per question.

    Dollar coordinate: pay the cheapest policy that got the question right.
    If none did, pay the cheapest policy. The 110 point is the dataset switch
    (naive on Hotpot, the λ=0 recipe on NQ), not that per-question pick.
    """
    naive = {row["id"]: row for row in naive_rows}
    lambda0 = {row["id"]: row for row in lambda0_rows}
    max_tools = {row["id"]: row for row in max_rows}
    ids = [row["id"] for row in naive_rows]
    n = len(ids)

    switch_usd = 0.0
    switch_correct = 0
    for qid in ids:
        src = naive[qid] if naive[qid].get("dataset") == "hotpot_qa" else lambda0[qid]
        switch_usd += _row_usd(src)
        switch_correct += _row_em(src)

    def oracle(pools: list[dict[str, dict]]) -> tuple[int, float]:
        correct = 0
        usd = 0.0
        for qid in ids:
            cands = [pool[qid] for pool in pools]
            good = [cand for cand in cands if _row_em(cand)]
            chosen = min(good or cands, key=_row_usd)
            usd += _row_usd(chosen)
            correct += _row_em(chosen)
        return correct, (usd / n if n else 0.0)

    pair_correct, pair_usd = oracle([naive, lambda0])
    triple_correct, triple_usd = oracle([naive, lambda0, max_tools])
    return [
        _ceiling_point(
            "naive_hotpot_lambda0_nq",
            str(switch_correct),
            switch_correct,
            switch_usd / n if n else 0.0,
            n,
        ),
        _ceiling_point("best_of_naive_or_lambda0", str(pair_correct), pair_correct, pair_usd, n),
        _ceiling_point(
            "best_of_naive_lambda0_max_tools",
            str(triple_correct),
            triple_correct,
            triple_usd,
            n,
        ),
    ]


def _base_preset(key: str) -> str:
    for name, _label in FRONTIER_LEARNED:
        if key == name or key.startswith(name + "_"):
            return name
    return key.removesuffix("_best")


def _learned_row(stats: dict, preset: str, *, checkpoint: str, extra: dict | None = None) -> dict:
    base_preset = _base_preset(preset)
    row = {
        "mean_em": float(stats.get("mean_em") or 0.0),
        "mean_total_usd": float(stats.get("mean_total_usd") or 0.0),
        "mean_n_steps": float(stats.get("mean_n_steps") or 0.0),
        "mean_n_retrieve": float(stats.get("mean_n_retrieve") or 0.0),
        "mean_n_verify": float(stats.get("mean_n_verify") or 0.0),
        "n_correct": float(stats.get("n_correct") or 0.0),
        "n_examples": float(stats.get("n_examples") or 0.0),
        "kind": "learned",
        "checkpoint": checkpoint,
        "preset": base_preset,
        "reward_preset": base_preset,
        **dict(FRONTIER_PRESET_META.get(base_preset) or {}),
    }
    if extra:
        row.update(extra)
    return row


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_frontier_table(metrics_dir: Path, tag: str | None = None) -> dict:
    """Build the EM-vs-$ table from frozen ranking JSON + per-preset learned evals.

    ``tag`` reads ``learned_{preset}_{tag}.json`` as the selected row and
    ``learned_{preset}_{tag}last.json`` as the last-epoch row. The unsuffixed
    v2 files stay in place.
    """
    metrics_dir = Path(metrics_dir)
    traj_dir = metrics_dir.parent / "trajectories"
    tag = (tag or "").strip().strip("_")
    frozen: dict[str, dict] = {}
    for name, fname in FRONTIER_FROZEN_FILES.items():
        path = metrics_dir / fname
        if not path.exists():
            continue
        stats = _summary_block(_load_json(path))
        frozen[name] = {
            "mean_em": float(stats.get("mean_em") or 0.0),
            "mean_total_usd": float(stats.get("mean_total_usd") or 0.0),
            "mean_n_steps": float(stats.get("mean_n_steps") or 0.0),
            "mean_n_retrieve": float(stats.get("mean_n_retrieve") or 0.0),
            "mean_n_verify": float(stats.get("mean_n_verify") or 0.0),
            "n_correct": float(stats.get("n_correct") or 0.0),
            "n_examples": float(stats.get("n_examples") or 0.0),
            "kind": "frozen",
        }
    learned: dict[str, dict] = {}
    if tag:
        pairs = ((f"_{tag}", "best"), (f"_{tag}last", "last"))
    else:
        pairs = (("", "last"), ("_best", "best"))
    for preset, _label in FRONTIER_LEARNED:
        for suffix, checkpoint in pairs:
            key = f"{preset}{suffix}"
            path = metrics_dir / f"learned_{key}.json"
            if not path.exists():
                continue
            blob = _load_json(path)
            stats = _summary_block(blob)
            meta = blob.get("meta") if isinstance(blob, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            extra = {
                "epoch": meta.get("epoch"),
                "n_action_sequences": meta.get("n_action_sequences"),
                "after_contradiction": meta.get("after_contradiction"),
            }
            traj_path = traj_dir / f"learned_{key}.jsonl"
            if traj_path.exists():
                extra.update(behavior_from_rows(_load_jsonl(traj_path)))
            learned[key] = _learned_row(stats, key, checkpoint=checkpoint, extra=extra)
    ceilings: list[dict] = []
    naive_path = traj_dir / "learned_frontier_act02.jsonl"
    lambda0_path = traj_dir / "learned_frontier_lambda0.jsonl"
    max_path = traj_dir / "max_tools_default.jsonl"
    if naive_path.exists() and lambda0_path.exists() and max_path.exists():
        ceilings = controller_ceilings(
            _load_jsonl(naive_path),
            _load_jsonl(lambda0_path),
            _load_jsonl(max_path),
        )
    return {
        "lambda_cost": [FRONTIER_PRESET_META[p]["lambda_cost"] for p, _ in FRONTIER_LEARNED],
        "act_penalty": [FRONTIER_PRESET_META[p]["act_penalty"] for p, _ in FRONTIER_LEARNED],
        "presets": [p for p, _ in FRONTIER_LEARNED],
        "frozen": frozen,
        "learned": learned,
        "ceilings": ceilings,
        "selection_rule": (
            f"Score the {tag} exam from _best.pt, chosen by greedy reward on the validation slice "
            "before opening the 300. The last epoch is a second row."
            if tag
            else
            "Score <stem>_best.pt, the max train-greedy eval_reward. "
            "Choose it before opening the 300. The last epoch is a second row, not the selected point."
        ),
        "tag": tag or None,
    }


def _selected_learned_key(learned: dict, preset: str) -> str | None:
    """The point we are allowed to claim: best exam if it exists, else last epoch."""
    for key, row in learned.items():
        if str(row.get("preset")) == preset and row.get("checkpoint") == "best":
            return key
    if f"{preset}_best" in learned:
        return f"{preset}_best"
    if preset in learned and learned[preset].get("checkpoint") != "last":
        return preset
    for key, row in learned.items():
        if str(row.get("preset")) == preset and row.get("checkpoint") == "last":
            return key
    if preset in learned:
        return preset
    return None


def plot_frontier_em_usd(table: dict, out_dir: Path, filename: str = "frontier_em_usd.png") -> None:
    """Headline figure: frozen anchors, selected checkpoints, last-epoch rows, ceilings."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    frozen = table.get("frozen") or {}
    learned = table.get("learned") or {}
    for name in FRONTIER_FROZEN:
        if name not in frozen:
            continue
        row = frozen[name]
        ax.scatter(
            row["mean_total_usd"],
            row["mean_em"],
            s=110,
            marker="s",
            color=COLORS.get(name, "#666"),
            label=POLICY_LABELS.get(name, name),
            zorder=3,
        )
        ax.annotate(
            POLICY_LABELS.get(name, name),
            (row["mean_total_usd"], row["mean_em"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    xs, ys = [], []
    selected_labels_at: dict[tuple[float, float], list[str]] = {}
    for preset, label in FRONTIER_LEARNED:
        key = _selected_learned_key(learned, preset)
        if key is None:
            continue
        row = learned[key]
        is_best = row.get("checkpoint") == "best" or key.endswith("_best")
        selected_label = f"{label} best" if is_best else label
        xs.append(row["mean_total_usd"])
        ys.append(row["mean_em"])
        ax.scatter(
            row["mean_total_usd"],
            row["mean_em"],
            s=100,
            marker="o",
            color=COLORS.get(preset, "#7B2D8E"),
            label=selected_label,
            zorder=4,
        )
        spot = (round(row["mean_total_usd"], 8), round(row["mean_em"], 6))
        selected_labels_at.setdefault(spot, []).append(selected_label)
        last_key = preset if preset in learned and preset != key else None
        if last_key is None:
            for candidate, cand_row in learned.items():
                if candidate == key:
                    continue
                if cand_row.get("preset") == preset and cand_row.get("checkpoint") == "last":
                    last_key = candidate
                    break
        last = learned.get(last_key) if last_key else None
        if is_best and last is not None:
            ax.scatter(
                last["mean_total_usd"],
                last["mean_em"],
                s=70,
                marker="x",
                color=COLORS.get(preset, "#7B2D8E"),
                label=f"{label} last",
                zorder=4,
            )
            ax.annotate(
                f"{label} last",
                (last["mean_total_usd"], last["mean_em"]),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=8,
            )
    for (x, y), labels in selected_labels_at.items():
        ax.annotate(
            " / ".join(labels),
            (x, y),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=8,
        )
    if len(xs) >= 2:
        ax.plot(xs, ys, color="#7B2D8E", linewidth=1.2, alpha=0.7, zorder=2, label="selected checkpoints")
    ceiling_labeled = False
    for row in table.get("ceilings") or []:
        ax.scatter(
            row["mean_total_usd"],
            row["mean_em"],
            s=160,
            marker="o",
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.6,
            label="controller ceiling" if not ceiling_labeled else None,
            zorder=5,
        )
        ceiling_labeled = True
        ax.annotate(
            str(row.get("label") or row.get("n_correct")),
            (row["mean_total_usd"], row["mean_em"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Mean USD / example")
    ax.set_ylabel("Mean EM")
    ax.set_title("Selected checkpoint vs last epoch (hollow = controller ceiling)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, Path(out_dir) / filename)


def plot_pareto(results: dict, out_dir: Path) -> None:
    policies = _ordered_policies(results)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for p in policies:
        x = results[p]["mean_total_usd"]
        y = results[p]["mean_em"]
        ax.scatter(x, y, s=90, color=COLORS.get(p, "#666"), label=POLICY_LABELS.get(p, p), zorder=3)
        ax.annotate(POLICY_LABELS.get(p, p), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Mean USD / example")
    ax.set_ylabel("Mean EM")
    ax.set_title("Quality–cost Pareto (overall EM, mix-weighted)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir / "policy_pareto_em_usd.png")


def _metric_from_slice(policy_stats: dict, key: str, dataset: str | None) -> float:
    if dataset is None:
        val = policy_stats.get(key)
    else:
        val = (policy_stats.get("by_dataset") or {}).get(dataset, {}).get(key)
    if val is None:
        return 0.0
    return float(val)


def plot_policy_by_dataset(results: dict, out_dir: Path) -> None:
    """Grouped bars: Hotpot vs NQ vs overall for EM/F1/reward/abstain. Overall is mix-weighted."""
    policies = _ordered_policies(results)
    if not any(isinstance(results[p].get("by_dataset"), dict) and results[p]["by_dataset"] for p in policies):
        print("Skip by-dataset figure (no by_dataset in summary)")
        return

    labels = [POLICY_LABELS.get(p, p) for p in policies]
    x = np.arange(len(policies))
    width = 0.25
    hop = "natural_questions"
    for policy in policies:
        present = set((results[policy].get("by_dataset") or {}).keys())
        for name in ("natural_questions", "trivia_qa", "squad"):
            if name in present:
                hop = name
                break
    series = [
        (None, "Overall (mix-weighted)", "#6C757D"),
        ("hotpot_qa", "HotpotQA", "#2A6F97"),
        (hop, DATASET_LABELS.get(hop, hop), "#E07A3D"),
    ]
    panels = (
        ("mean_em", "Exact match"),
        ("mean_f1", "Token F1"),
        ("mean_reward", "Mean reward"),
        ("abstain_rate", "Abstain rate"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for i, (ds, name, color) in enumerate(series):
            vals = [_metric_from_slice(results[p], key, ds) for p in policies]
            ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.axhline(0.0, color="#888", linewidth=0.6)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Policy comparison by dataset — overall is mix-weighted; read Hotpot and single-hop separately",
        y=1.07,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "policy_by_dataset.png")


def plot_reward_ablation_by_dataset(by_dataset_table: dict, out_dir: Path) -> None:
    datasets = ordered_dataset_names(by_dataset_table)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        table = by_dataset_table[ds]
        presets = [p for p in ABLATION_ORDER if p in table] + [p for p in table if p not in ABLATION_ORDER]
        rewards = [table[p]["mean_reward"] for p in presets]
        labels = [p.replace("_", "\n") for p in presets]
        bars = ax.bar(labels, rewards, color="#5C7CFA")
        for bar, val in zip(bars, rewards):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_ylabel("Mean reward")
        ymax = max(rewards) * 1.18 if rewards else 1.0
        ax.set_ylim(0, ymax)
    fig.suptitle("Reward-weight ablation by dataset (fixed rule_based)")
    fig.tight_layout()
    _save(fig, out_dir / "reward_ablation_by_dataset.png")


def write_figures(pilot_path: Path, out_dir: Path, ablation_path: Path | None = None) -> None:
    """Render pilot (and optional ablation) figures from already-written metric JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = Path(pilot_path)
    if not pilot_path.exists():
        raise FileNotFoundError(f"Missing pilot summary: {pilot_path}")

    pilot = _load_json(pilot_path)
    results = pilot.get("results") or pilot
    if not isinstance(results, dict) or not results:
        raise ValueError(f"No policy results in {pilot_path}")
    results = _normalize_results(results)

    plot_policy_quality_reward(results, out_dir)
    plot_policy_cost(results, out_dir)
    plot_action_mix(results, out_dir)
    plot_reward_components(results, out_dir)
    plot_pareto(results, out_dir)
    plot_policy_by_dataset(results, out_dir)

    if ablation_path is not None:
        ablation_path = Path(ablation_path)
        if ablation_path.exists():
            ablation = _load_json(ablation_path)
            plot_reward_ablation(ablation, out_dir)
            nested = {k: v.get("by_dataset") for k, v in ablation.items() if isinstance(v, dict) and v.get("by_dataset")}
            sibling = ablation_path.with_name("reward_ablation_by_dataset.json")
            if sibling.exists():
                plot_reward_ablation_by_dataset(_load_json(sibling), out_dir)
            elif nested:
                by_ds: dict[str, dict] = {}
                for preset, ds_map in nested.items():
                    for ds, stats in ds_map.items():
                        by_ds.setdefault(ds, {})[preset] = stats
                plot_reward_ablation_by_dataset(by_ds, out_dir)
            else:
                print("Skip ablation by-dataset figure (no by_dataset in table)")
        else:
            print(f"Skip ablation figure (missing {ablation_path})")

    print(f"Done. Figures in {out_dir}")


def plot_reward_ablation(table: dict, out_dir: Path) -> None:
    presets = [p for p in ABLATION_ORDER if p in table] + [p for p in table if p not in ABLATION_ORDER]
    rewards = [table[p]["mean_reward"] for p in presets]
    labels = [p.replace("_", "\n") for p in presets]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(labels, rewards, color="#5C7CFA")
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean reward")
    ax.set_title("Reward-weight ablation (overall, mix-weighted; fixed rule_based)")
    ax.set_ylim(0, max(rewards) * 1.18 if rewards else 1.0)
    fig.tight_layout()
    _save(fig, out_dir / "reward_ablation.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pilot/ablation metrics into results/figs")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--out-dir", default="results/figs")
    parser.add_argument("--pilot-summary", default=None, help="Override pilot summary JSON path")
    parser.add_argument("--ablation-table", default=None, help="Override ablation table JSON path")
    parser.add_argument(
        "--frontier",
        action="store_true",
        help="Also write frontier_sweep_table.json + frontier_em_usd.png from learned_frontier_*.json.",
    )
    parser.add_argument(
        "--frontier-tag",
        default=None,
        help="Read learned_{preset}_{tag}.json as the selected exam and "
        "{tag}last as the last epoch. Writes frontier_sweep_table_{tag}.json "
        "and frontier_em_usd_{tag}.png so the unsuffixed v2 figure stays.",
    )
    args = parser.parse_args()

    metrics_dir = (ROOT / args.metrics_dir).resolve() if not Path(args.metrics_dir).is_absolute() else Path(args.metrics_dir)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_path = Path(args.pilot_summary) if args.pilot_summary else metrics_dir / "pilot_summary_default.json"
    ablation_path = Path(args.ablation_table) if args.ablation_table else metrics_dir / "reward_ablation_table.json"

    if not pilot_path.is_absolute():
        pilot_path = ROOT / pilot_path
    if not ablation_path.is_absolute():
        ablation_path = ROOT / ablation_path

    try:
        write_figures(pilot_path, out_dir, ablation_path)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    if args.frontier or args.frontier_tag:
        tag = (args.frontier_tag or "").strip().strip("_")
        table = collect_frontier_table(metrics_dir, tag or None)
        table_name = f"frontier_sweep_table_{tag}.json" if tag else "frontier_sweep_table.json"
        table_path = metrics_dir / table_name
        table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"Wrote {table_path}")
        if table.get("learned"):
            fig_name = f"frontier_em_usd_{tag}.png" if tag else "frontier_em_usd.png"
            plot_frontier_em_usd(table, out_dir, fig_name)
        else:
            print("Skip frontier figure (no learned frontier JSON yet)")


if __name__ == "__main__":
    main()
