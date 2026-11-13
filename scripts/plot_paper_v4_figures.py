#!/usr/bin/env python3
"""Paper figures for paper_v4, drawn from the logged trajectories.

Writes paper_v4/figures/repairs.png and paper_v4/figures/frontier.png.
Does not touch the unsuffixed baseline metric files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "results" / "trajectories"
OUT = ROOT / "paper_v4" / "figures"

SOFT = {
    "both": "#A9D6C2",
    "repair": "#8ECAE6",
    "break": "#F4B183",
    "neither": "#E4E4E4",
    "ink": "#3D3D3D",
    "one": "#1F4E79",
    "learned": "#4C9BD4",
    "rule": "#E07A3D",
    "max": "#6A994E",
    "recipe": "#9C6644",
    "switch": "#B0B6BC",
    "oracle": "#1F4E79",
    "adaptive": "#6F2C91",
    "oracle2": "#C47B2B",
    "retrieve": "#9B2226",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def repairs() -> None:
    one = load_jsonl(TRAJ / "baseline_default_v4.jsonl")
    learned = load_jsonl(TRAJ / "learned_frontier_lambda20_v4s42.jsonl")
    order = [
        ("hotpot_qa", "HotpotQA"),
        ("natural_questions", "Natural Questions"),
    ]
    keys = ["both", "repair", "break", "neither"]
    labels = {
        "both": "Both correct",
        "repair": "Recipe repairs",
        "break": "Recipe breaks",
        "neither": "Both wrong",
    }
    counts = {name: {key: 0 for key in keys} for _, name in order}
    for example_id, base in one.items():
        other = learned[example_id]
        dataset = "HotpotQA" if base["dataset"] == "hotpot_qa" else "Natural Questions"
        base_ok = float(base["em"]) >= 1
        other_ok = float(other["em"]) >= 1
        if base_ok and other_ok:
            key = "both"
        elif other_ok:
            key = "repair"
        elif base_ok:
            key = "break"
        else:
            key = "neither"
        counts[dataset][key] += 1

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    left = [0, 0]
    names = [name for _, name in order]
    for key in keys:
        values = [counts[name][key] for name in names]
        bars = ax.barh(names, values, left=left, color=SOFT[key], height=0.55, label=labels[key])
        for bar, value, start in zip(bars, values, left):
            if value >= 5:
                ax.text(
                    start + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(value),
                    ha="center",
                    va="center",
                    color=SOFT["ink"],
                    fontsize=9,
                )
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, 150)
    ax.set_xlabel("Questions, out of 150")
    ax.tick_params(colors=SOFT["ink"])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "repairs.png", dpi=200)
    print("repairs", counts)


def frontier() -> None:
    def summary(name: str) -> tuple[float, float]:
        blob = json.loads((ROOT / "results" / "metrics" / name).read_text())
        row = blob["summary"]
        return float(row["mean_total_usd"]) * 1e4, float(row["n_correct"]) / 300

    one_shot = summary("baseline_default_v4.json")
    rule = summary("rule_based_default_v4.json")
    max_tools = summary("max_tools_default_v4.json")
    recipe = summary("fixed_recipe_default_v4.json")
    learned = summary("learned_frontier_lambda20_v4s42.json")
    # Dataset switch: one-shot on HotpotQA, six-step recipe on Natural Questions.
    one = load_jsonl(TRAJ / "baseline_default_v4.jsonl")
    six = load_jsonl(TRAJ / "fixed_recipe_default_v4.jsonl")
    switch_cost = 0.0
    for row in one.values():
        pick = row if row["dataset"] == "hotpot_qa" else six[row["id"]]
        switch_cost += float(pick["total_usd"])
    switch = (switch_cost / 300 * 1e4, 100 / 300)
    # Oracle: cheapest correct among one-shot, max tools, and the learned recipe.
    pools = [
        one,
        load_jsonl(TRAJ / "max_tools_default_v4.jsonl"),
        load_jsonl(TRAJ / "learned_frontier_lambda20_v4s42.jsonl"),
    ]
    correct = 0
    cost = 0.0
    ids = list(one)
    for example_id in ids:
        cands = [pool[example_id] for pool in pools]
        good = [row for row in cands if float(row["em"]) >= 1]
        pick = min(good or cands, key=lambda row: float(row["total_usd"]))
        correct += int(float(pick["em"]) >= 1)
        cost += float(pick["total_usd"])
    oracle = (cost / 300 * 1e4, correct / 300)
    # Oracle of one-shot and the learned recipe only. Adaptive-RAG is the
    # question-only router already logged in adaptive_rag_v4.json.
    pair_correct = 0
    pair_cost = 0.0
    learned_rows = load_jsonl(TRAJ / "learned_frontier_lambda20_v4s42.jsonl")
    for example_id in ids:
        cands = [one[example_id], learned_rows[example_id]]
        good = [row for row in cands if float(row["em"]) >= 1]
        pick = min(good or cands, key=lambda row: float(row["total_usd"]))
        pair_correct += int(float(pick["em"]) >= 1)
        pair_cost += float(pick["total_usd"])
    oracle_pair = (pair_cost / 300 * 1e4, pair_correct / 300)
    adaptive = summary("adaptive_rag_v4.json")
    # Retrieve-first controller, same 83,120-passage index. Seed 42 is the
    # cost sweep. Seeds 43 and 44 are the other two λ=20 exams.
    retrieve_first = [
        (0, "learned_frontier_lambda0_rf0s42.json", "42"),
        (5, "learned_frontier_lambda5_rf5s42.json", "42"),
        (20, "learned_frontier_lambda20_rfs42.json", "42"),
        (20, "learned_frontier_lambda20_rfs43.json", "43"),
        (20, "learned_frontier_lambda20_rfs44.json", "44"),
        (80, "learned_frontier_lambda80_rf80s42.json", "42"),
        (20, "learned_frontier_lambda20_rfinert45.json", "45"),
    ]
    rf_points = [(lam, seed, summary(name)) for lam, name, seed in retrieve_first]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = [
        (one_shot, "s", SOFT["one"], "One-shot"),
        (learned, "o", SOFT["learned"], r"Rewrite-first $\lambda=20$"),
        (adaptive, "*", SOFT["adaptive"], "Adaptive-RAG"),
        (switch, "o", SOFT["switch"], "Dataset switch, 100"),
        (recipe, "D", SOFT["recipe"], "Six-step recipe"),
        (rule, "s", SOFT["rule"], "Rule"),
        (max_tools, "^", SOFT["max"], "Max tools"),
    ]
    for (x, y), marker, color, label in series:
        face = "none" if label.startswith("Dataset") else color
        ax.scatter(
            [x],
            [y],
            s=90,
            marker=marker,
            facecolors=face,
            edgecolors=color,
            linewidths=1.6,
            zorder=3,
            label=label,
        )
    ax.scatter(
        [oracle_pair[0]],
        [oracle_pair[1]],
        s=120,
        marker="D",
        facecolors="none",
        edgecolors=SOFT["oracle2"],
        linewidths=1.8,
        zorder=4,
        label=f"Oracle of two, {pair_correct}",
    )
    ax.scatter(
        [oracle[0]],
        [oracle[1]],
        s=140,
        marker="o",
        facecolors="none",
        edgecolors=SOFT["oracle"],
        linewidths=1.8,
        zorder=4,
        label=f"Oracle with max tools, {correct}",
    )
    ax.annotate(str(pair_correct), oracle_pair, textcoords="offset points", xytext=(8, -2), fontsize=9, color=SOFT["ink"])
    ax.annotate(str(correct), oracle, textcoords="offset points", xytext=(8, -2), fontsize=9, color=SOFT["ink"])
    labeled = False
    # λ=80 and λ=20 seed 43 share the one-shot point. Leave them unlabeled.
    offsets = {
        (0, "42"): (0, 9),
        (5, "42"): (8, -8),
        (20, "42"): (-28, 8),
        (20, "44"): (8, 4),
        (20, "45"): (2, -13),
    }
    for lam, seed, (x, y) in rf_points:
        on_oneshot = abs(x - one_shot[0]) < 0.05 and abs(y - one_shot[1]) < 0.002
        ax.scatter(
            [x],
            [y],
            s=110 if on_oneshot else 70,
            marker="h",
            facecolors="none" if on_oneshot else SOFT["retrieve"],
            edgecolors=SOFT["retrieve"],
            linewidths=1.8 if on_oneshot else 1.2,
            zorder=5,
            label="Retrieve-first" if not labeled else None,
        )
        labeled = True
        if (lam, seed) in offsets:
            if seed == "45":
                tag = "masked"
            elif seed == "42":
                tag = rf"$\lambda={lam}$"
            else:
                tag = rf"$\lambda={lam}$, seed {seed}"
            ax.annotate(
                tag,
                (x, y),
                textcoords="offset points",
                xytext=offsets[(lam, seed)],
                fontsize=8,
                color=SOFT["ink"],
                ha="center" if (lam, seed) in {(0, "42"), (20, "45")} else "left",
            )
        print(f"retrieve-first λ={lam} seed {seed}", round(x, 2), round(y, 3))
    ax.set_xlim(1.2, 8.6)
    ax.set_ylim(0.30, 0.46)
    ax.set_xlabel(r"Mean dollars per question ($\times 10^{-4}$)")
    ax.set_ylabel("Exact match")
    ax.tick_params(colors=SOFT["ink"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "frontier.png", dpi=200, bbox_inches="tight")
    print("oracle", correct, round(oracle[0], 2), "pair", pair_correct, round(oracle_pair[0], 2))


def curves() -> None:
    """Greedy validation reward for the three λ=20 seeds."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = {42: SOFT["one"], 43: SOFT["learned"], 44: SOFT["recipe"]}
    for seed in (42, 43, 44):
        blob = json.loads(
            (ROOT / "results" / "metrics" / f"train_policy_curve_frontier_v4_lambda20_s{seed}.json").read_text()
        )
        epochs = [int(row["epoch"]) for row in blob]
        reward = [float(row["eval_reward"]) for row in blob]
        ax.plot(epochs, reward, color=colors[seed], linewidth=1.6, label=f"Seed {seed}")
        best = max(reward)
        chosen = epochs[reward.index(best)]
        ax.scatter([chosen], [best], s=40, color=colors[seed], zorder=3)
        print(f"seed {seed} selected epoch {chosen} eval_reward {best:.5f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Greedy reward on the 180")
    ax.set_xticks(range(1, 21, 2))
    ax.tick_params(colors=SOFT["ink"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "train_curves.png", dpi=200, bbox_inches="tight")
    print("curves")


def retrieve_first_curves() -> None:
    """Greedy validation reward for the seven retrieve-first runs."""
    jobs = (
        (20, 42, "train_policy_curve_retrieve_first_lambda20_s42.json", "#1F4E79", "λ=20, seed 42"),
        (20, 43, "train_policy_curve_retrieve_first_lambda20_s43.json", "#4C9BD4", "λ=20, seed 43"),
        (20, 44, "train_policy_curve_retrieve_first_lambda20_s44.json", "#9C6644", "λ=20, seed 44"),
        (0, 42, "train_policy_curve_retrieve_first_lambda0_s42.json", "#6A994E", "λ=0, seed 42"),
        (5, 42, "train_policy_curve_retrieve_first_lambda5_s42.json", "#E07A3D", "λ=5, seed 42"),
        (80, 42, "train_policy_curve_retrieve_first_lambda80_s42.json", "#9B2226", "λ=80, seed 42"),
        (20, 45, "train_policy_curve_retrieve_first_inert_lambda20_s45.json", "#6F2C91", "λ=20, seed 45, masked"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    metrics = ROOT / "results" / "metrics"
    for _lam, seed, name, color, label in jobs:
        blob = json.loads((metrics / name).read_text())
        epochs = [int(row["epoch"]) for row in blob]
        reward = [float(row["eval_reward"]) for row in blob]
        ax.plot(
            epochs,
            reward,
            color=color,
            linewidth=1.5,
            linestyle="--" if seed == 45 else "-",
            label=label,
        )
        best = max(reward)
        chosen = epochs[reward.index(best)]
        ax.scatter([chosen], [best], s=36, color=color, zorder=3)
        print(f"{label} selected epoch {chosen} eval_reward {best:.5f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Greedy reward on the 180")
    ax.set_xticks(range(1, 21, 2))
    ax.tick_params(colors=SOFT["ink"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "train_curves_retrieve_first.png", dpi=200, bbox_inches="tight")
    print("retrieve-first curves")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    repairs()
    frontier()
    curves()
    retrieve_first_curves()


if __name__ == "__main__":
    main()
