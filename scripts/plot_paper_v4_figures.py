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

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    series = [
        (one_shot, "s", SOFT["one"], "One-shot"),
        (learned, "o", SOFT["learned"], r"Learned $\lambda=20$, three seeds"),
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
        [oracle[0]],
        [oracle[1]],
        s=140,
        marker="o",
        facecolors="none",
        edgecolors=SOFT["oracle"],
        linewidths=1.8,
        zorder=4,
        label=f"Oracle, {correct}",
    )
    ax.annotate(str(correct), oracle, textcoords="offset points", xytext=(8, -2), fontsize=9, color=SOFT["ink"])
    ax.set_xlim(1.2, 8.6)
    ax.set_ylim(0.30, 0.46)
    ax.set_xlabel(r"Mean dollars per question ($\times 10^{-4}$)")
    ax.set_ylabel("Exact match")
    ax.tick_params(colors=SOFT["ink"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "frontier.png", dpi=200)
    print("oracle", correct, round(oracle[0], 2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    repairs()
    frontier()


if __name__ == "__main__":
    main()
