#!/usr/bin/env python3
"""Figures and tables for the retrieve-first λ=20 exams, seeds 42, 43, 44.

Reads the committed exam json, the trajectory jsonl, and the training curves.
Writes PNGs under results/figs/. Does not touch paper_v4 figures.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
TRAJ = ROOT / "results" / "trajectories"
OUT = ROOT / "results" / "figs"

INK = "#243040"
HEADER = "#D6E8F6"
LINE = "#C5D0DA"
ZEBRA = "#F4F8FB"
BOLD = "#142033"
ONE = "#1F4E79"
SEEDS_COLOR = {42: "#4C9BD4", 43: "#6A994E", 44: "#E07A3D"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def second_action(row: dict) -> tuple[str | None, str | None, float | None]:
    traj = row.get("trajectory") or []
    if len(traj) < 2:
        return None, None, None
    step = traj[1]
    obs = step.get("observation_before") or {}
    score = obs.get("mean_score")
    return traj[0].get("action"), step.get("action"), None if score is None else float(score)


def branch_info(rows: list[dict]) -> dict:
    pairs = []
    n_first = 0
    for row in rows:
        first, action, score = second_action(row)
        if first == "retrieve":
            n_first += 1
        if action in {"rewrite", "retrieve"} and score is not None:
            pairs.append((score, action))
    n_rewrite = sum(action == "rewrite" for _, action in pairs)
    n_retrieve = sum(action == "retrieve" for _, action in pairs)
    scores = sorted({score for score, _ in pairs})
    candidates = scores + ([scores[-1] + 1] if scores else [])
    best_hits, best_t = -1, None
    for threshold in candidates:
        hits = sum(
            (("rewrite" if score < threshold else "retrieve") == action)
            for score, action in pairs
        )
        if hits > best_hits:
            best_hits, best_t = hits, threshold
    return {
        "n_first_retrieve": n_first,
        "paired": len(pairs),
        "n_rewrite": n_rewrite,
        "n_retrieve": n_retrieve,
        "threshold": best_t,
        "hits": best_hits,
        "branch": n_rewrite >= 15 and n_retrieve >= 15 and best_hits >= 250,
    }


def selected_epoch(seed: int) -> dict:
    curve = json.loads(
        (METRICS / f"train_policy_curve_retrieve_first_lambda20_s{seed}.json").read_text()
    )
    best_reward = max(float(row["eval_reward"]) for row in curve)
    kept = next(row for row in curve if float(row["eval_reward"]) == best_reward)
    return kept


def exam(seed: int) -> dict:
    blob = json.loads((METRICS / f"learned_frontier_lambda20_rfs{seed}.json").read_text())
    rows = load_jsonl(TRAJ / f"learned_frontier_lambda20_rfs{seed}.jsonl")
    summary = blob["summary"]
    kept = selected_epoch(seed)
    sequences = Counter(
        tuple(step.get("action") for step in row.get("trajectory") or []) for row in rows
    )
    seconds = Counter(second_action(row)[1] for row in rows)
    return {
        "seed": seed,
        "summary": summary,
        "rows": rows,
        "kept": kept,
        "sequences": sequences,
        "seconds": seconds,
        "branch": branch_info(rows),
    }


def usd_4(value: float) -> str:
    return f"{value * 1e4:.2f}"


def draw_table(path: Path, headers: list[str], rows: list[list[str]], col_w: list[float],
               wraps: list[int] | None = None) -> None:
    if wraps is None:
        wraps = [80] * len(headers)

    def wrap(text: str, width: int) -> str:
        if "\n" in text or len(text) <= width:
            return text
        parts: list[str] = []
        line = ""
        for word in text.split():
            trial = word if not line else line + " " + word
            if len(trial) <= width:
                line = trial
            else:
                if line:
                    parts.append(line)
                line = word
        if line:
            parts.append(line)
        return "\n".join(parts)

    headers_w = [wrap(h, w) for h, w in zip(headers, wraps)]
    rows_w = [[wrap(cell, w) for cell, w in zip(row, wraps)] for row in rows]

    def nlines(text: str) -> int:
        return max(1, text.count("\n") + 1)

    line_h = 0.22
    pad = 0.08
    header_h = max(nlines(h) for h in headers_w) * line_h + pad
    body_h = [max(nlines(c) for c in row) * line_h + pad for row in rows_w]
    total_h = header_h + sum(body_h)
    total_w = sum(col_w)
    fig, ax = plt.subplots(figsize=(total_w, total_h + 0.02), dpi=220)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    y = total_h
    x0 = 0.0
    y -= header_h
    for header, width in zip(headers_w, col_w):
        ax.add_patch(Rectangle((x0, y), width, header_h, facecolor=HEADER,
                               edgecolor=LINE, linewidth=0.6))
        ax.text(x0 + width / 2, y + header_h / 2, header, ha="center", va="center",
                fontsize=8.0, color=BOLD, fontweight="bold", linespacing=1.12)
        x0 += width
    for r, row in enumerate(rows_w):
        h = body_h[r]
        y -= h
        x0 = 0.0
        face = ZEBRA if r % 2 else "white"
        for cell, width in zip(row, col_w):
            ax.add_patch(Rectangle((x0, y), width, h, facecolor=face,
                                   edgecolor=LINE, linewidth=0.6))
            ax.text(x0 + width / 2, y + h / 2, cell, ha="center", va="center",
                    fontsize=7.6, color=INK, linespacing=1.12)
            x0 += width
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(path)


def plot_em_usd(exams: list[dict], one_shot: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=220)
    x_one = one_shot["mean_total_usd"] * 1e4
    y_one = one_shot["n_correct"]
    for item in exams:
        summary = item["summary"]
        x = summary["mean_total_usd"] * 1e4
        y = summary["n_correct"]
        color = SEEDS_COLOR[item["seed"]]
        ax.scatter([x], [y], s=64, color=color, zorder=3)
        label = f"Seed {item['seed']}\n{int(y)} / 300"
        offset = (8, 8) if item["seed"] != 43 else (14, 10)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                    fontsize=8, color=color)
    ax.scatter(
        [x_one],
        [y_one],
        s=130,
        facecolors="none",
        edgecolors=ONE,
        linewidths=1.6,
        zorder=4,
    )
    ax.annotate(
        "One-shot, same point as seed 43",
        (x_one, y_one),
        textcoords="offset points",
        xytext=(14, -16),
        fontsize=8,
        color=ONE,
    )
    ax.set_xlabel("Mean dollars per question  ($ × 10⁻⁴)")
    ax.set_ylabel("Correct answers on the locked 300")
    ax.set_xlim(1.2, 6.4)
    ax.set_ylim(96, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.set_title("Retrieve-first λ = 20 exams", loc="left", color=BOLD, fontsize=11)
    fig.tight_layout()
    path = OUT / "retrieve_first_em_usd.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    print(path)


def plot_second_action(exams: list[dict]) -> None:
    actions = ["rewrite", "retrieve", "verify", "stop"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=220)
    import numpy as np

    x = np.arange(len(actions))
    width = 0.24
    for i, item in enumerate(exams):
        counts = [item["seconds"].get(action, 0) for action in actions]
        ax.bar(
            x + (i - 1) * width,
            counts,
            width,
            color=SEEDS_COLOR[item["seed"]],
            label=f"Seed {item['seed']}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.set_ylabel("Questions (of 300)")
    ax.set_ylim(0, 340)
    ax.axhline(15, color="#B0B6BC", linewidth=0.8, linestyle="--")
    ax.text(3.45, 22, "15", color="#8A929A", fontsize=7.5, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Second action after the forced retrieve", loc="left", color=BOLD, fontsize=11)
    fig.tight_layout()
    path = OUT / "retrieve_first_second_action.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    print(path)


def main() -> None:
    exams = [exam(seed) for seed in (42, 43, 44)]
    one_shot = json.loads((METRICS / "baseline_default_v4.json").read_text())["summary"]
    one_rows = {row["id"]: row for row in load_jsonl(TRAJ / "baseline_default_v4.jsonl")}

    table_rows = []
    compare_rows = []
    for item in exams:
        s = item["summary"]
        hot = s["by_dataset"]["hotpot_qa"]["n_correct"]
        nq = s["by_dataset"]["natural_questions"]["n_correct"]
        info = item["branch"]
        top_seq, top_n = item["sequences"].most_common(1)[0]
        branch_cell = "yes" if info["branch"] else "no"
        if info["paired"]:
            branch_cell = f"no, {info['hits']}/{info['paired']}"
        table_rows.append([
            str(item["seed"]),
            str(item["kept"]["epoch"]),
            f"{int(hot)}/150",
            f"{int(nq)}/150",
            f"{int(s['n_correct'])}/300",
            usd_4(s["mean_total_usd"]),
            f"{s['mean_n_steps']:.2f}".rstrip("0").rstrip("."),
            f"{s['mean_n_retrieve']:.2f}".rstrip("0").rstrip("."),
            f"{s['mean_n_rewrite']:.2f}".rstrip("0").rstrip("."),
            f"{s['mean_n_verify']:.2f}".rstrip("0").rstrip("."),
            str(len(item["sequences"])),
            branch_cell,
        ])
        both = repair = broken = 0
        for row in item["rows"]:
            base_ok = float(one_rows[row["id"]]["em"]) >= 1
            ours_ok = float(row["em"]) >= 1
            if base_ok and ours_ok:
                both += 1
            elif ours_ok:
                repair += 1
            elif base_ok:
                broken += 1
        compare_rows.append([
            str(item["seed"]),
            " → ".join(top_seq) + f"\n({top_n}/300)",
            f"+{repair} / −{broken}",
            f"{both + repair + broken}/300",
            f"{item['seconds'].get('rewrite', 0)} / {item['seconds'].get('retrieve', 0)}",
        ])

    draw_table(
        OUT / "retrieve_first_exams_table.png",
        ["Seed", "Epoch", "Hotpot", "NQ", "Correct", "$ ×10⁻⁴", "Steps",
         "Retrieve", "Rewrite", "Verify", "Seq.", "Branch"],
        table_rows,
        [0.55, 0.62, 0.72, 0.72, 0.85, 0.78, 0.62, 0.78, 0.75, 0.68, 0.55, 1.15],
    )
    draw_table(
        OUT / "retrieve_first_vs_oneshot_table.png",
        ["Seed", "Most common sequence", "Vs one-shot", "Union", "2nd rewrite / retrieve"],
        compare_rows,
        [0.6, 5.6, 1.1, 1.0, 1.8],
        [8, 70, 16, 12, 24],
    )
    plot_em_usd(exams, one_shot)
    plot_second_action(exams)
    n_branch = sum(item["branch"]["branch"] for item in exams)
    print(f"branch count {n_branch} of 3")


if __name__ == "__main__":
    main()
