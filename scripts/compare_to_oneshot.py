#!/usr/bin/env python3
"""Compare a learned exam with one-shot. No GPU.

Every row uses the same two sentences:

Repairs are questions one-shot got wrong and this row got right. Breaks are
the reverse. The paired sign test is an exact two-sided binomial test on those
discordant questions, success probability 1/2. The 95% interval is a percentile
bootstrap of exact match: 10000 draws with replacement from these questions,
seed 42, percentiles 2.5 and 97.5.

The dollar column is the price card (mean_total_usd). The millisecond column
is the logged wall clock (mean_total_latency_ms). The reward uses the latency
table, so these two numbers are reported side by side.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
TRAJ = ROOT / "results" / "trajectories"
FIGS = ROOT / "results" / "figs"

BASELINE_METRICS = METRICS / "baseline_default_v4.json"
BASELINE_TRAJ = TRAJ / "baseline_default_v4.jsonl"

N_BOOT = 10_000
BOOT_SEED = 42

METHOD = (
    "Repairs are questions one-shot got wrong and this row got right. "
    "Breaks are the reverse. "
    "The paired sign test is an exact two-sided binomial test on those discordant "
    "questions, success probability 1/2. "
    "The 95% interval is a percentile bootstrap of exact match: "
    "10000 draws with replacement from these questions, seed 42, "
    "percentiles 2.5 and 97.5."
)

# One seed. λ=20 keeps the suffix already committed as rfs42.
SWEEP = (
    {
        "lambda": 0,
        "preset": "frontier_lambda0",
        "seed": 42,
        "metrics": METRICS / "learned_frontier_lambda0_rf0s42.json",
        "traj": TRAJ / "learned_frontier_lambda0_rf0s42.jsonl",
        "curve": METRICS / "train_policy_curve_retrieve_first_lambda0_s42.json",
    },
    {
        "lambda": 5,
        "preset": "frontier_lambda5",
        "seed": 42,
        "metrics": METRICS / "learned_frontier_lambda5_rf5s42.json",
        "traj": TRAJ / "learned_frontier_lambda5_rf5s42.jsonl",
        "curve": METRICS / "train_policy_curve_retrieve_first_lambda5_s42.json",
    },
    {
        "lambda": 20,
        "preset": "frontier_lambda20",
        "seed": 42,
        "metrics": METRICS / "learned_frontier_lambda20_rfs42.json",
        "traj": TRAJ / "learned_frontier_lambda20_rfs42.jsonl",
        "curve": METRICS / "train_policy_curve_retrieve_first_lambda20_s42.json",
    },
    {
        "lambda": 80,
        "preset": "frontier_lambda80",
        "seed": 42,
        "metrics": METRICS / "learned_frontier_lambda80_rf80s42.json",
        "traj": TRAJ / "learned_frontier_lambda80_rf80s42.jsonl",
        "curve": METRICS / "train_policy_curve_retrieve_first_lambda80_s42.json",
    },
)

INK = "#243040"
HEADER = "#D6E8F6"
LINE = "#C5D0DA"
ZEBRA = "#F4F8FB"
BOLD = "#142033"
ONE = "#1F4E79"
LAM_COLOR = {0: "#6A994E", 5: "#1F4E79", 20: "#4C9BD4", 80: "#E07A3D"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def exact_sign_pvalue(repairs: int, breaks: int) -> float:
    """Two-sided exact binomial p-value. Ties are already excluded."""
    n = repairs + breaks
    if n == 0:
        return 1.0
    observed = math.comb(n, repairs) / (2**n)
    total = 0.0
    for k in range(n + 1):
        prob = math.comb(n, k) / (2**n)
        if prob <= observed + 1e-15:
            total += prob
    return min(1.0, total)


def bootstrap_em(correct: list[int], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Return (rate, low, high) from a percentile bootstrap."""
    arr = np.asarray(correct, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("exact match bootstrap needs at least one question")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_boot, arr.size))
    rates = arr[draws].mean(axis=1)
    low, high = np.quantile(rates, [0.025, 0.975])
    return float(arr.mean()), float(low), float(high)


def _correct(row: dict) -> int:
    return int(float(row["em"]) >= 1.0)


def paired_counts(baseline: list[dict], learned: list[dict]) -> dict:
    base = {row["id"]: row for row in baseline}
    missing = [row["id"] for row in learned if row["id"] not in base]
    if missing:
        raise SystemExit(f"{len(missing)} learned ids are missing from one-shot. First: {missing[0]}")
    if len(base) != len(learned):
        extra = sorted(set(base) - {row["id"] for row in learned})
        raise SystemExit(f"one-shot has {len(extra)} ids that this exam does not. First: {extra[0]}")
    repairs = breaks = both = neither = 0
    by_dataset = {}
    order = []
    for row in learned:
        other = base[row["id"]]
        ours = _correct(row)
        theirs = _correct(other)
        order.append(ours)
        dataset = row.get("dataset") or "unknown"
        slot = by_dataset.setdefault(dataset, {"repairs": 0, "breaks": 0})
        if ours and theirs:
            both += 1
        elif ours and not theirs:
            repairs += 1
            slot["repairs"] += 1
        elif theirs and not ours:
            breaks += 1
            slot["breaks"] += 1
        else:
            neither += 1
    rate, low, high = bootstrap_em(order)
    n = len(order)
    return {
        "n": n,
        "n_correct": int(sum(order)),
        "em": rate,
        "em_ci95": [low, high],
        "n_correct_ci95": [low * n, high * n],
        "repairs": repairs,
        "breaks": breaks,
        "both_correct": both,
        "both_wrong": neither,
        "sign_test_p": exact_sign_pvalue(repairs, breaks),
        "by_dataset": by_dataset,
        "method": METHOD,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
    }


def selected_epoch(curve_path: Path | None) -> dict | None:
    if curve_path is None or not curve_path.exists():
        return None
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    best_reward = max(float(row["eval_reward"]) for row in curve)
    kept = next(row for row in curve if float(row["eval_reward"]) == best_reward)
    return {
        "epoch": kept.get("epoch"),
        "eval_reward": kept.get("eval_reward"),
        "eval_split": kept.get("eval_split"),
    }


def cost_fields(metrics_path: Path | None) -> dict:
    if metrics_path is None or not metrics_path.exists():
        return {"mean_total_usd": None, "mean_total_latency_ms": None}
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))["summary"]
    return {
        "mean_total_usd": summary.get("mean_total_usd"),
        "mean_total_latency_ms": summary.get("mean_total_latency_ms"),
        "hotpot_correct": (summary.get("by_dataset") or {}).get("hotpot_qa", {}).get("n_correct"),
        "nq_correct": (summary.get("by_dataset") or {}).get("natural_questions", {}).get("n_correct"),
    }


def one_row(label: str, learned_traj: Path, learned_metrics: Path | None, curve: Path | None) -> dict:
    learned = load_jsonl(learned_traj)
    baseline = load_jsonl(BASELINE_TRAJ)
    compared = paired_counts(baseline, learned)
    picked = selected_epoch(curve)
    if picked is not None and picked.get("eval_split") not in (None, "valid_greedy"):
        raise SystemExit(
            f"{label} selected on {picked['eval_split']}, not valid_greedy."
        )
    row = {
        "label": label,
        "trajectory": str(learned_traj.relative_to(ROOT)),
        **compared,
        **cost_fields(learned_metrics),
        "selected_epoch": None if picked is None else picked["epoch"],
        "valid_greedy_reward": None if picked is None else picked["eval_reward"],
    }
    return row


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 1e4:.2f}"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.0f}"


def _fmt_ci(row: dict) -> str:
    low, high = row["em_ci95"]
    return f"{low:.3f}–{high:.3f}"


def print_row(row: dict) -> None:
    print(METHOD)
    print(
        f"{row['label']}  {row['n_correct']}/{row['n']}"
        f"  repairs {row['repairs']}  breaks {row['breaks']}"
        f"  sign-test p {_fmt_p(row['sign_test_p'])}"
        f"  exact-match 95% interval {_fmt_ci(row)}"
        f"  $ {row['mean_total_usd']}  latency_ms {row['mean_total_latency_ms']}"
    )
    if row.get("selected_epoch") is not None:
        print(f"  kept epoch {row['selected_epoch']}  valid greedy reward {row['valid_greedy_reward']}")
    for name, slot in sorted(row["by_dataset"].items()):
        print(f"  {name}  repairs {slot['repairs']}  breaks {slot['breaks']}")


def oneshot_row() -> dict:
    rows = load_jsonl(BASELINE_TRAJ)
    correct = [_correct(row) for row in rows]
    rate, low, high = bootstrap_em(correct)
    n = len(correct)
    costs = cost_fields(BASELINE_METRICS)
    return {
        "label": "one-shot",
        "trajectory": str(BASELINE_TRAJ.relative_to(ROOT)),
        "n": n,
        "n_correct": int(sum(correct)),
        "em": rate,
        "em_ci95": [low, high],
        "n_correct_ci95": [low * n, high * n],
        "repairs": 0,
        "breaks": 0,
        "both_correct": int(sum(correct)),
        "both_wrong": n - int(sum(correct)),
        "sign_test_p": None,
        "by_dataset": {},
        "method": METHOD,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "selected_epoch": None,
        "valid_greedy_reward": None,
        **costs,
    }


def draw_table(path: Path, headers: list[str], rows: list[list[str]], col_w: list[float]) -> None:
    line_h = 0.28
    pad = 0.08
    header_h = line_h + pad
    body_h = line_h + pad
    total_h = header_h + body_h * len(rows)
    total_w = sum(col_w)
    fig, ax = plt.subplots(figsize=(total_w, total_h + 0.02), dpi=220)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    y = total_h - header_h
    x0 = 0.0
    for header, width in zip(headers, col_w):
        ax.add_patch(Rectangle((x0, y), width, header_h, facecolor=HEADER, edgecolor=LINE, linewidth=0.6))
        ax.text(x0 + width / 2, y + header_h / 2, header, ha="center", va="center",
                fontsize=7.4, color=BOLD, fontweight="bold")
        x0 += width
    for r, row in enumerate(rows):
        y -= body_h
        x0 = 0.0
        face = ZEBRA if r % 2 else "white"
        for cell, width in zip(row, col_w):
            ax.add_patch(Rectangle((x0, y), width, body_h, facecolor=face, edgecolor=LINE, linewidth=0.6))
            ax.text(x0 + width / 2, y + body_h / 2, cell, ha="center", va="center", fontsize=7.2, color=INK)
            x0 += width
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(path)


def plot_sweep(rows: list[dict], anchor: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=220)
    ax.scatter(
        [anchor["mean_total_usd"] * 1e4],
        [anchor["n_correct"]],
        s=120,
        facecolors="none",
        edgecolors=ONE,
        linewidths=1.6,
        zorder=3,
    )
    ax.annotate(
        f"One-shot\n{_fmt_ms(anchor['mean_total_latency_ms'])} ms",
        (anchor["mean_total_usd"] * 1e4, anchor["n_correct"]),
        textcoords="offset points",
        xytext=(8, -18),
        fontsize=8,
        color=ONE,
    )
    for row in rows:
        if row["mean_total_usd"] is None:
            continue
        x = row["mean_total_usd"] * 1e4
        y = row["n_correct"]
        color = LAM_COLOR.get(row.get("lambda"), INK)
        ax.scatter([x], [y], s=64, color=color, zorder=4)
        offset = {
            0: (10, 8),
            5: (10, -32),
            20: (10, 8),
            80: (10, -32),
        }.get(row.get("lambda"), (8, 6))
        ax.annotate(
            f"λ={row['lambda']}\n{int(y)}/300\n{_fmt_ms(row['mean_total_latency_ms'])} ms",
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=7.5,
            color=color,
        )
    ax.set_xlabel("Priced cost, mean dollars per question  ($ × 10⁻⁴)")
    ax.set_ylabel("Correct answers on the locked 300")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Retrieve-first cost sweep, seed 42", loc="left", color=BOLD, fontsize=11)
    fig.tight_layout()
    path = FIGS / "retrieve_first_lambda_sweep_em_usd.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    print(path)


def write_sweep_figures(rows: list[dict], anchor: dict) -> None:
    table_rows = [[
        "one-shot",
        "",
        f"{anchor['n_correct']}/{anchor['n']}",
        _fmt_usd(anchor["mean_total_usd"]),
        _fmt_ms(anchor["mean_total_latency_ms"]),
        "",
        "",
        "",
        _fmt_ci(anchor),
    ]]
    for row in rows:
        table_rows.append([
            str(row["lambda"]),
            "" if row["selected_epoch"] is None else str(row["selected_epoch"]),
            f"{row['n_correct']}/{row['n']}",
            _fmt_usd(row["mean_total_usd"]),
            _fmt_ms(row["mean_total_latency_ms"]),
            str(row["repairs"]),
            str(row["breaks"]),
            _fmt_p(row["sign_test_p"]),
            _fmt_ci(row),
        ])
    draw_table(
        FIGS / "retrieve_first_lambda_sweep_table.png",
        ["λ", "Epoch", "Correct", "$ ×10⁻⁴", "Latency ms", "Repairs", "Breaks", "Sign p", "EM 95%"],
        table_rows,
        [0.85, 0.7, 0.95, 0.85, 1.15, 0.85, 0.8, 0.75, 1.35],
    )
    plot_sweep(rows, anchor)


def run_sweep(require_all: bool) -> dict:
    missing = []
    rows = []
    for spec in SWEEP:
        if not spec["traj"].exists() or not spec["metrics"].exists():
            missing.append(spec["lambda"])
            print(f"λ={spec['lambda']} exam is not on disk yet.")
            continue
        row = one_row(
            f"retrieve-first λ={spec['lambda']} seed {spec['seed']}",
            spec["traj"],
            spec["metrics"],
            spec["curve"],
        )
        row["lambda"] = spec["lambda"]
        row["preset"] = spec["preset"]
        row["seed"] = spec["seed"]
        rows.append(row)
        print_row(row)
    if require_all and missing:
        raise SystemExit(
            "Missing λ exams: " + ", ".join(str(lam) for lam in missing) + ". Finish those cells first."
        )
    anchor = oneshot_row()
    print_row(anchor)
    payload = {
        "method": METHOD,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "baseline": str(BASELINE_TRAJ.relative_to(ROOT)),
        "one_shot": anchor,
        "rows": rows,
        "missing_lambda": missing,
    }
    if rows:
        write_sweep_figures(rows, anchor)
    out = METRICS / "retrieve_first_lambda_sweep.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison with one-shot, plus priced cost and measured latency.")
    parser.add_argument("--learned", type=Path, default=None, help="Learned trajectory jsonl.")
    parser.add_argument("--metrics", type=Path, default=None, help="Learned metrics json.")
    parser.add_argument("--curve", type=Path, default=None, help="Training curve, for the kept epoch.")
    parser.add_argument("--label", default="learned")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sweep", action="store_true", help="Compare every retrieve-first λ exam that is on disk.")
    parser.add_argument("--require-all", action="store_true", help="With --sweep, stop if any λ exam is missing.")
    args = parser.parse_args()
    if args.sweep:
        run_sweep(require_all=args.require_all)
        return
    if args.learned is None:
        raise SystemExit("Pass --learned, or pass --sweep.")
    learned = args.learned if args.learned.is_absolute() else ROOT / args.learned
    metrics = None
    if args.metrics is not None:
        metrics = args.metrics if args.metrics.is_absolute() else ROOT / args.metrics
    curve = None
    if args.curve is not None:
        curve = args.curve if args.curve.is_absolute() else ROOT / args.curve
    row = one_row(args.label, learned, metrics, curve)
    print_row(row)
    if args.out is not None:
        dest = args.out if args.out.is_absolute() else ROOT / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print("wrote", dest)


if __name__ == "__main__":
    main()
