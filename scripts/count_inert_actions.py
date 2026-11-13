#!/usr/bin/env python3
"""Count actions that cannot change the answer written at stop. No GPU.

Stop generates the answer from the question and the passages then in hand
(src/agentic_rag.py). Three kinds of earlier action cannot change that answer:

- verify, including a draft generated inside the verify step
- a retrieve whose query was already searched
- a rewrite with no later retrieve

Rerank stays off this list: it reorders the passages the generator reads.
Stop stays off this list: it is the answer.

A step's dollars are the drop in remaining_usd, so the draft inside verify is
charged to verify. The last step is closed with the starting budget minus
total_usd.

  python scripts/count_inert_actions.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "results" / "trajectories"
OUT = ROOT / "results" / "metrics" / "inert_actions.json"

# The retrieve-first rows in paper_v4/paper.tex. Seed 43 is retrieve then stop,
# so it is the one-shot bill on this index.
ROWS = (
    {
        "name": "lambda20_s42",
        "lambda": 20,
        "seed": 42,
        "traj": "learned_frontier_lambda20_rfs42.jsonl",
    },
    {
        "name": "lambda20_s43",
        "lambda": 20,
        "seed": 43,
        "traj": "learned_frontier_lambda20_rfs43.jsonl",
    },
    {
        "name": "lambda20_s44",
        "lambda": 20,
        "seed": 44,
        "traj": "learned_frontier_lambda20_rfs44.jsonl",
    },
    {
        "name": "lambda0_s42",
        "lambda": 0,
        "seed": 42,
        "traj": "learned_frontier_lambda0_rf0s42.jsonl",
    },
    {
        "name": "lambda5_s42",
        "lambda": 5,
        "seed": 42,
        "traj": "learned_frontier_lambda5_rf5s42.jsonl",
    },
    {
        "name": "lambda80_s42",
        "lambda": 80,
        "seed": 42,
        "traj": "learned_frontier_lambda80_rf80s42.jsonl",
    },
    {
        "name": "lambda20_s45_masked",
        "lambda": 20,
        "seed": 45,
        "traj": "learned_frontier_lambda20_rfinert45.jsonl",
    },
)

ONESHOT = "learned_frontier_lambda20_rfs43.jsonl"
TOL = 1e-9


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def step_usd(row: dict) -> list[float]:
    """Price-card dollars of each trajectory step, in order."""
    steps = row["trajectory"]
    if not steps:
        raise ValueError(f"{row.get('id')}: empty trajectory")
    budget = float(steps[0]["observation_before"]["remaining_usd"])
    total = float(row["total_usd"])
    costs: list[float] = []
    for i, step in enumerate(steps):
        before = float(step["observation_before"]["remaining_usd"])
        if i + 1 < len(steps):
            after = float(steps[i + 1]["observation_before"]["remaining_usd"])
        else:
            after = budget - total
        costs.append(before - after)
    if abs(sum(costs) - total) > TOL:
        raise ValueError(
            f"{row.get('id')}: step dollars {sum(costs)} != total_usd {total}"
        )
    return costs


def idle_flags(row: dict) -> list[bool]:
    """True when that step cannot change the answer written at stop."""
    steps = row["trajectory"]
    actions = [step["action"] for step in steps]
    followed_by_retrieve = [False] * len(actions)
    seen_retrieve = False
    for i in range(len(actions) - 1, -1, -1):
        followed_by_retrieve[i] = seen_retrieve
        if actions[i] == "retrieve":
            seen_retrieve = True

    query = row["question"]
    searched: set[str] = set()
    flags: list[bool] = []
    for i, step in enumerate(steps):
        action = step["action"]
        if action == "verify":
            flags.append(True)
        elif action == "retrieve":
            flags.append(query in searched)
            searched.add(query)
        elif action == "rewrite":
            flags.append(not followed_by_retrieve[i])
            if "rewritten_query" not in step:
                raise ValueError(f"{row.get('id')}: rewrite without rewritten_query")
            query = step["rewritten_query"]
        else:
            flags.append(False)
    return flags


def kept_usd(row: dict) -> float:
    flags = idle_flags(row)
    return sum(usd for usd, idle in zip(step_usd(row), flags) if not idle)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no trajectories")
    idle_actions = 0
    tool_actions = 0
    idle_usd = 0.0
    total_usd = 0.0
    for row in rows:
        flags = idle_flags(row)
        costs = step_usd(row)
        actions = [step["action"] for step in row["trajectory"]]
        idle_actions += sum(flags)
        tool_actions += sum(action != "stop" for action in actions)
        idle_usd += sum(usd for usd, idle in zip(costs, flags) if idle)
        total_usd += float(row["total_usd"])
    n = len(rows)
    mean_idle = idle_usd / n
    mean_total = total_usd / n
    return {
        "n": n,
        "mean_idle_actions": idle_actions / n,
        "mean_tool_actions": tool_actions / n,
        "mean_idle_usd": mean_idle,
        "mean_total_usd": mean_total,
        "idle_share": (idle_usd / total_usd) if total_usd else 0.0,
        "idle_1e4": round(mean_idle * 1e4, 2),
    }


def dollar_term(learned: list[dict], oneshot: dict[str, dict], lam: float) -> float:
    """λ times the mean extra price-card dollars against one-shot on the same ids."""
    extra = 0.0
    for row in learned:
        extra += float(row["total_usd"]) - float(oneshot[row["id"]]["total_usd"])
    return lam * (extra / len(learned))


def main() -> None:
    oneshot_rows = load_jsonl(TRAJ / ONESHOT)
    oneshot = {row["id"]: row for row in oneshot_rows}
    report = []
    for spec in ROWS:
        rows = load_jsonl(TRAJ / spec["traj"])
        summary = summarize(rows)
        summary.update({"name": spec["name"], "lambda": spec["lambda"], "seed": spec["seed"]})
        if spec["lambda"] == 20 and spec["seed"] == 44:
            summary["dollar_term_vs_oneshot"] = dollar_term(rows, oneshot, 20)
        if spec["name"] == "lambda5_s42":
            mismatches = [
                row["id"]
                for row in rows
                if abs(kept_usd(row) - float(oneshot[row["id"]]["total_usd"])) > TOL
            ]
            summary["kept_matches_oneshot"] = len(rows) - len(mismatches)
            summary["kept_mismatches"] = len(mismatches)
        report.append(summary)
        share = 100.0 * summary["idle_share"]
        print(
            f"{spec['name']}: idle {summary['mean_idle_actions']:.2f} of "
            f"{summary['mean_tool_actions']:.2f} tool calls, "
            f"{summary['idle_1e4']:.2f} x 1e-4 dollars, {share:.1f}% of the bill"
        )
    lam5 = next(row for row in report if row["name"] == "lambda5_s42")
    lam44 = next(row for row in report if row["name"] == "lambda20_s44")
    print(
        f"lambda5 kept dollars match one-shot on "
        f"{lam5['kept_matches_oneshot']}/{lam5['n']}"
    )
    print(f"lambda20 seed44 extra dollars times 20: {lam44['dollar_term_vs_oneshot']:.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
