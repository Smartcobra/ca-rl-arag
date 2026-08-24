"""Evaluation metrics for quality and efficiency."""

from __future__ import annotations

from typing import Any, Iterable

from .utils import exact_match, token_f1

DATASET_DISPLAY_ORDER = ("hotpot_qa", "natural_questions", "trivia_qa", "squad")
DATASET_LABELS = {
    "hotpot_qa": "HotpotQA",
    "natural_questions": "Natural Questions",
    "trivia_qa": "TriviaQA",
    "squad": "SQuAD",
}

_MEAN_KEYS = [
    "em",
    "f1",
    "q_ans",
    "q_ground",
    "q_cal",
    "p_hall",
    "reward",
    "total_usd",
    "total_latency_ms",
    "total_tokens",
    "n_steps",
    "n_retrieve",
    "n_rewrite",
    "n_rerank",
    "n_verify",
    "usd_per_correct",
]


def compute_answer_metrics(pred: str, gold: str | list[str], abstained: bool = False) -> dict[str, float]:
    if abstained:
        # Abstention: no EM/F1 credit; calibration handled in reward.
        return {"em": 0.0, "f1": 0.0, "abstained": 1.0}
    return {
        "em": exact_match(pred, gold),
        "f1": token_f1(pred, gold),
        "abstained": 0.0,
    }


def ordered_dataset_names(names: Iterable[str]) -> list[str]:
    present = set(names)
    preferred = [d for d in DATASET_DISPLAY_ORDER if d in present]
    extra = sorted(n for n in present if n not in DATASET_DISPLAY_ORDER)
    return preferred + extra


def counts_by_dataset(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        ds = str(r.get("dataset") or "unknown")
        counts[ds] = counts.get(ds, 0) + 1
    return {k: counts[k] for k in ordered_dataset_names(counts)}


def _row_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Means and counts for one group of trajectory rows (overall or one dataset)."""
    if not rows:
        return {}
    out: dict[str, float] = {}
    for k in _MEAN_KEYS:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[f"mean_{k}"] = sum(vals) / len(vals)
    n = len(rows)
    out["n_examples"] = float(n)
    correct = sum(1 for r in rows if float(r.get("em", 0) or 0) >= 1.0)
    out["n_correct"] = float(correct)
    n_abstained = sum(1 for r in rows if r.get("abstained"))
    out["n_abstained"] = float(n_abstained)
    out["abstain_rate"] = n_abstained / n
    total_usd = sum(float(r.get("total_usd", 0.0) or 0.0) for r in rows)
    out["total_usd_all"] = total_usd
    out["usd_per_correct"] = (total_usd / correct) if correct else float("inf")
    return out


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall means plus the same aggregator grouped by ``row["dataset"]``.

    Overall is mix-weighted (Hotpot + NQ in one pool). Read ``by_dataset``
    before treating overall EM/F1/reward as a policy result.
    """
    if not rows:
        return {}
    out: dict[str, Any] = dict(_row_metrics(rows))
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "unknown")
        groups.setdefault(ds, []).append(r)
    out["by_dataset"] = {ds: _row_metrics(groups[ds]) for ds in ordered_dataset_names(groups)}
    return out


def format_metric_line(label: str, stats: dict[str, Any]) -> str:
    n = int(stats.get("n_examples") or 0)
    n_correct = int(stats.get("n_correct") or 0)
    n_abs = int(stats.get("n_abstained") or 0)
    em = float(stats.get("mean_em") or 0.0)
    f1 = float(stats.get("mean_f1") or 0.0)
    reward = float(stats.get("mean_reward") or 0.0)
    abstain = float(stats.get("abstain_rate") or 0.0)
    return (
        f"{label}: EM={em:.3f} F1={f1:.3f} reward={reward:.3f} "
        f"abstain={abstain:.3f} n_correct={n_correct}/{n} n_abstained={n_abs}"
    )


def format_eval_summary(policy: str, summary: dict[str, Any]) -> str:
    """Overall line, then one line per dataset (Hotpot, then NQ)."""
    lines = [format_metric_line(f"{policy} overall", summary)]
    by_ds = summary.get("by_dataset") or {}
    for ds in ordered_dataset_names(by_ds):
        label = DATASET_LABELS.get(ds, ds)
        lines.append("  " + format_metric_line(label, by_ds[ds]))
    return "\n".join(lines)


def compact_ablation_row(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mean_em",
        "mean_f1",
        "mean_reward",
        "mean_total_usd",
        "mean_n_retrieve",
        "usd_per_correct",
        "n_examples",
        "n_correct",
        "n_abstained",
        "abstain_rate",
    )
    row = {k: summary.get(k) for k in keys}
    by_ds = summary.get("by_dataset") or {}
    row["by_dataset"] = {ds: {k: by_ds[ds].get(k) for k in keys} for ds in ordered_dataset_names(by_ds)}
    return row
