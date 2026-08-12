"""Evaluation runners and metric aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from .agentic_rag import AgenticRAG
from .metrics import aggregate_metrics
from .rag_baseline import RAGBaseline
from .utils import append_jsonl, ensure_dir, write_jsonl


def evaluate_baseline(baseline: RAGBaseline, examples: list[dict[str, Any]], out_path: str | Path | None = None) -> dict[str, Any]:
    rows = []
    for ex in tqdm(examples, desc="baseline"):
        row = baseline.run(ex)
        rows.append(row)
        if out_path:
            append_jsonl(out_path, _serializable(row))
    summary = aggregate_metrics(rows)
    return {"summary": summary, "rows": rows}


def evaluate_agent(
    agent: AgenticRAG,
    examples: list[dict[str, Any]],
    policy_fn: Callable,
    policy_name: str,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = []
    for ex in tqdm(examples, desc=f"agent:{policy_name}"):
        row = agent.run_with_policy(ex, policy_fn, policy_name=policy_name)
        rows.append(row)
        if out_path:
            append_jsonl(out_path, _serializable(row))
    summary = aggregate_metrics(rows)
    return {"summary": summary, "rows": rows}


def save_metrics(path: str | Path, summary: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {"summary": summary, "meta": meta or {}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _serializable(row: dict[str, Any]) -> dict[str, Any]:
    # Drop bulky passage text optionally keep ids
    out = dict(row)
    if "retrieved" in out:
        out["retrieved"] = [
            {"passage_id": d.get("passage_id"), "title": d.get("title"), "score": d.get("score"), "rank": d.get("rank")}
            for d in out["retrieved"]
        ]
    return out
