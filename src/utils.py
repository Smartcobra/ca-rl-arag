"""Shared utilities: text normalization, tokenization, I/O, seeding."""

from __future__ import annotations

import json
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_answer(s: str) -> str:
    """SQuAD / Hotpot-style answer normalization."""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    articles = {"a", "an", "the"}
    tokens = [t for t in s.split() if t not in articles]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def exact_match(pred: str, gold: str | list[str]) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    pred_n = normalize_answer(pred)
    return float(any(pred_n == normalize_answer(g) for g in golds if g is not None))


def token_f1(pred: str, gold: str | list[str]) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    pred_toks = tokenize(normalize_answer(pred))
    if not pred_toks:
        return 0.0
    best = 0.0
    for g in golds:
        if g is None:
            continue
        gold_toks = tokenize(normalize_answer(str(g)))
        if not gold_toks:
            continue
        common = set(pred_toks) & set(gold_toks)
        if not common:
            continue
        # multiset-lite: count min occurrences
        num_same = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        best = max(best, f1)
    return float(best)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def timed() -> float:
    return time.perf_counter()


def ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def chunked(items: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def truncate(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
