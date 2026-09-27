"""Prefix splits for the v3 train and validation slices.

Validation is the next block of the same HF train split, after the train prefix.
It is not taken from the locked eval split.
"""

from __future__ import annotations

from typing import Any


def split_prefix(rows: list[dict[str, Any]], n_train: int, n_valid: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_train = int(n_train)
    n_valid = int(n_valid)
    need = n_train + n_valid
    if n_train < 1 or n_valid < 1:
        raise ValueError(f"need positive train and valid counts, got {n_train} and {n_valid}")
    if len(rows) < need:
        raise ValueError(f"need {need} rows for train+valid, got {len(rows)}")
    return list(rows[:n_train]), list(rows[n_train:need])


def tag_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Set the split field. Question ids stay as the loader wrote them."""
    tagged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["split"] = split
        tagged.append(item)
    return tagged
