"""Locked eval ids. Training code compares against this list and never opens the 300."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCKED_EVAL_IDS_PATH = ROOT / "tests" / "fixtures" / "locked_eval_ids.json"


def load_locked_eval_ids(path: str | Path | None = None) -> list[str]:
    path = Path(path) if path is not None else LOCKED_EVAL_IDS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing locked eval ids at {path}. "
            "Snapshot data/processed/eval_slice.jsonl before rebuilding train or valid."
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    ids = blob.get("ids") if isinstance(blob, dict) else blob
    return [str(item) for item in ids]


def assert_ids_unchanged(got: list[str], locked: list[str], label: str) -> None:
    got_ids = [str(item) for item in got]
    locked_ids = [str(item) for item in locked]
    if got_ids == locked_ids:
        return
    raise ValueError(
        f"{label} ids changed: got {len(got_ids)} rows, locked {len(locked_ids)}. "
        "Do not rewrite eval_slice.jsonl."
    )


def assert_disjoint(*named: tuple[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for name, ids in named:
        for raw in ids:
            item = str(raw)
            owner = seen.get(item)
            if owner is not None:
                raise ValueError(f"id {item} is in both {owner} and {name}")
            seen[item] = name
