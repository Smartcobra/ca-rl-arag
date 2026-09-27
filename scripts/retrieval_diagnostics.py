#!/usr/bin/env python3
"""BM25 gold recall@k on the current processed corpus. No GPU.

Usage:
  python scripts/retrieval_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.retrieval.diagnostics import gold_recall_by_dataset
from src.utils import read_jsonl


def main() -> None:
    cfg = load_config()
    processed = resolve_path(cfg, cfg["data"]["processed_dir"])
    passages = read_jsonl(processed / "corpus.jsonl")
    examples = read_jsonl(processed / "eval_slice.jsonl")
    diag = gold_recall_by_dataset(passages, examples)
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
