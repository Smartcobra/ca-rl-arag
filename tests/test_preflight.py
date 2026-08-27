#!/usr/bin/env python3
"""Unit tests for ranking data preflight (eval size + corpus size)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preflight import ranking_data_errors


def _cfg(**data):
    base = {
        "eval_hotpot": 150,
        "eval_nq": 150,
        "min_corpus_passages": 50000,
        "require_ranking_slice": True,
    }
    base.update(data)
    return {"data": base}


def _examples(n_h: int, n_n: int) -> list[dict]:
    return (
        [{"dataset": "hotpot_qa", "id": f"h{i}", "supporting_titles": [f"Hotpot {i}"]} for i in range(n_h)]
        + [{"dataset": "natural_questions", "id": f"n{i}", "supporting_titles": [f"NQ {i}"]} for i in range(n_n)]
    )


def test_tiny_corpus_fails_even_if_eval_is_300() -> None:
    errors = ranking_data_errors(_cfg(), _examples(150, 150), [{}] * 2276)
    assert any("2276 passages" in e and "50000" in e for e in errors)


def test_wrong_eval_mix_fails() -> None:
    errors = ranking_data_errors(_cfg(), _examples(300, 0), [{}] * 80000)
    assert any("300 examples" in e or "Hotpot count" in e for e in errors)


def test_ready_slice_and_corpus_passes() -> None:
    errors = ranking_data_errors(_cfg(), _examples(150, 150), [{}] * 80000)
    assert errors == []


def test_nq_anchors_fail_even_if_size_ok() -> None:
    corpus = [{"source": "nq_anchor", "title": "NQ anchor for x"}] * 80000
    errors = ranking_data_errors(_cfg(), _examples(150, 150), corpus)
    assert any("answer-anchor" in e for e in errors)


def test_trivia_single_hop_passes() -> None:
    examples = (
        [{"dataset": "hotpot_qa", "id": f"h{i}", "supporting_titles": [f"Hotpot {i}"]} for i in range(150)]
        + [{"dataset": "trivia_qa", "id": f"t{i}", "supporting_titles": [f"Trivia {i}"]} for i in range(150)]
    )
    errors = ranking_data_errors(_cfg(), examples, [{}] * 80000)
    assert errors == []


def test_single_topic_squad_eval_fails() -> None:
    """150 SQuAD questions over 16 articles is the old prefix-slice bug."""
    examples = (
        [{"dataset": "hotpot_qa", "id": f"h{i}", "supporting_titles": [f"Hotpot {i}"]} for i in range(150)]
        + [
            {"dataset": "squad", "id": f"s{i}", "supporting_titles": [f"Article {i % 16}"]}
            for i in range(150)
        ]
    )
    errors = ranking_data_errors(_cfg(), examples, [{}] * 80000)
    assert any("distinct gold articles" in e and "16" in e for e in errors)


def test_diverse_squad_eval_passes() -> None:
    examples = (
        [{"dataset": "hotpot_qa", "id": f"h{i}", "supporting_titles": [f"Hotpot {i}"]} for i in range(150)]
        + [{"dataset": "squad", "id": f"s{i}", "supporting_titles": [f"Squad {i}"]} for i in range(150)]
    )
    errors = ranking_data_errors(_cfg(), examples, [{}] * 80000)
    assert errors == []


def test_disabled_when_no_min_and_no_require() -> None:
    cfg = _cfg(min_corpus_passages=0, require_ranking_slice=False)
    errors = ranking_data_errors(cfg, _examples(10, 5), [{}] * 21)
    assert errors == []


def main() -> None:
    test_tiny_corpus_fails_even_if_eval_is_300()
    test_wrong_eval_mix_fails()
    test_ready_slice_and_corpus_passes()
    test_nq_anchors_fail_even_if_size_ok()
    test_trivia_single_hop_passes()
    test_single_topic_squad_eval_fails()
    test_diverse_squad_eval_passes()
    test_disabled_when_no_min_and_no_require()
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
