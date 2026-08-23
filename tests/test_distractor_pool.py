#!/usr/bin/env python3
"""Unit tests for unused-Hotpot distractor pooling. No GPU / no HF download."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import (
    HOTPOT_DISTRACTOR_SOURCE,
    HOTPOT_SLICE_SOURCE,
    add_hotpot_distractor_pool,
    hotpot_to_examples,
    passages_from_hotpot_row,
)
from src.retrieval.diagnostics import gold_passage_ids, gold_recall


def _row(qid: str, paragraphs: list[tuple[str, str]], gold_titles: list[str]) -> dict:
    return {
        "id": qid,
        "question": f"Question about {gold_titles[0]}?",
        "answer": gold_titles[0],
        "type": "bridge",
        "level": "easy",
        "context": {
            "title": [t for t, _ in paragraphs],
            "sentences": [[text] for _, text in paragraphs],
        },
        "supporting_facts": {"title": gold_titles, "sent_id": [0] * len(gold_titles)},
    }


def test_slice_keeps_gold_flags() -> None:
    row = _row(
        "1",
        [("Paris", "Paris is the capital of France."), ("Tourism", "Tourists visit museums.")],
        ["Paris"],
    )
    examples, passages = hotpot_to_examples([row], "eval")
    assert len(examples) == 1
    assert examples[0]["supporting_titles"] == ["Paris"]
    assert examples[0]["local_passage_ids"] == [p["passage_id"] for p in passages]
    by_title = {p["title"]: p for p in passages}
    assert by_title["Paris"]["is_gold_support"] is True
    assert by_title["Paris"]["source"] == HOTPOT_SLICE_SOURCE
    assert by_title["Tourism"]["is_gold_support"] is False


def test_unused_rows_are_never_gold() -> None:
    unused = _row(
        "unused",
        [("Berlin", "Berlin is the capital of Germany."), ("Noise", "Unrelated paragraph.")],
        ["Berlin"],
    )
    passages = passages_from_hotpot_row(unused, as_distractor=True)
    assert passages
    assert all(p["source"] == HOTPOT_DISTRACTOR_SOURCE for p in passages)
    assert all(p["is_gold_support"] is False for p in passages)


def test_pool_preserves_golds_and_stops_at_target() -> None:
    slice_row = _row(
        "keep",
        [("Paris", "Paris is the capital of France."), ("Tourism", "Tourists visit museums.")],
        ["Paris"],
    )
    examples, slice_passages = hotpot_to_examples([slice_row], "eval")
    unused = [
        _row("u1", [("Berlin", "Berlin is the capital of Germany.")], ["Berlin"]),
        _row("u2", [("Rome", "Rome is the capital of Italy.")], ["Rome"]),
        _row("u3", [("Madrid", "Madrid is the capital of Spain.")], ["Madrid"]),
    ]
    target = len(slice_passages) + 2
    corpus, stats = add_hotpot_distractor_pool(slice_passages, unused, target)
    assert stats["hit_target"] is True
    assert stats["n_distractors_added"] == 2
    assert stats["n_slice_passages"] == len(slice_passages)
    assert len(corpus) == target
    golds = [p for p in corpus if p["title"] == "Paris"]
    assert len(golds) == 1
    assert golds[0]["is_gold_support"] is True
    assert golds[0]["source"] == HOTPOT_SLICE_SOURCE
    assert examples[0]["local_passage_ids"] == [p["passage_id"] for p in slice_passages]
    added_ids = {p["passage_id"] for p in corpus} - {p["passage_id"] for p in slice_passages}
    assert added_ids.isdisjoint(set(examples[0]["local_passage_ids"]))


def test_dedup_keeps_slice_gold() -> None:
    slice_row = _row("keep", [("Paris", "Paris is the capital of France.")], ["Paris"])
    _, slice_passages = hotpot_to_examples([slice_row], "eval")
    # Same title+text as the gold, plus a new distractor.
    unused = [
        _row(
            "dup",
            [
                ("Paris", "Paris is the capital of France."),
                ("Noise", "Completely different unused paragraph."),
            ],
            ["Paris"],
        )
    ]
    corpus, stats = add_hotpot_distractor_pool(slice_passages, unused, target_size=100)
    assert stats["n_distractor_dups_skipped"] >= 1
    paris = [p for p in corpus if p["title"] == "Paris"]
    assert len(paris) == 1
    assert paris[0]["is_gold_support"] is True
    assert paris[0]["source"] == HOTPOT_SLICE_SOURCE
    noise = [p for p in corpus if p["title"] == "Noise"]
    assert len(noise) == 1
    assert noise[0]["is_gold_support"] is False
    assert noise[0]["source"] == HOTPOT_DISTRACTOR_SOURCE


def test_target_zero_or_already_met_is_noop() -> None:
    row = _row("1", [("Paris", "Paris is the capital of France.")], ["Paris"])
    _, slice_passages = hotpot_to_examples([row], "eval")
    unused = [_row("u", [("Berlin", "Berlin is the capital of Germany.")], ["Berlin"])]
    corpus, stats = add_hotpot_distractor_pool(slice_passages, unused, 0)
    assert corpus == slice_passages
    assert stats["n_distractors_added"] == 0
    corpus2, stats2 = add_hotpot_distractor_pool(slice_passages, unused, len(slice_passages))
    assert corpus2 == slice_passages
    assert stats2["n_distractors_added"] == 0


def test_gold_recall_finds_obvious_gold() -> None:
    row = _row("1", [("Paris", "Paris is the capital of France.")], ["Paris"])
    examples, passages = hotpot_to_examples([row], "eval")
    unused = [_row("u", [("Tourism notes", "Many cities have parks and museums.")], ["Tourism notes"])]
    corpus, _ = add_hotpot_distractor_pool(passages, unused, target_size=100)
    by_id = {p["passage_id"]: p for p in corpus}
    golds = gold_passage_ids(examples[0], by_id)
    assert golds == [passages[0]["passage_id"]]
    diag = gold_recall(corpus, examples, ks=(1, 5))
    assert diag["n_examples"] == 1
    assert diag["recall@1"] == 1.0
    assert diag["recall@5"] == 1.0


def main() -> None:
    test_slice_keeps_gold_flags()
    test_unused_rows_are_never_gold()
    test_pool_preserves_golds_and_stops_at_target()
    test_dedup_keeps_slice_gold()
    test_target_zero_or_already_met_is_noop()
    test_gold_recall_finds_obvious_gold()
    print("DISTRACTOR POOL OK")


if __name__ == "__main__":
    main()
