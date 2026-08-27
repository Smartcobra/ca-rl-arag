#!/usr/bin/env python3
"""NQ evidence must be real Wikipedia passages, not answer-anchor leakage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import NQ_ANCHOR_SOURCE, NQ_WIKI_NEG_SOURCE, NQ_WIKI_SOURCE
from src.data.preflight import ranking_data_errors
from src.data.wiki_passages import (
    count_leaky_anchors,
    examples_from_rows,
    looks_like_answer_anchor,
    merge_passages,
    squad_row_to_example,
    tevatron_row_to_example,
)
from src.retrieval.diagnostics import gold_passage_ids, gold_recall


def _dpr_row(
    query: str,
    answers: list[str],
    gold_text: str,
    gold_title: str = "Moon",
    gold_id: str = "100",
    neg_text: str = "Unrelated alpine geology and tourism.",
    neg_id: str = "200",
    query_id: str = "1",
) -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "answers": answers,
        "positive_passages": [{"docid": gold_id, "title": gold_title, "text": gold_text}],
        "negative_passages": [{"docid": neg_id, "title": "Alps", "text": neg_text}],
    }


def test_anchor_detector() -> None:
    q = "when did the last person land on the moon"
    ans = ["14 December 1972 UTC"]
    leak = f"{q} The answer is {ans[0]}. According to reference sources, the answer is {ans[0]}."
    wiki = "Apollo 17 (1972) was the last crewed Moon landing; Cernan and Schmitt left on 14 December 1972 UTC."
    assert looks_like_answer_anchor(q, ans, leak) is True
    assert looks_like_answer_anchor(q, ans, wiki) is False
    assert wiki.lower().startswith(q) is False


def test_tevatron_row_keeps_wikipedia_not_question() -> None:
    q = "when did the last person land on the moon"
    row = _dpr_row(
        q,
        ["14 December 1972 UTC"],
        "Apollo 17 was the final Apollo mission to the Moon and landed in December 1972.",
    )
    converted = tevatron_row_to_example(row, split="eval", dataset="natural_questions", index=0)
    assert converted is not None
    example, passages = converted
    golds = [p for p in passages if p["is_gold_support"]]
    negs = [p for p in passages if not p["is_gold_support"]]
    assert example["dataset"] == "natural_questions"
    assert example["question"] == q
    assert golds and golds[0]["source"] == NQ_WIKI_SOURCE
    assert q not in golds[0]["text"]
    assert "The answer is" not in golds[0]["text"]
    assert negs and negs[0]["source"] == NQ_WIKI_NEG_SOURCE
    assert example["local_passage_ids"] == [golds[0]["passage_id"]]


def test_leaky_tevatron_row_is_skipped() -> None:
    q = "what is the capital of france"
    ans = ["Paris"]
    leak = f"{q} The answer is {ans[0]}. According to reference sources, the answer is {ans[0]}."
    row = _dpr_row(q, ans, leak, gold_title="NQ anchor for nq_eval_0")
    assert tevatron_row_to_example(row, split="eval", dataset="natural_questions", index=0) is None


def test_examples_from_rows_fills_quota_and_skips_leaks() -> None:
    q = "who wrote pride and prejudice"
    leak_q = "what is the capital of france"
    rows = [
        _dpr_row(
            leak_q,
            ["Paris"],
            f"{leak_q} The answer is Paris. According to reference sources, the answer is Paris.",
            query_id="leak",
        ),
        _dpr_row(
            q,
            ["Jane Austen"],
            "Pride and Prejudice is an 1813 novel by Jane Austen.",
            gold_title="Pride and Prejudice",
            gold_id="9",
            query_id="ok",
        ),
    ]
    examples, passages, stats = examples_from_rows(
        rows, split="eval", n=1, kind="tevatron", dataset="natural_questions"
    )
    assert stats["n_skipped"] == 1
    assert len(examples) == 1
    assert examples[0]["answer"] == "Jane Austen"
    assert count_leaky_anchors(passages) == 0
    assert all(p["source"] != NQ_ANCHOR_SOURCE for p in passages)


def test_squad_fallback_uses_article_context() -> None:
    row = {
        "id": "s1",
        "title": "Super Bowl 50",
        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League for the 2015 season.",
        "question": "Which NFL season was Super Bowl 50 played for?",
        "answers": {"text": ["2015"], "answer_start": [110]},
    }
    converted = squad_row_to_example(row, split="eval", index=0)
    assert converted is not None
    example, passages = converted
    assert example["dataset"] == "squad"
    assert passages[0]["text"].startswith("Super Bowl 50")
    assert "The answer is" not in passages[0]["text"]


def _squad_prefix_then_other_articles() -> list[dict]:
    rows = []
    for i in range(20):
        rows.append(
            {
                "id": f"sb{i}",
                "title": "Super Bowl 50",
                "context": f"Super Bowl 50 paragraph {i} was an American football game in the 2015 season.",
                "question": f"Which NFL season was Super Bowl 50 paragraph {i}?",
                "answers": {"text": ["2015"], "answer_start": [0]},
            }
        )
    for i in range(20):
        rows.append(
            {
                "id": f"o{i}",
                "title": f"Article {i}",
                "context": f"Article {i} is a distinct Wikipedia page about an unrelated topic.",
                "question": f"What is article {i} about?",
                "answers": {"text": [f"topic {i}"], "answer_start": [0]},
            }
        )
    return rows


def test_squad_prefix_without_diversify_is_one_article() -> None:
    examples, _, stats = examples_from_rows(
        _squad_prefix_then_other_articles(),
        split="eval",
        n=10,
        kind="squad",
        dataset="squad",
        diversify_by_title=False,
    )
    titles = {ex["supporting_titles"][0] for ex in examples}
    assert titles == {"Super Bowl 50"}
    assert stats["diversify_by_title"] is False


def test_squad_fallback_samples_across_articles() -> None:
    """Fallback must not take the article-grouped prefix (16 golds / 150 Q)."""
    examples, _, stats = examples_from_rows(
        _squad_prefix_then_other_articles(),
        split="eval",
        n=10,
        kind="squad",
        dataset="squad",
        seed=42,
    )
    titles = {ex["supporting_titles"][0] for ex in examples}
    assert stats["diversify_by_title"] is True
    assert len(titles) == 10
    assert stats["n_gold_articles"] == 10


def test_gold_wins_over_negative_on_merge() -> None:
    gold = {
        "passage_id": "dpr_1",
        "title": "Paris",
        "text": "Paris is the capital of France.",
        "source": NQ_WIKI_SOURCE,
        "is_gold_support": True,
    }
    neg = {
        "passage_id": "dpr_1",
        "title": "Paris",
        "text": "Paris is the capital of France.",
        "source": NQ_WIKI_NEG_SOURCE,
        "is_gold_support": False,
    }
    merged = merge_passages([neg], [gold])
    assert len(merged) == 1
    assert merged[0]["is_gold_support"] is True
    assert merged[0]["source"] == NQ_WIKI_SOURCE


def test_gold_recall_on_real_wiki_passage() -> None:
    row = _dpr_row(
        "what is the capital of france",
        ["Paris"],
        "Paris is the capital and most populous city of France, on the river Seine.",
        gold_title="Paris",
        gold_id="12",
        neg_text="Berlin is the capital of Germany.",
        neg_id="99",
    )
    example, passages = tevatron_row_to_example(
        row, split="eval", dataset="natural_questions", index=0
    )
    by_id = {p["passage_id"]: p for p in passages}
    golds = gold_passage_ids(example, by_id)
    assert golds == example["local_passage_ids"]
    diag = gold_recall(passages, [example], ks=(1, 5))
    assert diag["n_examples"] == 1
    assert diag["recall@1"] == 1.0  # tiny 2-passage index; not the 80k ranking corpus


def test_preflight_rejects_answer_anchors() -> None:
    cfg = {
        "data": {
            "eval_hotpot": 150,
            "eval_nq": 150,
            "min_corpus_passages": 50000,
            "require_ranking_slice": True,
        }
    }
    examples = (
        [{"dataset": "hotpot_qa", "supporting_titles": [f"H{i}"]} for i in range(150)]
        + [{"dataset": "natural_questions", "supporting_titles": [f"N{i}"]} for i in range(150)]
    )
    corpus = (
        [{"source": "hotpot_qa", "title": "Paris"}] * 49900
        + [{"source": NQ_ANCHOR_SOURCE, "title": "NQ anchor for nq_eval_0", "text": "q The answer is a."}]
        * 100
    )
    errors = ranking_data_errors(cfg, examples, corpus)
    assert any("answer-anchor" in e for e in errors)


def test_preflight_accepts_trivia_fallback_mix() -> None:
    cfg = {
        "data": {
            "eval_hotpot": 150,
            "eval_nq": 150,
            "min_corpus_passages": 50000,
            "require_ranking_slice": True,
        }
    }
    examples = (
        [{"dataset": "hotpot_qa", "supporting_titles": [f"H{i}"]} for i in range(150)]
        + [{"dataset": "trivia_qa", "supporting_titles": [f"T{i}"]} for i in range(150)]
    )
    corpus = [{"source": NQ_WIKI_SOURCE, "title": "Paris"}] * 80000
    assert ranking_data_errors(cfg, examples, corpus) == []


def main() -> None:
    test_anchor_detector()
    test_tevatron_row_keeps_wikipedia_not_question()
    test_leaky_tevatron_row_is_skipped()
    test_examples_from_rows_fills_quota_and_skips_leaks()
    test_squad_fallback_uses_article_context()
    test_squad_prefix_without_diversify_is_one_article()
    test_squad_fallback_samples_across_articles()
    test_gold_wins_over_negative_on_merge()
    test_gold_recall_on_real_wiki_passage()
    test_preflight_rejects_answer_anchors()
    test_preflight_accepts_trivia_fallback_mix()
    print("NQ PASSAGES OK")


if __name__ == "__main__":
    main()
