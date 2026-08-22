#!/usr/bin/env python3
"""Unit tests for exclusive answer / ABSTAIN parsing. No GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.llm import (
    ExtractiveGenerator,
    is_yes_no_question,
    parse_answer_or_abstain,
    parse_yes_no,
    should_allow_abstain,
)


def test_pure_abstain() -> None:
    assert parse_answer_or_abstain("ABSTAIN") == ("ABSTAIN", "abstain")
    assert parse_answer_or_abstain("abstain") == ("ABSTAIN", "abstain")
    assert parse_answer_or_abstain("ABSTAIN.") == ("ABSTAIN", "abstain")


def test_trailing_abstain_stripped_as_answer() -> None:
    pred, mode = parse_answer_or_abstain(
        "Secretary of State for Constitutional Affairs ABSTAIN"
    )
    assert mode == "answer"
    assert pred == "Secretary of State for Constitutional Affairs"
    assert "ABSTAIN" not in pred.upper()

    pred, mode = parse_answer_or_abstain("Lee Hazlewood ABSTAIN")
    assert (pred, mode) == ("Lee Hazlewood", "answer")

    pred, mode = parse_answer_or_abstain("Paris ABSTAIN.")
    assert (pred, mode) == ("Paris", "answer")

    pred, mode = parse_answer_or_abstain("Paris\nABSTAIN")
    assert (pred, mode) == ("Paris", "answer")


def test_leading_abstain_is_refuse() -> None:
    assert parse_answer_or_abstain("ABSTAIN: not enough evidence") == ("ABSTAIN", "abstain")
    assert parse_answer_or_abstain("ABSTAIN not enough evidence") == ("ABSTAIN", "abstain")
    assert parse_answer_or_abstain("ABSTAIN\nParis") == ("ABSTAIN", "abstain")


def test_answer_prefix_stripped() -> None:
    assert parse_answer_or_abstain("Answer: Paris") == ("Paris", "answer")
    assert parse_answer_or_abstain("Final answer: Paris") == ("Paris", "answer")
    assert parse_answer_or_abstain("A: Paris") == ("Paris", "answer")
    assert parse_answer_or_abstain("Answer: ABSTAIN") == ("ABSTAIN", "abstain")


def test_empty_respects_allow_abstain() -> None:
    assert parse_answer_or_abstain("") == ("ABSTAIN", "abstain")
    assert parse_answer_or_abstain("   ") == ("ABSTAIN", "abstain")
    assert parse_answer_or_abstain("", allow_abstain=False) == ("", "answer")
    assert parse_answer_or_abstain("ABSTAIN", allow_abstain=False) == ("", "answer")


def test_never_returns_mixed_string() -> None:
    cases = [
        "ABSTAIN",
        "Secretary of State for Constitutional Affairs ABSTAIN",
        "ABSTAIN: not enough evidence",
        "Answer: Paris",
        "",
        "Lee Hazlewood ABSTAIN",
    ]
    for raw in cases:
        pred, mode = parse_answer_or_abstain(raw)
        if mode == "abstain":
            assert pred == "ABSTAIN"
        else:
            assert "ABSTAIN" not in pred.upper()


def test_yes_no_questions() -> None:
    yn = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "Do the drinks Gibson and Zurracapote both contain gin?",
        "Are both Dictyosperma, and Huernia described as a genus?",
    ]
    not_yn = [
        "What science fantasy young adult series, told in first person, has companion books?",
        "Who was known by his stage name Aladin?",
        "In what city did the Prince of tenors star in a film?",
        "How are Local H and For Against related?",
    ]
    for q in yn:
        assert is_yes_no_question(q), q
        assert should_allow_abstain(q, allow_abstain=True, force_yes_no=True) is False
    for q in not_yn:
        assert not is_yes_no_question(q), q
        assert should_allow_abstain(q, allow_abstain=True, force_yes_no=True) is True
    assert should_allow_abstain(not_yn[0], allow_abstain=False, force_yes_no=True) is False


def test_parse_yes_no() -> None:
    assert parse_yes_no("yes") == ("yes", "answer")
    assert parse_yes_no("Yes.") == ("yes", "answer")
    assert parse_yes_no("Yes, they were both American.") == ("yes", "answer")
    assert parse_yes_no("no") == ("no", "answer")
    assert parse_yes_no("No, they are not in the same neighborhood.") == ("no", "answer")
    assert parse_yes_no("Answer: no") == ("no", "answer")
    assert parse_yes_no("Norway") == ("", "answer")
    assert parse_yes_no("ABSTAIN") == ("", "answer")
    assert parse_yes_no("ABSTAIN", allow_abstain=True) == ("ABSTAIN", "abstain")


def test_extractive_yes_no_does_not_abstain() -> None:
    gen = ExtractiveGenerator(force_yes_no=True, allow_abstain=True)
    pred, mode, _ = gen.generate(
        "Are Ferocactus and Silene both types of plant?",
        [{"title": "Ferocactus", "text": "Ferocactus is a genus of cactus.", "score": 1.0}],
    )
    assert mode == "answer"
    assert pred.upper() != "ABSTAIN"


def main() -> None:
    test_pure_abstain()
    test_trailing_abstain_stripped_as_answer()
    test_leading_abstain_is_refuse()
    test_answer_prefix_stripped()
    test_empty_respects_allow_abstain()
    test_never_returns_mixed_string()
    test_yes_no_questions()
    test_parse_yes_no()
    test_extractive_yes_no_does_not_abstain()
    print("ABSTAIN PARSE OK")


if __name__ == "__main__":
    main()
