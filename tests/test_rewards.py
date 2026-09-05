#!/usr/bin/env python3
"""Unit tests for calibration_score lazy-abstain gates. No GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rewards import calibration_score

PRED = "ABSTAIN"
GOLD = "Paris"
STRONG = [{"title": "Paris", "text": "Paris is the capital of France.", "score": 8.0}]
WEAK = [{"title": "Paris", "text": "Paris is the capital of France.", "score": 1.0}]


def test_abstain_empty_or_weak_evidence_is_justified() -> None:
    empty = calibration_score(PRED, GOLD, abstained=True, evidence=[])
    weak = calibration_score(PRED, GOLD, abstained=True, evidence=WEAK)
    assert empty == 0.6
    assert weak == 0.6


def test_abstain_strong_evidence_without_contradiction_is_lazy() -> None:
    no_verify = calibration_score(PRED, GOLD, abstained=True, evidence=STRONG)
    support = calibration_score(
        PRED, GOLD, abstained=True, evidence=STRONG, verify_out={"label": "support"}
    )
    assert no_verify == -0.2
    assert support == -0.2


def test_abstain_contradiction_is_justified() -> None:
    score = calibration_score(
        PRED,
        GOLD,
        abstained=True,
        evidence=STRONG,
        verify_out={"label": "contradiction"},
    )
    assert score == 0.6


def main() -> None:
    test_abstain_empty_or_weak_evidence_is_justified()
    test_abstain_strong_evidence_without_contradiction_is_lazy()
    test_abstain_contradiction_is_justified()
    print("REWARDS CALIBRATION OK")


if __name__ == "__main__":
    main()
