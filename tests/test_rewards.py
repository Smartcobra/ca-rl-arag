#!/usr/bin/env python3
"""Unit tests for calibration_score. No GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rewards import calibration_score


STRONG = [{"title": "Paris", "text": "Paris is the capital of France.", "score": 12.0}]
WEAK = [{"title": "Noise", "text": "unrelated", "score": 1.0}]
CONTRA = {"label": "contradiction", "support": 0.0, "contradiction": 1.0}


def test_abstain_empty_evidence_is_justified() -> None:
    assert calibration_score("ABSTAIN", "Paris", True, []) == 0.6


def test_abstain_strong_evidence_is_lazy() -> None:
    # The cheat path: refuse even though retrieval was usable.
    assert calibration_score("ABSTAIN", "Paris", True, STRONG) == -0.2
    assert calibration_score("ABSTAIN", "Paris", True, STRONG, {"label": "support"}) == -0.2


def test_abstain_weak_mean_score_is_justified() -> None:
    assert calibration_score("ABSTAIN", "Paris", True, WEAK) == 0.6


def test_abstain_verify_contradiction_is_justified() -> None:
    assert calibration_score("ABSTAIN", "Paris", True, STRONG, CONTRA) == 0.6


def test_non_abstain_paths_unchanged() -> None:
    assert calibration_score("Paris", "Paris", False, STRONG) == 0.3
    assert calibration_score("Lyon", "Paris", False, STRONG) == -0.4
    assert calibration_score("Lyon", "Paris", False, []) == -0.2


def main() -> None:
    test_abstain_empty_evidence_is_justified()
    test_abstain_strong_evidence_is_lazy()
    test_abstain_weak_mean_score_is_justified()
    test_abstain_verify_contradiction_is_justified()
    test_non_abstain_paths_unchanged()
    print("REWARDS CALIBRATION OK")


if __name__ == "__main__":
    main()
