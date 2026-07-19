from __future__ import annotations

import pytest

from experiments.m3.rewrite_score_automatic import (
    distribution_l2,
    hard_content_preserved,
    ngram_distribution,
    normalized_similarity,
    proportion_interval,
)


def test_similarity_and_hard_content_rules() -> None:
    row = {"protected_literals": ["2026", "Ada"]}
    assert hard_content_preserved(row, "Ada reported this in 2026.")
    assert not hard_content_preserved(row, "Ada reported this.")
    assert normalized_similarity("A  sentence.\n", "a sentence.") == pytest.approx(1.0)


def test_ngram_l2_is_zero_only_for_matching_distribution() -> None:
    first = ngram_distribution([[1, 2, 1], [2, 1]], 1)
    same = ngram_distribution([[2, 1], [1, 2, 1]], 1)
    other = ngram_distribution([[3, 3, 3]], 1)
    assert distribution_l2(first, same) == pytest.approx(0.0)
    assert distribution_l2(first, other) > 0


def test_bootstrap_proportion_is_bounded() -> None:
    result = proportion_interval([True] * 8 + [False] * 2)
    assert result["point"] == pytest.approx(0.8)
    assert 0 <= result["ci95_low"] <= result["ci95_high"] <= 1
