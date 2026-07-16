import pytest

from harness.metrics.validity import (
    collapse_flags,
    language_integrity,
    non_target_script_char_rate,
    outline_fact_recall,
    repeated_sentence_start_rate,
    unsupported_claim_rate,
)


def test_outline_fact_recall_accepts_canonical_outline():
    outlines = [[{"section": "Facts", "supported_facts": ["Mars has two moons", "Venus is hot"]}]]
    assert outline_fact_recall(["Mars has two moons."], outlines) == 0.5
    assert outline_fact_recall(["anything"], [[]]) == 1.0


def test_unsupported_claim_rate_uses_fact_tables():
    texts = ["Mars has exactly two small moons. Bananas grow on Mars every winter."]
    facts = [[{"supported_facts": ["Mars has two small moons"]}]]
    assert unsupported_claim_rate(texts, facts) == 0.5
    assert unsupported_claim_rate(["Fine."], [[]]) == 0.0


def test_language_integrity_counts_non_latin_letters():
    assert non_target_script_char_rate("plain English 123") == 0.0
    assert non_target_script_char_rate("你好") == 1.0
    calibration = {"non_target_script_char_rate": {"low": 0.0, "high": 0.1}}
    assert language_integrity(["English"], calibration)
    assert not language_integrity(["你好"], calibration)


def test_collapse_ranges_include_lower_bound_and_repetition():
    texts = [
        "Same opening occurs here. Same begins again. Same appears a third time.",
        "Same starts once. Different starts next. Same is not consecutive.",
    ]
    repetition = repeated_sentence_start_rate(texts)
    assert repetition == 0.5
    unconstrained = {
        "self_bleu": {"low": 0.0, "high": 1.0},
        "repeated_sentence_start_rate": {"low": 0.0, "high": 1.0},
    }
    assert collapse_flags(texts, unconstrained)["pass"]
    constrained = {
        "self_bleu": {"low": 1.1, "high": None},
        "repeated_sentence_start_rate": {"low": 0.0, "high": 0.0},
    }
    result = collapse_flags(texts, constrained)
    assert not result["self_bleu_in_range"]
    assert not result["repetition_in_range"]
    assert not result["pass"]


def test_repeated_start_requires_three_consecutive_identical_first_words():
    assert repeated_sentence_start_rate(["Again one. Again two."]) == 0.0
    assert repeated_sentence_start_rate(["Again one. Other. Again two. Again three."]) == 0.0
    assert repeated_sentence_start_rate(["Again one. Again two! Again three?"]) == 1.0
    assert repeated_sentence_start_rate(
        ["Again one. Again two. Again three.", "No run. Another sentence. Last sentence."]
    ) == 0.5


def test_null_calibration_bounds_fail_closed():
    calibration = {
        "self_bleu": {"low": None, "high": None},
        "repeated_sentence_start_rate": {"low": None, "high": None},
        "non_target_script_char_rate": {"low": 0.0, "high": None},
    }
    assert not collapse_flags(["One text.", "Another text."], calibration)["pass"]
    assert not language_integrity(["English"], calibration)


def test_v2_repetition_bound_is_upper_only_without_changing_self_bleu_bounds():
    zero_repetition = ["One sentence. Another begins. Finally done."] * 2
    calibration = {
        "self_bleu": {"low": 0.0, "high": 1.0},
        "repeated_sentence_start_rate": {
            "low": 0.005,
            "high": 0.1575,
            "bound_mode": "upper_only",
        },
    }
    result = collapse_flags(zero_repetition, calibration)
    assert result["repetition_rate"] == 0.0
    assert result["repetition_in_range"] is True
    assert result["repetition_bound_mode"] == "upper_only"

    collapsed = ["Again one. Again two. Again three."] * 2
    assert collapse_flags(collapsed, calibration)["repetition_in_range"] is False


def test_validity_length_mismatches_raise():
    with pytest.raises(ValueError, match="equal"):
        outline_fact_recall(["x"], [])
    with pytest.raises(ValueError, match="equal"):
        unsupported_claim_rate(["x"], [])
