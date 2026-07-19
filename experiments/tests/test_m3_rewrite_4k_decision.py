from __future__ import annotations

from copy import deepcopy

import pytest

from data.m3_rewrite_judge import MODELS
from experiments.m3.rewrite_4k_decision import M3RewriteDecisionError, decide
from experiments.m3.rewrite_embedding_score import FAMILIES


def fixtures():
    automatic = {
        "artifact_schema": "humanwrite.m3.rewrite_automatic_score.v1",
        "records": 256,
        "arms": {
            "SFT14": {
                "unexpected_non_latin": {"point": 0.01},
            },
            "HUMANWRITE14": {
                "meaningful_edit_ai_inputs": {"point": 0.75},
                "hard_content_preservation": {"point": 1.0},
                "unexpected_non_latin": {"point": 0.02},
                "replacement_character_count": 0,
            },
        },
        "lexical": {
            "token_1gram_l2": {
                "treatment_minus_sft": -0.001,
                "human_split": {"sd": 0.002},
            },
            "token_2gram_l2": {
                "treatment_minus_sft": 0.001,
                "human_split": {"sd": 0.002},
            },
            "token_3gram_l2": {
                "treatment_minus_sft": 0.0,
                "human_split": {"sd": 0.002},
            },
        },
    }
    judge = {
        "artifact_schema": "humanwrite.m3.rewrite_judge_summary.v2",
        "comparisons": 2048,
        "models": {
            model: {
                "pairwise": {
                    "human_style": {"preference_rate_ties_half": 0.56},
                    "overall_quality": {"preference_rate_ties_half": 0.51},
                },
                "content_preservation": {"treatment_minus_sft": -0.02},
            }
            for model in MODELS
        },
    }
    embedding = {
        "artifact_schema": "humanwrite.m3.rewrite_embedding_score.v1",
        "families": {
            family: {"treatment_minus_sft": -0.001} for family in FAMILIES
        },
    }
    return automatic, judge, embedding


def test_all_frozen_4k_gates_promote():
    report = decide(*fixtures())
    assert report["passed"] is True
    assert report["decision"] == "promote_to_16k"
    assert report["failed_checks"] == []


@pytest.mark.parametrize(
    ("mutation", "failed_name"),
    [
        (("automatic", "meaningful"), "meaningful_edit_ai_inputs"),
        (("automatic", "literal"), "treatment_protected_literal_and_nonempty_sentinel"),
        (("automatic", "replacement"), "replacement_character_count"),
        (("automatic", "language"), "unexpected_non_latin_treatment_minus_sft"),
        (("automatic", "no_lexical_gain"), "at_least_one_lexical_distribution_metric_improves"),
        (("automatic", "lexical_regression"), "no_lexical_metric_worsens_over_one_human_split_sd"),
        (("judge", "style"), f"{MODELS[0]}:human_style_preference_ties_half"),
        (("judge", "quality"), f"{MODELS[0]}:overall_quality_preference_ties_half"),
        (("judge", "preservation"), f"{MODELS[0]}:content_preservation_treatment_minus_sft"),
    ],
)
def test_each_frozen_gate_fails_closed(mutation, failed_name):
    automatic, judge, embedding = (deepcopy(value) for value in fixtures())
    source, kind = mutation
    if source == "automatic":
        if kind == "meaningful": automatic["arms"]["HUMANWRITE14"]["meaningful_edit_ai_inputs"]["point"] = 0.69
        elif kind == "literal": automatic["arms"]["HUMANWRITE14"]["hard_content_preservation"]["point"] = 0.99
        elif kind == "replacement": automatic["arms"]["HUMANWRITE14"]["replacement_character_count"] = 1
        elif kind == "language": automatic["arms"]["HUMANWRITE14"]["unexpected_non_latin"]["point"] = 0.031
        elif kind == "no_lexical_gain":
            for value in automatic["lexical"].values(): value["treatment_minus_sft"] = 0.0
        elif kind == "lexical_regression": automatic["lexical"]["token_3gram_l2"]["treatment_minus_sft"] = 0.0021
    elif kind == "style": judge["models"][MODELS[0]]["pairwise"]["human_style"]["preference_rate_ties_half"] = 0.549
    elif kind == "quality": judge["models"][MODELS[0]]["pairwise"]["overall_quality"]["preference_rate_ties_half"] = 0.499
    elif kind == "preservation": judge["models"][MODELS[0]]["content_preservation"]["treatment_minus_sft"] = -0.031
    report = decide(automatic, judge, embedding)
    assert report["passed"] is False
    assert report["decision"] == "stop_after_4k"
    assert failed_name in report["failed_checks"]


def test_missing_independent_family_is_rejected():
    automatic, judge, embedding = fixtures()
    judge["models"].pop(MODELS[0])
    with pytest.raises(M3RewriteDecisionError, match="judge family"):
        decide(automatic, judge, embedding)

