from types import SimpleNamespace

import numpy as np
import pytest

from harness.metrics.quality import authorship_auc, fresh_authorship_auc, jmq, quality_preference


def generated_wins(*, prompt, candidate_a, candidate_b):
    del prompt
    return "A" if "GENERATED" in candidate_a else "B"


def test_quality_randomizes_order_but_maps_winner_to_model():
    generated = ["GENERATED one", "GENERATED two", "GENERATED three"]
    human = ["human one", "human two", "human three"]
    prompts = ["p1", "p2", "p3"]
    assert quality_preference(generated, human, prompts, generated_wins) == 1.0
    assert jmq(generated, human, prompts, generated_wins) == 2.0


def test_quality_ties_are_half_and_lengths_checked():
    assert quality_preference(["g"], ["h"], ["p"], lambda **_: {"winner": "tie"}) == 0.5
    with pytest.raises(ValueError, match="equal"):
        quality_preference(["g"], [], ["p"], generated_wins)


def test_openai_compatible_judge_abstraction():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="TIE"))])

    judge = SimpleNamespace(
        model="offline-judge",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    assert quality_preference(["g"], ["h"], ["prompt"], judge) == 0.5
    assert captured["temperature"] == 0


class Probe:
    def predict_proba(self, texts):
        scores = np.array([0.1 if text.startswith("human") else 0.9 for text in texts])
        return np.column_stack([1 - scores, scores])


def test_authorship_auc_and_ci_are_deterministic():
    args = (["generated 1", "generated 2"], ["human 1", "human 2"], Probe())
    first = authorship_auc(*args)
    assert first == authorship_auc(*args)
    assert first == (1.0, 1.0, 1.0)


def test_fresh_authorship_probe_uses_deterministic_out_of_fold_predictions():
    generated = [
        "Generated output uses a repeated synthetic cadence.",
        "Generated response uses a repeated synthetic pattern.",
        "Generated prose uses a repeated machine cadence.",
        "Generated answer uses a repeated machine pattern.",
    ]
    humans = [
        "I found the old map beneath a drawer.",
        "Rain moved softly over the empty field.",
        "The committee met after lunch on Tuesday.",
        "Several bright boats crossed the bay at dawn.",
    ]
    first = fresh_authorship_auc(generated, humans)
    assert first == fresh_authorship_auc(generated, humans)
    assert 0.0 <= first[0] <= 1.0
    assert first[1] <= first[0] <= first[2]


def test_fresh_authorship_probe_fails_closed_when_oof_is_impossible():
    with pytest.raises(ValueError, match="at least two"):
        fresh_authorship_auc(["generated"], ["human one", "human two"])
