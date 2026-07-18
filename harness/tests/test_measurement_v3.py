from __future__ import annotations

import numpy as np
import pytest

from harness.measurement_v3 import (
    EmbeddingFamily,
    MeasurementV3Error,
    UnpairedPanelDesign,
    equivalence_decision,
    human_calibrated_equivalence,
    human_calibrated_noninferiority,
    human_floor_bandwidths,
    human_floor_margin,
    noninferiority_decision,
    one_sided_absolute_mmd_permutation_test,
    paired_treatment_control_swap_test,
    prospective_exact_decision_power,
    mmd2_unbiased,
    token_unigram_l2,
    two_family_distribution_report,
)


def design(n: int = 6) -> UnpairedPanelDesign:
    return UnpairedPanelDesign.build(
        prompt_ids=[f"prompt-{index}" for index in range(n)],
        prompt_source_document_ids=[f"source-{index}" for index in range(n)],
        distribution_reference_ids=[f"reference-{index}" for index in range(n)],
        human_floor_a_ids=[f"floor-a-{index}" for index in range(n)],
        human_floor_b_ids=[f"floor-b-{index}" for index in range(n)],
    )


def family(panel: UnpairedPanelDesign, name: str, scale: float) -> EmbeddingFamily:
    n = len(panel.prompt_ids)
    reference = np.column_stack((np.linspace(-0.2, 0.2, n), np.zeros(n))) * scale
    treatment = reference + np.asarray([0.01, 0.0]) * scale
    control = reference + np.asarray([2.0, 1.0]) * scale
    floor_a = np.column_stack((np.linspace(-0.3, 0.1, n), np.full(n, -0.05))) * scale
    floor_b = np.column_stack((np.linspace(-0.1, 0.3, n), np.full(n, 0.05))) * scale
    return EmbeddingFamily.build(
        panel,
        family_id=name,
        model_id=f"vendor/{name}",
        model_revision=f"revision-{name}",
        treatment=treatment,
        control=control,
        distribution_reference=reference,
        human_floor_a=floor_a,
        human_floor_b=floor_b,
    )


def test_unpaired_design_rejects_prompt_source_leakage() -> None:
    with pytest.raises(MeasurementV3Error, match="prompt-source"):
        UnpairedPanelDesign.build(
            prompt_ids=["p1", "p2"],
            prompt_source_document_ids=["source-1", "leaked"],
            distribution_reference_ids=["leaked", "reference-2"],
            human_floor_a_ids=["a1", "a2"],
            human_floor_b_ids=["b1", "b2"],
        )
    assert (
        design().prompt_source_document_ids[0] != design().distribution_reference_ids[0]
    )


def test_two_independent_embedding_families_are_required() -> None:
    panel = design()
    first, second = family(panel, "bge", 1.0), family(panel, "e5", 3.0)
    report = two_family_distribution_report(
        panel, [first, second], permutation_draws=63, seed=12
    )
    assert report["family_count"] == 2
    assert report["primary_direction_agreement"] is True
    assert set(report["families"]) == {"bge", "e5"}
    with pytest.raises(MeasurementV3Error, match="exactly two"):
        two_family_distribution_report(panel, [first], permutation_draws=7)
    with pytest.raises(MeasurementV3Error, match="training reward"):
        two_family_distribution_report(
            panel,
            [first, second],
            training_reward_model_ids=[second.model_id],
            permutation_draws=7,
        )


def test_absolute_mmd_uses_raw_one_sided_right_tail() -> None:
    sample = np.asarray([[0.0], [1.0]])
    human = sample.copy()
    bandwidths = (1.0,)
    result = one_sided_absolute_mmd_permutation_test(
        sample, human, bandwidths, exact=True
    )
    assert result["statistic"] < 0
    assert result["absolute_value_transform"] is False
    assert result["tail"] == "right"
    assert result["pvalue"] == 1.0

    separated = one_sided_absolute_mmd_permutation_test(
        np.asarray([[8.0], [9.0], [10.0], [11.0]]),
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        bandwidths,
        exact=True,
    )
    assert separated["statistic"] > 0
    assert separated["pvalue"] < 0.05


def test_paired_swap_is_one_sided_and_prompt_clustered() -> None:
    n = 8
    reference = np.arange(n, dtype=float)[:, None] / 10
    treatment = reference.copy()
    control = reference + 4.0
    bandwidths = (1.0,)
    prompts = [f"p-{index}" for index in range(n)]
    improved = paired_treatment_control_swap_test(
        treatment,
        control,
        reference,
        bandwidths,
        prompt_ids=prompts,
        exact=True,
    )
    wrong_tail = paired_treatment_control_swap_test(
        treatment,
        control,
        reference,
        bandwidths,
        prompt_ids=prompts,
        alternative="treatment_higher",
        exact=True,
    )
    assert improved["effect"] < 0
    assert improved["pvalue"] < 0.05
    assert wrong_tail["pvalue"] > 0.95
    assert improved["effective_prompt_clusters"] == n


def test_precomputed_permutation_statistics_match_direct_mmd() -> None:
    rng = np.random.default_rng(91)
    treatment = rng.normal(size=(6, 3))
    control = rng.normal(size=(6, 3))
    reference = rng.normal(size=(9, 3))
    bandwidths = (0.5, 1.5)
    direct_effect = mmd2_unbiased(
        treatment, reference, bandwidths
    ) - mmd2_unbiased(control, reference, bandwidths)

    result = paired_treatment_control_swap_test(
        treatment,
        control,
        reference,
        bandwidths,
        prompt_ids=[f"p-{index}" for index in range(6)],
        exact=True,
    )
    assert result["effect"] == pytest.approx(direct_effect, abs=1e-14)


def test_token_unigram_l2_uses_normalized_corpus_counts() -> None:
    result = token_unigram_l2([["a", "a"], ["b"]], [["a"], ["b", "b"]])
    expected = np.sqrt((2 / 3 - 1 / 3) ** 2 + (1 / 3 - 2 / 3) ** 2)
    assert result["l2"] == pytest.approx(expected)
    assert result["sample_tokens"] == result["human_tokens"] == 3
    with pytest.raises(MeasurementV3Error, match="token sequences"):
        token_unigram_l2(["raw text"], [["token"]])


def test_equivalence_and_noninferiority_use_complete_interval() -> None:
    assert (
        equivalence_decision(interval_low=-0.1, interval_high=0.1, margin=0.2)[
            "decision"
        ]
        == "pass"
    )
    assert (
        equivalence_decision(interval_low=-0.1, interval_high=0.21, margin=0.2)[
            "decision"
        ]
        == "fail"
    )
    assert (
        noninferiority_decision(
            interval_low=-0.3, interval_high=0.09, margin=0.1, lower_is_better=True
        )["decision"]
        == "pass"
    )
    assert (
        noninferiority_decision(
            interval_low=-0.09, interval_high=0.4, margin=0.1, lower_is_better=False
        )["decision"]
        == "pass"
    )


def test_human_calibrated_bootstraps_are_deterministic() -> None:
    candidate = [0.99, 1.0, 1.01, 1.0, 1.0, 0.99]
    human = [1.0, 1.01, 0.99, 1.0, 1.0, 1.01]
    first = human_calibrated_equivalence(
        candidate, human, margin=0.05, draws=1_000, seed=7
    )
    second = human_calibrated_equivalence(
        candidate, human, margin=0.05, draws=1_000, seed=7
    )
    assert first == second
    assert first["decision"] == "pass"
    noninferior = human_calibrated_noninferiority(
        candidate,
        human,
        margin=0.05,
        lower_is_better=True,
        draws=1_000,
        seed=8,
    )
    assert noninferior["decision"] == "pass"
    calibrated = human_floor_margin(
        [0.9, 1.0, 1.1, 1.0],
        [1.0, 0.95, 1.05, 1.0],
        draws=1_000,
        seed=9,
    )
    assert calibrated == human_floor_margin(
        [0.9, 1.0, 1.1, 1.0],
        [1.0, 0.95, 1.05, 1.0],
        draws=1_000,
        seed=9,
    )
    assert calibrated["margin"] > 0


def test_power_executes_exact_composite_rule_and_requires_beyond_boundary() -> None:
    calls = {"count": 0}

    def generator(mean: float):
        def draw(rng: np.random.Generator, _index: int):
            primary = float(rng.normal(mean, 1.0, 32).mean())
            safety = bool(rng.random() < 0.995)
            return {"primary": primary, "safety": safety}

        return draw

    def exact_rule(trial) -> bool:
        calls["count"] += 1
        return bool(trial["primary"] > 0.35 and trial["safety"])

    result = prospective_exact_decision_power(
        null_generator=generator(0.0),
        alternative_generator=generator(0.9),
        decision_rule=exact_rule,
        rule_id="primary-and-safety-v1",
        trials=1_000,
        seed=19,
        decision_boundary=0.35,
        alternative_effect=0.9,
        effect_direction="greater",
        type_i_max=0.05,
        power_min=0.80,
    )
    assert calls["count"] == 2_000
    assert result["type_i_pass"] is True
    assert result["power_pass"] is True
    assert result["all_targets_pass"] is True
    with pytest.raises(MeasurementV3Error, match="strictly beyond"):
        prospective_exact_decision_power(
            null_generator=generator(0.0),
            alternative_generator=generator(0.35),
            decision_rule=exact_rule,
            rule_id="bad-boundary-v1",
            trials=1_000,
            seed=19,
            decision_boundary=0.35,
            alternative_effect=0.35,
            effect_direction="greater",
        )


def test_human_only_bandwidth_rejects_degenerate_floors() -> None:
    with pytest.raises(MeasurementV3Error, match="degenerate"):
        human_floor_bandwidths(np.zeros((3, 2)), np.zeros((3, 2)))
