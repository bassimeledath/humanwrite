import math

import numpy as np
import pytest

from harness.metrics.distribution_v2 import (
    EmbeddingPanel,
    MeasurementV2Error,
    bandwidth_hash,
    clustered_paired_interval,
    common_kernel_report,
    fixed_kernel,
    human_only_bandwidths,
    mmd_unbiased_fixed,
    paired_prompt_swap_pvalue,
    require_disjoint_equal_human_panels,
)


def panel(name, ids, values):
    return EmbeddingPanel.build(name, ids, values)


def explicit_mmd(x, y, bandwidths):
    def kernel(left, right):
        return sum(
            math.exp(-sum((a - b) ** 2 for a, b in zip(left, right)) / (2 * bw))
            for bw in bandwidths
        ) / len(bandwidths)

    xx = sum(kernel(x[i], x[j]) for i in range(len(x)) for j in range(len(x)) if i != j)
    yy = sum(kernel(y[i], y[j]) for i in range(len(y)) for j in range(len(y)) if i != j)
    xy = sum(kernel(left, right) for left in x for right in y)
    return xx / (len(x) * (len(x) - 1)) + yy / (len(y) * (len(y) - 1)) - 2 * xy / (len(x) * len(y))


def test_small_matrix_oracle_matches_explicit_unbiased_formula():
    x = np.asarray([[0.0, 0.0], [1.0, 0.5], [0.5, 1.5]])
    y = np.asarray([[0.1, 0.2], [1.1, 0.4], [0.4, 1.2]])
    bandwidths = (0.3, 1.2)
    assert mmd_unbiased_fixed(x, y, bandwidths) == pytest.approx(
        explicit_mmd(x.tolist(), y.tolist(), bandwidths), abs=1e-12
    )
    assert fixed_kernel(x, y, bandwidths).shape == (3, 3)


def test_human_only_bandwidth_is_candidate_invariant():
    floor_a = panel("a", ["a1", "a2", "a3"], [[0, 0], [1, 0], [0, 2]])
    floor_b = panel("b", ["b1", "b2", "b3"], [[2, 0], [1, 1], [2, 2]])
    frozen = human_only_bandwidths(floor_a, floor_b)
    wildly_different_candidate = panel("candidate", ["p1", "p2", "p3"], [[99, 99], [-99, -99], [50, -50]])
    assert human_only_bandwidths(floor_a, floor_b) == frozen
    assert len(bandwidth_hash(frozen)) == 64
    assert wildly_different_candidate.embeddings.shape == (3, 2)


def test_panels_fail_closed_on_replacement_overlap_and_cardinality():
    with pytest.raises(MeasurementV2Error, match="duplicate"):
        panel("bad", ["same", "same"], [[0], [1]])
    evaluation = panel("eval", ["e1", "e2"], [[0], [1]])
    floor_a = panel("a", ["a1", "a2"], [[2], [3]])
    overlap = panel("b", ["a2", "b2"], [[4], [5]])
    with pytest.raises(MeasurementV2Error, match="disjoint"):
        require_disjoint_equal_human_panels(evaluation, floor_a, overlap)
    short = panel("short", ["s1"], [[6]])
    with pytest.raises(MeasurementV2Error, match="cardinality"):
        require_disjoint_equal_human_panels(evaluation, floor_a, short)


def test_common_kernel_report_uses_equal_n_one_hash_and_raw_cells():
    candidate = panel("candidate", ["p1", "p2", "p3"], [[0.0], [1.0], [2.0]])
    control = panel("control", ["p1", "p2", "p3"], [[0.2], [1.2], [2.2]])
    evaluation = panel("eval", ["e1", "e2", "e3"], [[0.1], [1.1], [2.1]])
    floor_a = panel("a", ["a1", "a2", "a3"], [[0.0], [0.9], [2.1]])
    floor_b = panel("b", ["b1", "b2", "b3"], [[0.2], [1.0], [2.0]])
    bandwidths = human_only_bandwidths(floor_a, floor_b)
    report = common_kernel_report(
        candidate, control, evaluation, floor_a, floor_b, bandwidths,
        permutation_draws=19, seed=41,
    )
    assert report["documents_per_cell"] == report["human_documents_per_panel"] == 3
    assert report["bandwidth_sha256"] == bandwidth_hash(bandwidths)
    assert {"candidate_mmd2_unbiased", "control_mmd2_unbiased", "human_floor_mmd2_unbiased"} <= report.keys()
    assert 0 < report["paired_candidate_control_p"] <= 1


def test_common_kernel_aligns_control_by_prompt_id_and_rejects_set_changes():
    candidate = panel("candidate", ["p1", "p2", "p3"], [[0.0], [1.0], [2.0]])
    control = panel("control", ["p1", "p2", "p3"], [[0.2], [1.2], [2.2]])
    reordered = panel("control", ["p3", "p1", "p2"], [[2.2], [0.2], [1.2]])
    evaluation = panel("eval", ["e1", "e2", "e3"], [[0.1], [1.1], [2.1]])
    floor_a = panel("a", ["a1", "a2", "a3"], [[0.0], [0.9], [2.1]])
    floor_b = panel("b", ["b1", "b2", "b3"], [[0.2], [1.0], [2.0]])
    bandwidths = human_only_bandwidths(floor_a, floor_b)
    ordered_report = common_kernel_report(
        candidate, control, evaluation, floor_a, floor_b, bandwidths,
        permutation_draws=9, seed=12,
    )
    reordered_report = common_kernel_report(
        candidate, reordered, evaluation, floor_a, floor_b, bandwidths,
        permutation_draws=9, seed=12,
    )
    assert reordered_report == ordered_report
    replaced = panel("control", ["p1", "p2", "other"], [[0.2], [1.2], [2.2]])
    with pytest.raises(MeasurementV2Error, match="sets must match"):
        common_kernel_report(
            candidate, replaced, evaluation, floor_a, floor_b, bandwidths,
            permutation_draws=2,
        )


def test_paired_prompt_swap_rejects_nonunique_or_misaligned_prompts():
    values = np.asarray([[0.0], [1.0]])
    with pytest.raises(MeasurementV2Error, match="unique aligned"):
        paired_prompt_swap_pvalue(values, values, values, (1.0,), prompt_ids=["p", "p"], draws=2)


def test_cluster_interval_reports_effective_prompt_not_seed_rows():
    result = clustered_paired_interval(
        [2.0, 2.2, 1.0, 1.1], [1.0, 1.2, 1.5, 1.6],
        ["prompt-1", "prompt-1", "prompt-2", "prompt-2"], draws=50, seed=9,
    )
    assert result["effective_clusters"] == 2
    assert result["effect"] == pytest.approx(0.25)
    unbalanced_repeats = clustered_paired_interval(
        [2.0, 2.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.5],
        ["prompt-1", "prompt-1", "prompt-1", "prompt-2"], draws=20, seed=9,
    )
    assert unbalanced_repeats["effect"] == pytest.approx(0.25)
