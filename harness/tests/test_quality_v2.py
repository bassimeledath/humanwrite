import pytest

from harness.metrics.distribution_v2 import MeasurementV2Error
from harness.metrics.quality_v2 import (
    align_prompt_linked_references,
    grouped_authorship_auc,
    prompt_linked_quality,
    validate_selection_firewall,
)


def generated(prompt_id, brief="brief", fingerprint=None):
    row = {
        "prompt_id": prompt_id,
        "brief_sha256": brief,
        "text": f"generated prose for {prompt_id} with a distinct cadence",
        "cluster_id": prompt_id,
    }
    if fingerprint:
        row["reference_fingerprint"] = fingerprint
    return row


def human(prompt_id, brief="brief", fingerprint=None):
    return {
        "prompt_id": prompt_id,
        "brief_sha256": brief,
        "reference_fingerprint": fingerprint or f"human-{prompt_id}",
        "split": "quality_visible_human",
        "text": f"human essay for {prompt_id} with varied phrasing",
        "cluster_id": prompt_id,
    }


def test_prompt_linkage_is_one_to_one_provenanced_and_order_independent():
    generated_rows = [generated("p1", fingerprint="human-p1"), generated("p2")]
    human_rows = [human("p2"), human("p1")]
    aligned = align_prompt_linked_references(generated_rows, human_rows)
    assert [pair[0]["prompt_id"] for pair in aligned] == ["p1", "p2"]
    with pytest.raises(MeasurementV2Error, match="brief hash mismatch"):
        align_prompt_linked_references(generated_rows, [human("p2"), human("p1", brief="wrong")])
    with pytest.raises(MeasurementV2Error, match="one-to-one"):
        align_prompt_linked_references(generated_rows, [human("p1", fingerprint="dup"), human("p2", fingerprint="dup")])
    wrong_split = human("p2")
    wrong_split["split"] = "sealed_human"
    with pytest.raises(MeasurementV2Error, match="provenance"):
        align_prompt_linked_references(generated_rows, [human("p1"), wrong_split])


def test_quality_unavailable_is_explicitly_not_measured():
    assert prompt_linked_quality([generated("p")], [human("p")], None)["status"] == "not_measured"


def test_prompt_linked_quality_reports_prompt_cluster_uncertainty():
    rows = [generated("p1"), generated("p2")]
    references = [human("p1"), human("p2")]
    result = prompt_linked_quality(
        rows, references, lambda **kwargs: "TIE", bootstrap_draws=20, seed=3
    )
    assert result["status"] == "measured"
    assert result["jmq"] == 1.0
    assert result["effective_prompt_clusters"] == 2


def test_grouped_authorship_refits_full_pipeline_and_marks_small_n_underpowered():
    generated_rows = []
    human_rows = []
    for index in range(8):
        cluster = f"prompt-{index}"
        generated_rows.append({"text": f"generated robotic token stream number {index}", "cluster_id": cluster})
        human_rows.append({"text": f"human lyrical narrative passage number {index}", "cluster_id": cluster})
    result = grouped_authorship_auc(
        generated_rows, human_rows, fold_seeds=(11, 12),
        uncertainty_refits=20, min_effective_clusters=64, seed=17,
    )
    assert result["status"] == "underpowered"
    assert result["effective_clusters"] == 8
    assert result["fit_count"] >= 12
    assert result["separability"] == pytest.approx(abs(result["auc"] - 0.5))


def test_selection_firewall_rejects_promotion_endpoint_selection():
    accepted = validate_selection_firewall(
        {"selection": {"rule_type": "fixed_seed", "seed": 29}}
    )
    assert accepted == {"status": "pass", "rule_type": "fixed_seed"}
    assert validate_selection_firewall(
        {"selection": {"rule_type": "all_preregistered_seeds", "seeds": [11, 29, 47]}}
    )["status"] == "pass"
    with pytest.raises(MeasurementV2Error, match="promotion endpoints"):
        validate_selection_firewall(
            {"selection": {"rule_type": "fixed_seed", "ranking_metric": "authorship_auc"}}
        )
    with pytest.raises(MeasurementV2Error, match="promotion endpoints"):
        validate_selection_firewall(
            {"selection": {"rule_type": "fixed_seed", "ranking_metric": "S"}}
        )
