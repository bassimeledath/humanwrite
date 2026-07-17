"""Independent black-box adversarial tests for measurement v2.

These tests are tester-owned.  They are ordinary assertions after the b71aaa0
qualification-boundary repair; no implementation semantics are changed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "harness" / "src"))

from harness.measurement_v2 import (  # noqa: E402
    REQUIRED_BLIND_GROUPS,
    build_attestation,
    protocol_readiness,
    validate_report_v2,
)
from harness.metrics.distribution_v2 import (  # noqa: E402
    EmbeddingPanel,
    MeasurementV2Error,
    bandwidth_hash,
    human_only_bandwidths,
    mmd_unbiased_fixed,
)
from harness.metrics.quality_v2 import (  # noqa: E402
    align_prompt_linked_references,
    grouped_authorship_auc,
    validate_selection_firewall,
)
import harness.metrics.quality_v2 as quality_v2  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def explicit_mmd(x: np.ndarray, y: np.ndarray, bandwidths: tuple[float, ...]) -> float:
    def kernel(left: np.ndarray, right: np.ndarray) -> float:
        squared_distance = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
        return sum(math.exp(-squared_distance / (2.0 * bw)) for bw in bandwidths) / len(bandwidths)

    n, m = len(x), len(y)
    xx = sum(kernel(x[i], x[j]) for i in range(n) for j in range(n) if i != j)
    yy = sum(kernel(y[i], y[j]) for i in range(m) for j in range(m) if i != j)
    xy = sum(kernel(left, right) for left in x for right in y)
    return xx / (n * (n - 1)) + yy / (m * (m - 1)) - 2.0 * xy / (n * m)


def ready_protocol() -> dict:
    hashes = {
        key: sha(key)
        for key in (
            "dependency_lock_sha256",
            "metric_code_sha256",
            "prompt_panel_sha256",
            "human_panels_sha256",
            "bandwidths_sha256",
            "power_plan_sha256",
            "selection_policy_sha256",
        )
    }
    panels = {
        name: {
            "status": "materialized",
            "frozen": True,
            "document_count": 64,
            "document_ids": [f"{prefix}-{index}" for index in range(64)],
        }
        for name, prefix in (
            ("human_eval", "e"),
            ("human_floor_a", "a"),
            ("human_floor_b", "b"),
        )
    }
    return {
        "artifact_schema": "dftr.measurement.protocol.v2",
        "status": "ready",
        "frozen": True,
        "design": {
            "documents_per_cell": 64,
            "human_pool_documents": 192,
            "replacement_sampling": False,
        },
        "hashes": hashes,
        "panels": panels,
        "bandwidth_contract": {
            "status": "frozen",
            "source": "human_floor_a_union_human_floor_b",
            "values": [0.5, 1.0],
        },
        "power": {
            field: "pass"
            for field in (
                "mmd_type_i",
                "mmd_power",
                "auc_power",
                "repetition_power",
                "coverage",
            )
        },
        "seeds": {"permutation": 41, "bootstrap": 42, "authorship_split": 43},
        "selection_policy": {"selection": {"rule_type": "fixed_seed", "seed": 29}},
        "operator_approval": {
            "approved": True,
            "reviewer": "operator",
            "reviewed_at": "2026-07-16T00:00:00Z",
        },
    }


def valid_report() -> dict:
    hashes = {
        key: sha(key)
        for key in (
            "protocol_sha256",
            "prompt_panel_sha256",
            "human_eval_sha256",
            "human_floor_a_sha256",
            "human_floor_b_sha256",
            "bandwidths_sha256",
            "power_plan_sha256",
            "dependency_lock_sha256",
            "evaluator_commit_sha256",
        )
    }
    return {
        "artifact_schema": "dftr.measurement.report.v2",
        "evidence_class": "prospective_screen",
        "counts": {
            "documents_per_cell": 64,
            "human_documents_per_panel": 64,
            "effective_prompt_clusters": 64,
        },
        "hashes": hashes,
        "seeds": {"training": [29], "sampling": [101]},
        "checkpoint_manifest": {"selection": {"rule_type": "fixed_seed", "seed": 29}},
        "distribution": {
            "documents_per_cell": 64,
            "human_documents_per_panel": 64,
            "bandwidth_sha256": hashes["bandwidths_sha256"],
            "candidate_mmd2_unbiased": 0.01,
            "control_mmd2_unbiased": 0.02,
            "human_floor_mmd2_unbiased": -0.001,
            "permutation_seed": 41,
        },
        "quality": {"status": "not_measured"},
        "repetition": {
            "status": "ready",
            "decision": "pass",
            "documents_per_panel": 64,
            "power_plan_passed": True,
        },
        "authorship": {
            "status": "ready",
            "grouped": True,
            "effective_clusters": 64,
            "fit_count": 105,
        },
        "promotion": {"eligible": False},
    }


def test_exact_oracle_includes_negative_unbiased_result() -> None:
    x = np.asarray([[0.0], [1.0], [2.0]])
    y = np.asarray([[0.1], [1.1], [2.1]])
    bandwidths = (1.0,)
    expected = explicit_mmd(x, y, bandwidths)
    assert expected < 0.0
    assert mmd_unbiased_fixed(x, y, bandwidths) == pytest.approx(expected, abs=1e-12)


def test_human_only_bandwidth_does_not_depend_on_candidate() -> None:
    floor_a = EmbeddingPanel.build("a", ["a1", "a2", "a3"], [[0, 0], [1, 0], [0, 2]])
    floor_b = EmbeddingPanel.build("b", ["b1", "b2", "b3"], [[2, 0], [1, 1], [2, 2]])
    first = human_only_bandwidths(floor_a, floor_b)
    _candidate = EmbeddingPanel.build("candidate", ["p1", "p2", "p3"], [[1e9, 1e9], [-1e9, -1e9], [0, 0]])
    second = human_only_bandwidths(floor_a, floor_b)
    assert first == second
    assert bandwidth_hash(first) == bandwidth_hash(second)


def test_protocol_rejects_unmaterialized_panel_id_lists() -> None:
    protocol = ready_protocol()
    for panel in protocol["panels"].values():
        panel["document_ids"] = []
    assert protocol_readiness(protocol)["status"] == "fail_closed"


def test_protocol_requires_calibration_baseline_and_operator_signature() -> None:
    protocol = ready_protocol()
    assert protocol_readiness(protocol)["status"] == "fail_closed"


def test_report_rejects_mismatched_frozen_bandwidth_hash() -> None:
    report = valid_report()
    report["distribution"]["bandwidth_sha256"] = sha("candidate-selected-kernel")
    with pytest.raises(MeasurementV2Error, match="bandwidth"):
        validate_report_v2(report)


def test_post_hoc_shadow_report_can_never_promote() -> None:
    report = valid_report()
    report["evidence_class"] = "post_hoc_shadow"
    report["promotion"]["eligible"] = True
    with pytest.raises(MeasurementV2Error, match="post.hoc|promotion"):
        validate_report_v2(report)


def test_underpowered_authorship_report_can_never_promote() -> None:
    report = valid_report()
    report["authorship"]["status"] = "underpowered"
    report["promotion"]["eligible"] = True
    with pytest.raises(MeasurementV2Error, match="underpowered|promotion"):
        validate_report_v2(report)


def test_report_rejects_fixed_seed_mismatch() -> None:
    report = valid_report()
    report["seeds"]["training"] = [11]
    with pytest.raises(MeasurementV2Error, match="seed"):
        validate_report_v2(report)


def test_selection_firewall_requires_a_deterministic_seed_contract() -> None:
    with pytest.raises(MeasurementV2Error, match="seed"):
        validate_selection_firewall({"selection": {"rule_type": "fixed_seed"}})
    with pytest.raises(MeasurementV2Error, match="seed"):
        validate_selection_firewall({"selection": {"rule_type": "all_preregistered_seeds", "seeds": []}})


def test_selection_firewall_rejects_other_visible_promotion_metrics() -> None:
    with pytest.raises(MeasurementV2Error, match="promotion endpoints"):
        validate_selection_firewall(
            {"selection": {"rule_type": "fixed_seed", "seed": 29, "ranking_metric": "bleu"}}
        )


def test_quality_rejects_unbound_lookalike_reference() -> None:
    generated = [{"prompt_id": "p1", "brief_sha256": sha("brief"), "text": "candidate"}]
    unrelated = [{
        "prompt_id": "p1",
        "brief_sha256": sha("brief"),
        "reference_fingerprint": sha("unrelated-human"),
        "split": "quality_visible_human",
        "text": "unrelated human text",
    }]
    with pytest.raises(MeasurementV2Error, match="fingerprint"):
        align_prompt_linked_references(generated, unrelated)


def test_authorship_reports_the_instrumented_full_pipeline_fit_count(monkeypatch) -> None:
    original_factory = quality_v2._probe_pipeline
    observed = {"fits": 0}

    class CountingPipeline:
        def __init__(self) -> None:
            self.inner = original_factory()

        def fit(self, *args, **kwargs):
            observed["fits"] += 1
            self.inner.fit(*args, **kwargs)
            return self

        def predict_proba(self, *args, **kwargs):
            return self.inner.predict_proba(*args, **kwargs)

    monkeypatch.setattr(quality_v2, "_probe_pipeline", CountingPipeline)
    generated, humans = [], []
    for index in range(8):
        cluster = f"prompt-{index}"
        generated.append({"text": f"generated robotic token stream {index}", "cluster_id": cluster})
        humans.append({"text": f"human lyrical narrative passage {index}", "cluster_id": cluster})
    result = grouped_authorship_auc(
        generated,
        humans,
        fold_seeds=(11, 12),
        uncertainty_refits=10,
        min_effective_clusters=64,
        seed=17,
    )
    assert result["status"] == "underpowered"
    assert result["fit_count"] == observed["fits"]


def test_attestation_requires_a_verifiable_operator_signature() -> None:
    protocol = ready_protocol()
    manifest = {
        "tests": [{"name": name, "status": "pass"} for name in REQUIRED_BLIND_GROUPS],
        "evaluator_commit": sha("commit"),
        "dependency_lock_sha256": sha("lock"),
        "fixture_pack_sha256": sha("fixtures"),
        "no_sealed_imitation": True,
    }
    with pytest.raises(MeasurementV2Error, match="signature"):
        build_attestation(
            protocol=protocol,
            inventory_check={"status": "pass"},
            blind_test_manifest=manifest,
            operator="operator",
            attested_at="2026-07-16T00:00:00Z",
        )
