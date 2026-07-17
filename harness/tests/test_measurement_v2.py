import hashlib
import json
from pathlib import Path

import pytest

from harness.measurement_v2 import (
    REQUIRED_BLIND_GROUPS,
    build_attestation,
    prepare_protocol_transfer,
    protocol_readiness,
    validate_report_v2,
    verify_historical_inventory,
)
from harness.metrics.distribution_v2 import MeasurementV2Error


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "harness" / "measurement_v2"


def sha(value="x"):
    return hashlib.sha256(value.encode()).hexdigest()


def test_checked_in_historical_inventory_is_byte_verified():
    inventory = json.loads((V2_ROOT / "historical_v1_inventory.json").read_text())
    result = verify_historical_inventory(inventory, repo_root=REPO_ROOT)
    assert result["status"] == "pass"
    assert all(row["status"] == "pass" for row in result["artifact_sets"])


def test_historical_inventory_detects_changed_bytes(tmp_path):
    artifact = tmp_path / "historical.json"
    artifact.write_text("original")
    file_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(f"historical.json\0{file_digest}\n".encode()).hexdigest()
    inventory = {
        "artifact_schema": "dftr.measurement.historical_inventory.v1",
        "artifact_sets": [{
            "name": "test", "include_globs": ["*.json"], "file_count": 1,
            "manifest_sha256": manifest_digest,
        }],
    }
    assert verify_historical_inventory(inventory, repo_root=tmp_path)["status"] == "pass"
    artifact.write_text("changed")
    assert verify_historical_inventory(inventory, repo_root=tmp_path)["status"] == "fail"


def test_unmaterialized_candidate_is_fail_closed_and_not_transferable():
    path = V2_ROOT / "measurement_protocol_v2.candidate.json"
    candidate = json.loads(path.read_text())
    readiness = protocol_readiness(candidate)
    assert readiness["status"] == "fail_closed"
    assert "protocol_not_frozen" in readiness["reasons"]
    with pytest.raises(MeasurementV2Error, match="operator approval"):
        prepare_protocol_transfer(path, hashlib.sha256(path.read_bytes()).hexdigest())


def valid_report():
    hashes = {
        key: sha(key)
        for key in (
            "protocol_sha256", "prompt_panel_sha256", "human_eval_sha256",
            "human_floor_a_sha256", "human_floor_b_sha256", "bandwidths_sha256",
            "power_plan_sha256", "dependency_lock_sha256", "evaluator_commit_sha256",
        )
    }
    return {
        "artifact_schema": "dftr.measurement.report.v2",
        "evidence_class": "prospective_screen",
        "counts": {"documents_per_cell": 64, "human_documents_per_panel": 64, "effective_prompt_clusters": 64},
        "hashes": hashes,
        "seeds": {"training": [29], "sampling": [101]},
        "checkpoint_manifest": {"selection": {"rule_type": "fixed_seed", "seed": 29}},
        "distribution": {
            "documents_per_cell": 64,
            "human_documents_per_panel": 64,
            "bandwidth_sha256": sha("kernel"),
            "candidate_mmd2_unbiased": 0.01,
            "control_mmd2_unbiased": 0.02,
            "human_floor_mmd2_unbiased": -0.001,
            "permutation_seed": 41,
        },
        "quality": {"status": "not_measured"},
        "repetition": {"status": "underpowered", "documents_per_panel": 64, "power_plan_passed": False},
        "authorship": {"status": "ready", "grouped": True, "effective_clusters": 64, "fit_count": 105},
        "promotion": {"eligible": False},
    }


def test_report_requires_matched_n_prompt_n_and_nonpromoting_underpowered_state():
    report = valid_report()
    assert validate_report_v2(report)["status"] == "pass"
    report["counts"]["effective_prompt_clusters"] = 32
    with pytest.raises(MeasurementV2Error, match="inflate"):
        validate_report_v2(report)
    malformed = valid_report()
    malformed["counts"]["documents_per_cell"] = "64"
    with pytest.raises(MeasurementV2Error, match="must be an integer"):
        validate_report_v2(malformed)


def test_attestation_requires_all_thirteen_blind_groups():
    assert len(REQUIRED_BLIND_GROUPS) == 13
    with pytest.raises(MeasurementV2Error):
        build_attestation(
            protocol={}, inventory_check={"status": "pass"}, blind_test_manifest={},
            operator="operator", attested_at="2026-07-16T00:00:00Z",
        )


def ready_protocol():
    hashes = {
        key: sha(key)
        for key in (
            "dependency_lock_sha256", "metric_code_sha256", "prompt_panel_sha256",
            "human_panels_sha256", "bandwidths_sha256", "power_plan_sha256",
            "selection_policy_sha256",
        )
    }
    panels = {
        name: {
            "status": "materialized", "frozen": True, "document_count": 64,
            "document_ids": [f"{prefix}-{index}" for index in range(64)],
        }
        for name, prefix in (("human_eval", "e"), ("human_floor_a", "a"), ("human_floor_b", "b"))
    }
    return {
        "artifact_schema": "dftr.measurement.protocol.v2",
        "status": "ready",
        "frozen": True,
        "design": {"documents_per_cell": 64, "human_pool_documents": 192, "replacement_sampling": False},
        "hashes": hashes,
        "panels": panels,
        "bandwidth_contract": {"status": "frozen", "source": "human_floor_a_union_human_floor_b", "values": [0.5, 1.0]},
        "power": {field: "pass" for field in ("mmd_type_i", "mmd_power", "auc_power", "repetition_power", "coverage")},
        "seeds": {"permutation": 41, "bootstrap": 42, "authorship_split": 43},
        "selection_policy": {"selection": {"rule_type": "fixed_seed", "seed": 29}},
        "operator_approval": {"approved": True, "reviewer": "operator", "reviewed_at": "2026-07-16T00:00:00Z"},
    }


def test_complete_operator_attestation_binds_protocol_and_blind_evidence():
    manifest = {
        "tests": [{"name": name, "status": "pass"} for name in REQUIRED_BLIND_GROUPS],
        "evaluator_commit": sha("commit"),
        "dependency_lock_sha256": sha("lock"),
        "fixture_pack_sha256": sha("fixtures"),
        "no_sealed_imitation": True,
    }
    result = build_attestation(
        protocol=ready_protocol(),
        inventory_check={"status": "pass"},
        blind_test_manifest=manifest,
        operator="independent-operator",
        attested_at="2026-07-16T00:00:00Z",
    )
    assert result["status"] == "qualified"
    assert result["historical_inventory_verified"] is True
    assert len(result["blind_test_groups"]) == 13
    assert len(result["protocol_sha256"]) == 64
