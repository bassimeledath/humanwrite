"""Second-round tester-owned adversarial checks for real evidence binding."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "harness" / "src"))

from harness.measurement_v2 import (  # noqa: E402
    REQUIRED_BLIND_GROUPS,
    build_attestation,
    protocol_readiness,
    validate_report_v2,
)
from harness.metrics.distribution_v2 import MeasurementV2Error, bandwidth_hash  # noqa: E402


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode())


def write_json(root: Path, name: str, value: dict) -> str:
    path = root / name
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return sha_bytes(path.read_bytes())


def sign(document: dict, private_key: Ed25519PrivateKey) -> None:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document["operator_signature"] = {
        "algorithm": "ed25519",
        "key_id": "independent-operator-key",
        "signed_payload_sha256": sha_bytes(payload),
        "signature_base64": base64.b64encode(private_key.sign(payload)).decode(),
    }


def bound_protocol(
    root: Path,
    *,
    baseline_row_count: int = 64,
    truthful_human_fingerprints: bool = True,
) -> tuple[dict, Ed25519PrivateKey, dict[str, str]]:
    """Build independently signed public synthetic evidence with controllable defects."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trusted = {"independent-operator-key": base64.b64encode(public_key).decode()}

    (root / "uv.lock").write_text("locked synthetic dependencies\n")
    (root / "evaluator.bundle").write_text("content-addressed synthetic evaluator bundle\n")
    (root / "prompt-brief.txt").write_text("complete frozen brief bytes")
    dependency_sha = sha_bytes((root / "uv.lock").read_bytes())
    evaluator_sha = sha_bytes((root / "evaluator.bundle").read_bytes())
    brief_sha = sha_bytes((root / "prompt-brief.txt").read_bytes())

    prompt_ids = [f"prompt-{index}" for index in range(64)]
    prompt_panel_sha = write_json(root, "prompt-panel.json", {
        "artifact_schema": "dftr.measurement.prompt_panel.v2",
        "status": "frozen",
        "frozen": True,
        "prompt_ids": prompt_ids,
        "full_brief_sha256": brief_sha,
    })

    protocol_panels, manifest_panels, content_rows = {}, {}, []
    for panel_name, prefix in (
        ("human_eval", "eval"),
        ("human_floor_a", "floor-a"),
        ("human_floor_b", "floor-b"),
    ):
        ids = [f"{prefix}-{index}" for index in range(64)]
        manifest_rows = []
        for document_id in ids:
            text = f"actual visible human content for {document_id}"
            content_rows.append({"document_id": document_id, "text": text})
            fingerprint_source = text if truthful_human_fingerprints else f"unrelated claim for {document_id}"
            manifest_rows.append({
                "document_id": document_id,
                "content_sha256": sha_text(fingerprint_source),
            })
        manifest_panels[panel_name] = manifest_rows
        protocol_panels[panel_name] = {
            "status": "materialized",
            "frozen": True,
            "document_count": 64,
            "document_ids": ids,
            "content_manifest_sha256": sha_text(
                json.dumps(manifest_rows, sort_keys=True, separators=(",", ":"))
            ),
        }
    human_content_path = root / "human-content.jsonl"
    human_content_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in content_rows)
    )
    human_content_sha = sha_bytes(human_content_path.read_bytes())
    human_panel_sha = write_json(root, "human-panels.json", {
        "artifact_schema": "dftr.measurement.human_panels.v2",
        "status": "materialized",
        "frozen": True,
        "target_n_per_panel": 64,
        "sampling": "disjoint_without_replacement",
        "content_bundle_sha256": human_content_sha,
        "eligibility_attestation_sha256": sha_text("synthetic eligibility assertion"),
        "panels": manifest_panels,
    })

    values = [0.5, 1.0]
    bandwidth_artifact_sha = write_json(root, "bandwidths.json", {
        "artifact_schema": "dftr.measurement.bandwidths.v2",
        "status": "frozen",
        "frozen": True,
        "source": "human_floor_a_union_human_floor_b_only",
        "values": values,
        "panel_manifest_sha256": human_panel_sha,
        "floor_a_content_manifest_sha256": protocol_panels["human_floor_a"]["content_manifest_sha256"],
        "floor_b_content_manifest_sha256": protocol_panels["human_floor_b"]["content_manifest_sha256"],
        "embedder_sha256": sha_text("visible embedder revision"),
        "preprocessing_sha256": sha_text("visible preprocessing contract"),
        "bandwidth_sha256": bandwidth_hash(values),
    })

    sampling_grid_sha = sha_text("sampling grid")
    seed_grid = [{"training_seed": 29, "sampling_seeds": [101]}]
    output_path = root / "matched-control-outputs.jsonl"
    output_path.write_text("".join(
        json.dumps({
            "prompt_id": prompt_ids[index % len(prompt_ids)],
            "training_seed": 29,
            "sampling_seed": 101,
            "text": f"control output {index}",
        }, sort_keys=True) + "\n"
        for index in range(baseline_row_count)
    ))
    output_sha = sha_bytes(output_path.read_bytes())
    baseline_sha = write_json(root, "matched-baseline.json", {
        "artifact_schema": "dftr.measurement.matched_sft_baseline.v2",
        "status": "materialized",
        "frozen": True,
        "documents_per_cell": 64,
        "prompt_panel_sha256": prompt_panel_sha,
        "full_brief_sha256": brief_sha,
        "sampling_grid_sha256": sampling_grid_sha,
        "seed_grid": seed_grid,
        "output_manifest_sha256": output_sha,
    })
    calibration_sha = write_json(root, "calibration.json", {
        "artifact_schema": "dftr.measurement.calibration.v2",
        "status": "frozen",
        "frozen": True,
        "documents_per_cell": 64,
        "human_panels_sha256": human_panel_sha,
        "bandwidths_sha256": bandwidth_artifact_sha,
        "matched_baseline_sha256": baseline_sha,
        "dependency_lock_sha256": dependency_sha,
    })
    power_sha = write_json(root, "power.json", {
        "artifact_schema": "dftr.measurement.power_plan.v2",
        "status": "frozen",
        "frozen": True,
        "all_targets_pass": True,
        "documents_per_cell": 64,
        "human_panels_sha256": human_panel_sha,
        "bandwidths_sha256": bandwidth_artifact_sha,
        "calibration_sha256": calibration_sha,
        "matched_baseline_sha256": baseline_sha,
        "dependency_lock_sha256": dependency_sha,
        "results": {
            "mmd_type_i_rate": 0.049,
            "mmd_type_i_max": 0.05,
            "mmd_power": 0.81,
            "auc_power": 0.82,
            "repetition_power": 0.83,
            "coverage": 0.95,
        },
        "multiplicity": {"method": "holm"},
    })
    selection = {"rule_type": "fixed_seed", "seed": 29}
    selection_sha = write_json(root, "selection.json", {
        "artifact_schema": "dftr.measurement.selection_policy.v2",
        "status": "frozen",
        "frozen": True,
        "selection": selection,
    })

    hashes = {
        "dependency_lock_sha256": dependency_sha,
        "metric_code_sha256": evaluator_sha,
        "prompt_panel_sha256": prompt_panel_sha,
        "prompt_brief_sha256": brief_sha,
        "human_panels_sha256": human_panel_sha,
        "human_panel_contents_sha256": human_content_sha,
        "bandwidths_sha256": bandwidth_artifact_sha,
        "power_plan_sha256": power_sha,
        "calibration_sha256": calibration_sha,
        "matched_baseline_sha256": baseline_sha,
        "matched_baseline_outputs_sha256": output_sha,
        "selection_policy_sha256": selection_sha,
    }
    paths = {
        "dependency_lock": "uv.lock",
        "metric_code": "evaluator.bundle",
        "prompt_panel": "prompt-panel.json",
        "prompt_brief": "prompt-brief.txt",
        "human_panels": "human-panels.json",
        "human_panel_contents": "human-content.jsonl",
        "bandwidths": "bandwidths.json",
        "power_plan": "power.json",
        "calibration": "calibration.json",
        "matched_baseline": "matched-baseline.json",
        "matched_baseline_outputs": "matched-control-outputs.jsonl",
        "selection_policy": "selection.json",
    }
    hash_fields = {
        "dependency_lock": "dependency_lock_sha256",
        "metric_code": "metric_code_sha256",
        "prompt_panel": "prompt_panel_sha256",
        "prompt_brief": "prompt_brief_sha256",
        "human_panels": "human_panels_sha256",
        "human_panel_contents": "human_panel_contents_sha256",
        "bandwidths": "bandwidths_sha256",
        "power_plan": "power_plan_sha256",
        "calibration": "calibration_sha256",
        "matched_baseline": "matched_baseline_sha256",
        "matched_baseline_outputs": "matched_baseline_outputs_sha256",
        "selection_policy": "selection_policy_sha256",
    }
    protocol = {
        "artifact_schema": "dftr.measurement.protocol.v2",
        "status": "ready",
        "frozen": True,
        "design": {
            "documents_per_cell": 64,
            "human_pool_documents": 192,
            "replacement_sampling": False,
        },
        "hashes": hashes,
        "panels": protocol_panels,
        "bandwidth_contract": {
            "status": "frozen",
            "source": "human_floor_a_union_human_floor_b",
            "values": values,
            "bandwidth_value_sha256": bandwidth_hash(values),
        },
        "power": {
            field: "pass"
            for field in ("mmd_type_i", "mmd_power", "auc_power", "repetition_power", "coverage")
        },
        "seeds": {"permutation": 41, "bootstrap": 42, "authorship_split": 43},
        "selection_policy": {"selection": selection},
        "matched_design": {
            "candidate_full_brief_sha256": brief_sha,
            "control_full_brief_sha256": brief_sha,
            "sampling_grid_sha256": sampling_grid_sha,
            "seed_grid": seed_grid,
            "control_output_manifest_sha256": output_sha,
        },
        "artifact_bindings": {
            name: {"path": path, "sha256": hashes[hash_fields[name]]}
            for name, path in paths.items()
        },
        "operator_approval": {
            "approved": True,
            "reviewer": "independent-operator",
            "reviewed_at": "2026-07-16T00:00:00Z",
        },
    }
    sign(protocol, private_key)
    return protocol, private_key, trusted


def report_for(protocol: dict, *, eligible: bool) -> dict:
    hashes = protocol["hashes"]
    matched = protocol["matched_design"]
    report_hashes = {
        "protocol_sha256": sha_text(json.dumps(protocol, sort_keys=True, separators=(",", ":"))),
        "prompt_panel_sha256": hashes["prompt_panel_sha256"],
        "human_eval_sha256": protocol["panels"]["human_eval"]["content_manifest_sha256"],
        "human_floor_a_sha256": protocol["panels"]["human_floor_a"]["content_manifest_sha256"],
        "human_floor_b_sha256": protocol["panels"]["human_floor_b"]["content_manifest_sha256"],
        "bandwidths_sha256": hashes["bandwidths_sha256"],
        "power_plan_sha256": hashes["power_plan_sha256"],
        "calibration_sha256": hashes["calibration_sha256"],
        "matched_baseline_sha256": hashes["matched_baseline_sha256"],
        "selection_policy_sha256": hashes["selection_policy_sha256"],
        "candidate_full_brief_sha256": matched["candidate_full_brief_sha256"],
        "control_full_brief_sha256": matched["control_full_brief_sha256"],
        "sampling_grid_sha256": matched["sampling_grid_sha256"],
        "control_output_manifest_sha256": matched["control_output_manifest_sha256"],
        "dependency_lock_sha256": hashes["dependency_lock_sha256"],
        "evaluator_commit_sha256": hashes["metric_code_sha256"],
    }
    return {
        "artifact_schema": "dftr.measurement.report.v2",
        "evidence_class": "prospective_screen",
        "counts": {
            "documents_per_cell": 64,
            "human_documents_per_panel": 64,
            "effective_prompt_clusters": 64,
        },
        "hashes": report_hashes,
        "seeds": {"training": [29], "sampling": [101], "cells": matched["seed_grid"]},
        "checkpoint_manifest": {"selection": {"rule_type": "fixed_seed", "seed": 29}},
        "distribution": {
            "documents_per_cell": 64,
            "human_documents_per_panel": 64,
            "bandwidth_sha256": hashes["bandwidths_sha256"],
            "candidate_mmd2_unbiased": 0.01,
            "control_mmd2_unbiased": 0.02,
            "human_floor_mmd2_unbiased": -0.001,
            "permutation_seed": 41,
            "decision": "pass",
            "power_plan_passed": True,
        },
        "quality": {"status": "measured", "decision": "pass"},
        "quality_linkage": {"status": "verified", "matched_pairs": 64},
        "repetition": {
            "status": "ready",
            "decision": "pass",
            "documents_per_panel": 64,
            "power_plan_passed": True,
        },
        "authorship": {
            "status": "ready",
            "decision": "pass",
            "grouped": True,
            "effective_clusters": 64,
            "fit_count": 320,
        },
        "hard_gates": {"placeholder": True},
        "promotion": {"eligible": eligible},
    }


def signed_blind_manifest(protocol: dict, private_key: Ed25519PrivateKey) -> dict:
    manifest = {
        "artifact_schema": "dftr.measurement.blind_test_manifest.v2",
        "status": "qualified",
        "protocol_sha256": sha_text(json.dumps(protocol, sort_keys=True, separators=(",", ":"))),
        "evaluator_commit": protocol["hashes"]["metric_code_sha256"],
        "dependency_lock_sha256": protocol["hashes"]["dependency_lock_sha256"],
        "fixture_pack_sha256": sha_text("fixture pack"),
        "runtime_versions": {"python": "tester"},
        "tested_at": "2026-07-17T00:00:00Z",
        "signer_identity": "independent-operator",
        "tests": [{"name": name, "status": "pass"} for name in REQUIRED_BLIND_GROUPS],
        "no_sealed_imitation": True,
    }
    sign(manifest, private_key)
    return manifest


@pytest.mark.xfail(strict=True, reason="one output row satisfies a baseline that claims 64 documents")
def test_protocol_parses_matched_baseline_output_cardinality(tmp_path: Path) -> None:
    protocol, _private_key, trusted = bound_protocol(tmp_path, baseline_row_count=1)
    assert protocol_readiness(
        protocol, artifact_root=tmp_path, trusted_public_keys=trusted
    )["status"] == "fail_closed"


@pytest.mark.xfail(strict=True, reason="per-item human hashes are not checked against bound content bytes")
def test_protocol_verifies_human_content_fingerprints(tmp_path: Path) -> None:
    protocol, _private_key, trusted = bound_protocol(
        tmp_path, truthful_human_fingerprints=False
    )
    assert protocol_readiness(
        protocol, artifact_root=tmp_path, trusted_public_keys=trusted
    )["status"] == "fail_closed"


@pytest.mark.xfail(strict=True, reason="attestation trusts an unstructured inventory status boolean")
def test_attestation_requires_verified_bound_inventory_evidence(tmp_path: Path) -> None:
    protocol, private_key, trusted = bound_protocol(tmp_path)
    manifest = signed_blind_manifest(protocol, private_key)
    with pytest.raises(MeasurementV2Error, match="inventory"):
        build_attestation(
            protocol=protocol,
            inventory_check={"status": "pass"},
            blind_test_manifest=manifest,
            operator="independent-operator",
            attested_at="2026-07-17T00:00:00Z",
            artifact_root=tmp_path,
            trusted_public_keys=trusted,
        )


@pytest.mark.xfail(strict=True, reason="six claimed rates pass without simulation trials, effects, or cluster design")
def test_power_artifact_requires_a_prospective_simulation_contract(tmp_path: Path) -> None:
    protocol, _private_key, trusted = bound_protocol(tmp_path)
    assert protocol_readiness(
        protocol, artifact_root=tmp_path, trusted_public_keys=trusted
    )["status"] == "fail_closed"


@pytest.mark.xfail(strict=True, reason="an arbitrary placeholder is accepted as the complete hard-gate set")
def test_promotion_requires_named_frozen_hard_gates(tmp_path: Path) -> None:
    protocol, _private_key, trusted = bound_protocol(tmp_path)
    report = report_for(protocol, eligible=True)
    with pytest.raises(MeasurementV2Error, match="hard gate|intersection"):
        validate_report_v2(
            report,
            protocol=protocol,
            artifact_root=tmp_path,
            trusted_public_keys=trusted,
        )


@pytest.mark.xfail(strict=True, reason="promotion reports are content-bound but not signed")
def test_promotion_report_requires_a_verifiable_signature(tmp_path: Path) -> None:
    protocol, _private_key, trusted = bound_protocol(tmp_path)
    report = report_for(protocol, eligible=True)
    with pytest.raises(MeasurementV2Error, match="signature"):
        validate_report_v2(
            report,
            protocol=protocol,
            artifact_root=tmp_path,
            trusted_public_keys=trusted,
        )
