import base64
import hashlib
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from harness.measurement_v2 import (
    REQUIRED_BLIND_GROUPS,
    build_attestation,
    protocol_readiness,
    validate_report_v2,
)
from harness.metrics.distribution_v2 import bandwidth_hash


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def write_json(root, name, value):
    path = root / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(document, private_key, field="operator_signature", key_id="operator-key"):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document[field] = {
        "algorithm": "ed25519",
        "key_id": key_id,
        "signed_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signature_base64": base64.b64encode(private_key.sign(payload)).decode(),
    }


def synthetic_evidence(root):
    private_key = Ed25519PrivateKey.generate()
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trusted = {"operator-key": base64.b64encode(public_raw).decode()}
    (root / "uv.lock").write_text("synthetic locked dependencies\n")
    (root / "evaluator.py").write_text("# synthetic evaluator image\n")
    dependency_sha = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    metric_sha = hashlib.sha256((root / "evaluator.py").read_bytes()).hexdigest()
    (root / "prompt-brief.txt").write_text("complete prompt brief bytes")
    brief_sha = hashlib.sha256((root / "prompt-brief.txt").read_bytes()).hexdigest()
    prompt_ids = [f"prompt-{index}" for index in range(64)]
    prompt_sha = write_json(root, "prompts.json", {
        "artifact_schema": "dftr.measurement.prompt_panel.v2",
        "status": "frozen", "frozen": True,
        "prompt_ids": prompt_ids, "full_brief_sha256": brief_sha,
    })
    protocol_panels = {}
    manifest_panels = {}
    for name, prefix in (("human_eval", "e"), ("human_floor_a", "a"), ("human_floor_b", "b")):
        ids = [f"{prefix}-{index}" for index in range(64)]
        protocol_panels[name] = {
            "status": "materialized", "frozen": True,
            "document_count": 64, "document_ids": ids,
        }
        manifest_panels[name] = [
            {"document_id": document_id, "content_sha256": sha(f"content:{document_id}")}
            for document_id in ids
        ]
        protocol_panels[name]["content_manifest_sha256"] = sha(
            json.dumps(manifest_panels[name], sort_keys=True, separators=(",", ":"))
        )
    (root / "human-contents.bin").write_bytes(
        "\n".join(f"synthetic content:{row['document_id']}" for rows in manifest_panels.values() for row in rows).encode()
    )
    human_contents_sha = hashlib.sha256((root / "human-contents.bin").read_bytes()).hexdigest()
    human_sha = write_json(root, "humans.json", {
        "artifact_schema": "dftr.measurement.human_panels.v2",
        "status": "materialized", "frozen": True,
        "target_n_per_panel": 64, "sampling": "disjoint_without_replacement",
        "eligibility_attestation_sha256": sha("eligibility"),
        "content_bundle_sha256": human_contents_sha,
        "panels": manifest_panels,
    })
    bandwidth_values = [0.5, 1.0]
    bandwidth_sha = write_json(root, "bandwidths.json", {
        "artifact_schema": "dftr.measurement.bandwidths.v2",
        "status": "frozen", "frozen": True,
        "source": "human_floor_a_union_human_floor_b_only",
        "values": bandwidth_values,
        "panel_manifest_sha256": human_sha,
        "floor_a_content_manifest_sha256": protocol_panels["human_floor_a"]["content_manifest_sha256"],
        "floor_b_content_manifest_sha256": protocol_panels["human_floor_b"]["content_manifest_sha256"],
        "embedder_sha256": sha("embedder-revision"),
        "preprocessing_sha256": sha("preprocessing-contract"),
        "bandwidth_sha256": bandwidth_hash(bandwidth_values),
    })
    sampling_grid_sha = sha("sampling-grid")
    (root / "control-outputs.jsonl").write_text("{\"synthetic\": true}\n")
    output_manifest_sha = hashlib.sha256((root / "control-outputs.jsonl").read_bytes()).hexdigest()
    seed_grid = [{"training_seed": 29, "sampling_seeds": [101]}]
    baseline_sha = write_json(root, "baseline.json", {
        "artifact_schema": "dftr.measurement.matched_sft_baseline.v2",
        "status": "materialized", "frozen": True,
        "documents_per_cell": 64,
        "prompt_panel_sha256": prompt_sha,
        "full_brief_sha256": brief_sha,
        "sampling_grid_sha256": sampling_grid_sha,
        "seed_grid": seed_grid,
        "output_manifest_sha256": output_manifest_sha,
    })
    calibration_sha = write_json(root, "calibration.json", {
        "artifact_schema": "dftr.measurement.calibration.v2",
        "status": "frozen", "frozen": True, "documents_per_cell": 64,
        "human_panels_sha256": human_sha,
        "bandwidths_sha256": bandwidth_sha,
        "matched_baseline_sha256": baseline_sha,
        "dependency_lock_sha256": dependency_sha,
    })
    power_sha = write_json(root, "power.json", {
        "artifact_schema": "dftr.measurement.power_plan.v2",
        "status": "frozen", "frozen": True, "all_targets_pass": True,
        "documents_per_cell": 64,
        "human_panels_sha256": human_sha,
        "bandwidths_sha256": bandwidth_sha,
        "calibration_sha256": calibration_sha,
        "matched_baseline_sha256": baseline_sha,
        "dependency_lock_sha256": dependency_sha,
        "results": {
            "mmd_type_i_rate": 0.049, "mmd_type_i_max": 0.05,
            "mmd_power": 0.81, "auc_power": 0.82,
            "repetition_power": 0.83, "coverage": 0.95,
        },
        "multiplicity": {"method": "holm"},
    })
    selection = {"rule_type": "fixed_seed", "seed": 29}
    selection_sha = write_json(root, "selection.json", {
        "artifact_schema": "dftr.measurement.selection_policy.v2",
        "status": "frozen", "frozen": True, "selection": selection,
    })
    hashes = {
        "dependency_lock_sha256": dependency_sha,
        "metric_code_sha256": metric_sha,
        "prompt_panel_sha256": prompt_sha,
        "prompt_brief_sha256": brief_sha,
        "human_panels_sha256": human_sha,
        "human_panel_contents_sha256": human_contents_sha,
        "bandwidths_sha256": bandwidth_sha,
        "power_plan_sha256": power_sha,
        "calibration_sha256": calibration_sha,
        "matched_baseline_sha256": baseline_sha,
        "matched_baseline_outputs_sha256": output_manifest_sha,
        "selection_policy_sha256": selection_sha,
    }
    paths = {
        "dependency_lock": "uv.lock", "metric_code": "evaluator.py",
        "prompt_panel": "prompts.json", "prompt_brief": "prompt-brief.txt",
        "human_panels": "humans.json", "human_panel_contents": "human-contents.bin",
        "bandwidths": "bandwidths.json", "power_plan": "power.json",
        "calibration": "calibration.json", "matched_baseline": "baseline.json",
        "matched_baseline_outputs": "control-outputs.jsonl",
        "selection_policy": "selection.json",
    }
    hash_fields = {
        "dependency_lock": "dependency_lock_sha256", "metric_code": "metric_code_sha256",
        "prompt_panel": "prompt_panel_sha256", "human_panels": "human_panels_sha256",
        "prompt_brief": "prompt_brief_sha256", "human_panel_contents": "human_panel_contents_sha256",
        "bandwidths": "bandwidths_sha256", "power_plan": "power_plan_sha256",
        "calibration": "calibration_sha256", "matched_baseline": "matched_baseline_sha256",
        "matched_baseline_outputs": "matched_baseline_outputs_sha256",
        "selection_policy": "selection_policy_sha256",
    }
    protocol = {
        "artifact_schema": "dftr.measurement.protocol.v2", "status": "ready", "frozen": True,
        "design": {"documents_per_cell": 64, "human_pool_documents": 192, "replacement_sampling": False},
        "hashes": hashes, "panels": protocol_panels,
        "bandwidth_contract": {"status": "frozen", "source": "human_floor_a_union_human_floor_b", "values": bandwidth_values, "bandwidth_value_sha256": bandwidth_hash(bandwidth_values)},
        "power": {field: "pass" for field in ("mmd_type_i", "mmd_power", "auc_power", "repetition_power", "coverage")},
        "seeds": {"permutation": 41, "bootstrap": 42, "authorship_split": 43},
        "selection_policy": {"selection": selection},
        "matched_design": {
            "candidate_full_brief_sha256": brief_sha,
            "control_full_brief_sha256": brief_sha,
            "sampling_grid_sha256": sampling_grid_sha,
            "seed_grid": seed_grid,
            "control_output_manifest_sha256": output_manifest_sha,
        },
        "artifact_bindings": {
            name: {"path": path, "sha256": hashes[hash_fields[name]]}
            for name, path in paths.items()
        },
        "operator_approval": {"approved": True, "reviewer": "independent-operator", "reviewed_at": "2026-07-16T00:00:00Z"},
    }
    sign(protocol, private_key)
    return protocol, private_key, trusted


def test_signed_content_addressed_protocol_report_and_attestation(tmp_path):
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    assert protocol_readiness(
        protocol, artifact_root=tmp_path, trusted_public_keys=trusted
    )["status"] == "ready"
    hashes = protocol["hashes"]
    report_hashes = {
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "prompt_panel_sha256": hashes["prompt_panel_sha256"],
        "human_eval_sha256": protocol["panels"]["human_eval"]["content_manifest_sha256"],
        "human_floor_a_sha256": protocol["panels"]["human_floor_a"]["content_manifest_sha256"],
        "human_floor_b_sha256": protocol["panels"]["human_floor_b"]["content_manifest_sha256"],
        "bandwidths_sha256": hashes["bandwidths_sha256"],
        "power_plan_sha256": hashes["power_plan_sha256"],
        "calibration_sha256": hashes["calibration_sha256"],
        "matched_baseline_sha256": hashes["matched_baseline_sha256"],
        "selection_policy_sha256": hashes["selection_policy_sha256"],
        "candidate_full_brief_sha256": protocol["matched_design"]["candidate_full_brief_sha256"],
        "control_full_brief_sha256": protocol["matched_design"]["control_full_brief_sha256"],
        "sampling_grid_sha256": protocol["matched_design"]["sampling_grid_sha256"],
        "control_output_manifest_sha256": protocol["matched_design"]["control_output_manifest_sha256"],
        "dependency_lock_sha256": hashes["dependency_lock_sha256"],
        "evaluator_commit_sha256": hashes["metric_code_sha256"],
    }
    report = {
        "artifact_schema": "dftr.measurement.report.v2", "evidence_class": "prospective_screen",
        "counts": {"documents_per_cell": 64, "human_documents_per_panel": 64, "effective_prompt_clusters": 64},
        "hashes": report_hashes,
        "seeds": {"training": [29], "sampling": [101], "cells": protocol["matched_design"]["seed_grid"]},
        "checkpoint_manifest": {"selection": {"rule_type": "fixed_seed", "seed": 29}},
        "distribution": {"documents_per_cell": 64, "human_documents_per_panel": 64, "bandwidth_sha256": hashes["bandwidths_sha256"], "candidate_mmd2_unbiased": 0.01, "control_mmd2_unbiased": 0.02, "human_floor_mmd2_unbiased": -0.001, "permutation_seed": 41},
        "quality": {"status": "not_measured"},
        "repetition": {"status": "underpowered", "documents_per_panel": 64, "power_plan_passed": False},
        "authorship": {"status": "ready", "grouped": True, "effective_clusters": 64, "fit_count": 105},
        "promotion": {"eligible": False},
    }
    assert validate_report_v2(
        report, protocol=protocol, artifact_root=tmp_path, trusted_public_keys=trusted
    )["status"] == "pass"
    manifest = {
        "artifact_schema": "dftr.measurement.blind_test_manifest.v2",
        "status": "qualified",
        "protocol_sha256": report_hashes["protocol_sha256"],
        "tests": [{"name": name, "status": "pass"} for name in REQUIRED_BLIND_GROUPS],
        "evaluator_commit": hashes["metric_code_sha256"],
        "dependency_lock_sha256": hashes["dependency_lock_sha256"],
        "fixture_pack_sha256": sha("fixtures"),
        "no_sealed_imitation": True,
        "signer_identity": "independent-operator",
        "tested_at": "2026-07-16T00:00:00Z",
        "runtime_versions": {"python": "synthetic-test"},
    }
    sign(manifest, private_key)
    attestation = build_attestation(
        protocol=protocol, inventory_check={"status": "pass"},
        blind_test_manifest=manifest, operator="independent-operator",
        attested_at="2026-07-16T00:00:00Z", artifact_root=tmp_path,
        trusted_public_keys=trusted,
    )
    assert attestation["status"] == "qualified"
    assert attestation["signature_verification"]["status"] == "verified"


def test_bound_protocol_detects_artifact_byte_tampering(tmp_path):
    protocol, _private_key, trusted = synthetic_evidence(tmp_path)
    (tmp_path / "uv.lock").write_text("tampered dependency set\n")
    readiness = protocol_readiness(
        protocol, artifact_root=tmp_path, trusted_public_keys=trusted
    )
    assert readiness["status"] == "fail_closed"
    assert any("hash mismatch" in reason for reason in readiness["reasons"])
