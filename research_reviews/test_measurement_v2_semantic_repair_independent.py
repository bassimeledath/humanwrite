"""Fresh black-box attacks on the measurement-v2 semantic repair."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "harness" / "src"))

from harness.measurement_v2 import (  # noqa: E402
    REQUIRED_BLIND_GROUPS,
    build_attestation,
    protocol_readiness,
    validate_report_v2,
    verify_historical_inventory,
)
from harness.metrics.distribution_v2 import MeasurementV2Error  # noqa: E402


IMPLEMENTATION_FIXTURES = runpy.run_path(
    str(REPO_ROOT / "harness" / "tests" / "test_measurement_v2_bindings.py")
)
synthetic_evidence = IMPLEMENTATION_FIXTURES["synthetic_evidence"]
sign = IMPLEMENTATION_FIXTURES["sign"]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: object) -> str:
    return sha_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def load_json(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text())


def write_json(root: Path, name: str, value: dict) -> str:
    path = root / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return sha_bytes(path.read_bytes())


def load_jsonl(root: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (root / name).read_text().splitlines()]


def write_jsonl(root: Path, name: str, rows: list[dict]) -> str:
    path = root / name
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    return sha_bytes(path.read_bytes())


def resign(document: dict, private_key) -> None:
    document.pop("operator_signature", None)
    sign(document, private_key)


def refresh_protocol_chain(root: Path, protocol: dict, private_key) -> None:
    """Rebind changed bytes through every dependent artifact and signature."""
    dependency_sha = sha_bytes((root / "uv.lock").read_bytes())
    metric_sha = sha_bytes((root / "evaluator.py").read_bytes())
    prompt_sha = sha_bytes((root / "prompts.json").read_bytes())
    brief_sha = sha_bytes((root / "prompt-brief.txt").read_bytes())
    content_sha = sha_bytes((root / "human-contents.jsonl").read_bytes())

    humans = load_json(root, "humans.json")
    humans["content_bundle_sha256"] = content_sha
    human_sha = write_json(root, "humans.json", humans)

    bandwidths = load_json(root, "bandwidths.json")
    bandwidths["panel_manifest_sha256"] = human_sha
    bandwidth_sha = write_json(root, "bandwidths.json", bandwidths)

    output_sha = sha_bytes((root / "control-outputs.jsonl").read_bytes())
    baseline = load_json(root, "baseline.json")
    baseline["output_manifest_sha256"] = output_sha
    baseline_sha = write_json(root, "baseline.json", baseline)

    calibration = load_json(root, "calibration.json")
    calibration.update(
        {
            "human_panels_sha256": human_sha,
            "bandwidths_sha256": bandwidth_sha,
            "matched_baseline_sha256": baseline_sha,
            "dependency_lock_sha256": dependency_sha,
        }
    )
    calibration_sha = write_json(root, "calibration.json", calibration)

    power = load_json(root, "power.json")
    power.update(
        {
            "human_panels_sha256": human_sha,
            "bandwidths_sha256": bandwidth_sha,
            "calibration_sha256": calibration_sha,
            "matched_baseline_sha256": baseline_sha,
            "dependency_lock_sha256": dependency_sha,
        }
    )
    power_sha = write_json(root, "power.json", power)
    selection_sha = sha_bytes((root / "selection.json").read_bytes())

    rebound = {
        "dependency_lock": ("dependency_lock_sha256", dependency_sha),
        "metric_code": ("metric_code_sha256", metric_sha),
        "prompt_panel": ("prompt_panel_sha256", prompt_sha),
        "prompt_brief": ("prompt_brief_sha256", brief_sha),
        "human_panels": ("human_panels_sha256", human_sha),
        "human_panel_contents": ("human_panel_contents_sha256", content_sha),
        "bandwidths": ("bandwidths_sha256", bandwidth_sha),
        "power_plan": ("power_plan_sha256", power_sha),
        "calibration": ("calibration_sha256", calibration_sha),
        "matched_baseline": ("matched_baseline_sha256", baseline_sha),
        "matched_baseline_outputs": (
            "matched_baseline_outputs_sha256",
            output_sha,
        ),
        "selection_policy": ("selection_policy_sha256", selection_sha),
    }
    for binding_name, (hash_name, digest) in rebound.items():
        protocol["hashes"][hash_name] = digest
        protocol["artifact_bindings"][binding_name]["sha256"] = digest
    protocol["matched_design"]["control_output_manifest_sha256"] = output_sha
    resign(protocol, private_key)


def assert_protocol_fails(protocol: dict, root: Path, trusted: dict[str, str]) -> None:
    result = protocol_readiness(
        protocol, artifact_root=root, trusted_public_keys=trusted
    )
    assert result["status"] == "fail_closed", result


def test_duplicate_control_cell_cannot_hide_behind_correct_row_count(tmp_path: Path) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    rows = load_jsonl(tmp_path, "control-outputs.jsonl")
    rows[-1]["prompt_id"] = rows[0]["prompt_id"]
    write_jsonl(tmp_path, "control-outputs.jsonl", rows)
    refresh_protocol_chain(tmp_path, protocol, private_key)

    assert len(rows) == 64
    assert_protocol_fails(protocol, tmp_path, trusted)


def test_swapped_human_text_cannot_preserve_per_item_fingerprint_claims(
    tmp_path: Path,
) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    rows = load_jsonl(tmp_path, "human-contents.jsonl")
    rows[0]["text"], rows[1]["text"] = rows[1]["text"], rows[0]["text"]
    write_jsonl(tmp_path, "human-contents.jsonl", rows)
    refresh_protocol_chain(tmp_path, protocol, private_key)

    assert_protocol_fails(protocol, tmp_path, trusted)


@pytest.mark.parametrize(
    "mutation",
    ["reported_rate", "trial_count", "effect", "prompt_clusters"],
)
def test_power_semantics_survive_fully_rehashed_artifacts(
    tmp_path: Path, mutation: str
) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    power = load_json(tmp_path, "power.json")
    if mutation == "reported_rate":
        power["simulation_results"][1]["successes"] = 811
    elif mutation == "trial_count":
        power["simulation_results"][1].update({"trials": 999, "successes": 810})
    elif mutation == "effect":
        power["simulation_results"][1]["effect"] = 0.02
    else:
        power["simulation_contract"]["prompt_clusters"] = 65
    power["trial_manifest_sha256"] = canonical_sha(power["simulation_contract"])
    power["simulation_results_sha256"] = canonical_sha(power["simulation_results"])
    write_json(tmp_path, "power.json", power)
    refresh_protocol_chain(tmp_path, protocol, private_key)

    assert_protocol_fails(protocol, tmp_path, trusted)


def blind_manifest(protocol: dict, private_key) -> dict:
    value = {
        "artifact_schema": "dftr.measurement.blind_test_manifest.v2",
        "status": "qualified",
        "protocol_sha256": canonical_sha(protocol),
        "evaluator_commit": protocol["hashes"]["metric_code_sha256"],
        "dependency_lock_sha256": protocol["hashes"]["dependency_lock_sha256"],
        "fixture_pack_sha256": sha_bytes(b"independent fixture pack"),
        "runtime_versions": {"python": "independent-test"},
        "tested_at": "2026-07-17T00:00:00Z",
        "signer_identity": "independent-operator",
        "tests": [
            {"name": name, "status": "pass"} for name in REQUIRED_BLIND_GROUPS
        ],
        "no_sealed_imitation": True,
    }
    sign(value, private_key)
    return value


def test_signed_inventory_is_rerun_after_repository_bytes_change(tmp_path: Path) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    manifest = blind_manifest(protocol, private_key)
    dependency_sha = sha_bytes((tmp_path / "uv.lock").read_bytes())
    inventory = {
        "artifact_schema": "dftr.measurement.historical_inventory.v1",
        "artifact_sets": [
            {
                "name": "dependency",
                "include_globs": ["uv.lock"],
                "file_count": 1,
                "manifest_sha256": sha_bytes(
                    f"uv.lock\0{dependency_sha}\n".encode()
                ),
            }
        ],
    }
    inventory_check = verify_historical_inventory(inventory, repo_root=tmp_path)
    sign(inventory_check, private_key)
    (tmp_path / "uv.lock").write_text("post-signature replacement\n")

    with pytest.raises(MeasurementV2Error, match="inventory"):
        build_attestation(
            protocol=protocol,
            inventory_check=inventory_check,
            blind_test_manifest=manifest,
            operator="independent-operator",
            attested_at="2026-07-17T00:00:00Z",
            artifact_root=tmp_path,
            trusted_public_keys=trusted,
        )


def build_promotion_report(
    root: Path,
    protocol: dict,
    private_key,
    *,
    placeholder_gate_evidence: bool,
) -> dict:
    hashes = protocol["hashes"]
    matched = protocol["matched_design"]
    output_rows = [
        {"prompt_id": f"prompt-{index}", "text": f"candidate output {index}"}
        for index in range(64)
    ]
    output_sha = write_jsonl(root, "candidate-outputs.jsonl", output_rows)
    report = {
        "artifact_schema": "dftr.measurement.report.v2",
        "evidence_class": "prospective_screen",
        "counts": {
            "documents_per_cell": 64,
            "human_documents_per_panel": 64,
            "effective_prompt_clusters": 64,
        },
        "hashes": {
            "protocol_sha256": canonical_sha(protocol),
            "prompt_panel_sha256": hashes["prompt_panel_sha256"],
            "human_eval_sha256": protocol["panels"]["human_eval"][
                "content_manifest_sha256"
            ],
            "human_floor_a_sha256": protocol["panels"]["human_floor_a"][
                "content_manifest_sha256"
            ],
            "human_floor_b_sha256": protocol["panels"]["human_floor_b"][
                "content_manifest_sha256"
            ],
            "bandwidths_sha256": hashes["bandwidths_sha256"],
            "power_plan_sha256": hashes["power_plan_sha256"],
            "calibration_sha256": hashes["calibration_sha256"],
            "matched_baseline_sha256": hashes["matched_baseline_sha256"],
            "selection_policy_sha256": hashes["selection_policy_sha256"],
            "candidate_full_brief_sha256": matched[
                "candidate_full_brief_sha256"
            ],
            "control_full_brief_sha256": matched["control_full_brief_sha256"],
            "sampling_grid_sha256": matched["sampling_grid_sha256"],
            "control_output_manifest_sha256": matched[
                "control_output_manifest_sha256"
            ],
            "candidate_output_manifest_sha256": output_sha,
            "dependency_lock_sha256": hashes["dependency_lock_sha256"],
            "evaluator_commit_sha256": hashes["metric_code_sha256"],
        },
        "seeds": {
            "training": [29],
            "sampling": [101],
            "cells": matched["seed_grid"],
        },
        "checkpoint_manifest": {
            "selection": {"rule_type": "fixed_seed", "seed": 29}
        },
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
        "hard_gates": {},
        "candidate_output_binding": {
            "path": "candidate-outputs.jsonl",
            "sha256": output_sha,
        },
        "promotion": {"eligible": True},
    }
    if placeholder_gate_evidence:
        placeholder_path = root / "gate-placeholder.txt"
        placeholder_path.write_text("pass\n")
        evidence = (placeholder_path.name, sha_bytes(placeholder_path.read_bytes()))
        for name, version in protocol["required_hard_gates"].items():
            report["hard_gates"][name] = {
                "version": version,
                "decision": "pass",
                "evidence_path": evidence[0],
                "evidence_sha256": evidence[1],
            }
    else:
        for name, version in protocol["required_hard_gates"].items():
            path = root / f"gate-{name}.json"
            path.write_text(
                json.dumps(
                    {
                        "artifact_schema": version,
                        "name": name,
                        "decision": "pass",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            report["hard_gates"][name] = {
                "version": version,
                "decision": "pass",
                "evidence_path": path.name,
                "evidence_sha256": sha_bytes(path.read_bytes()),
            }
    sign(report, private_key)
    return report


def test_shared_plaintext_placeholder_cannot_satisfy_all_hard_gates(
    tmp_path: Path,
) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=True
    )
    with pytest.raises(MeasurementV2Error, match="hard gate"):
        validate_report_v2(
            report,
            protocol=protocol,
            artifact_root=tmp_path,
            trusted_public_keys=trusted,
        )


def test_signed_candidate_output_binding_detects_post_signature_byte_swap(
    tmp_path: Path,
) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=False
    )
    assert validate_report_v2(
        report,
        protocol=protocol,
        artifact_root=tmp_path,
        trusted_public_keys=trusted,
    )["status"] == "pass"
    (tmp_path / "candidate-outputs.jsonl").write_text("replaced after signing\n")

    with pytest.raises(MeasurementV2Error, match="candidate output"):
        validate_report_v2(
            report,
            protocol=protocol,
            artifact_root=tmp_path,
            trusted_public_keys=trusted,
        )
