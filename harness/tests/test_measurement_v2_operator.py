from __future__ import annotations

import hashlib
import json
from pathlib import Path
import base64

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import harness.measurement_v2_operator as operator_module

from harness.measurement_v2 import REQUIRED_HARD_GATE_SCHEMAS, protocol_readiness
from harness.measurement_v2_operator import (
    DECISION_SCHEMA,
    EMBEDDING_SCHEMA,
    POWER_ASSUMPTIONS_SCHEMA,
    MeasurementV2Error,
    _load_hard_gate_sources,
    attest_operator_bundle,
    freeze_operator_bundle,
    generate_operator_key,
    score_candidate_bundle,
)


TEST_RECEIPT_PRIVATE = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
operator_module.WRAPPER_RECEIPT_PUBLIC_KEY_BASE64 = base64.b64encode(
    TEST_RECEIPT_PRIVATE.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
).decode("ascii")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_generation_manifest(
    path: Path,
    outputs: Path,
    config: Path,
    ledger: Path,
    *,
    arm: str,
    checkpoint: str,
) -> Path:
    config_value = json.loads(config.read_text())
    config_sha = hashlib.sha256(
        json.dumps(config_value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_id = "dftr-1784280000-deadbeef" if arm == "A0" else "dftr-1784280001-feedface"
    git_sha = "a" * 40
    comparison = "M2-score-function-MMD-A0-vs-A64-v1"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "run",
                    "run_id": run_id,
                    "config_hash": config_sha,
                    "git_sha": git_sha,
                    "comparison": comparison,
                },
                sort_keys=True,
            )
            + "\n"
        )
    write_json(
        path,
        {
            "artifact_schema": "dftr.m2.adapter_native_generation.v1",
            "status": "completed",
            "run_id": run_id,
            "git_sha": git_sha,
            "config_sha256": config_sha,
            "comparison_id": comparison,
            "arm": arm,
            "adapter_native": True,
            "checkpoint_sha256": checkpoint,
            "generation_contract_sha256": sha("generation-contract"),
            "decoding_policy_sha256": sha("decoding-policy"),
            "documents": 64,
            "generated_tokens_per_document": 64,
            "output_path": f"/checkpoints/runs/{run_id}/outputs.jsonl",
            "output_sha256": hashlib.sha256(outputs.read_bytes()).hexdigest(),
            "token_accounting": {"total_tokens": 4096},
        },
    )
    receipt = {
        "artifact_schema": "dftr.wrapper.generation_receipt.v1",
        "status": "completed",
        "key_id": "humanwrite-modal-wrapper-receipt-v1",
        "run_id": run_id,
        "comparison_id": comparison,
        "config_sha256": config_sha,
        "git_sha": git_sha,
        "manifest_path": f"/checkpoints/runs/{run_id}/run_manifest.json",
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_path": f"/checkpoints/runs/{run_id}/outputs.jsonl",
        "output_sha256": hashlib.sha256(outputs.read_bytes()).hexdigest(),
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["signature"] = {
        "algorithm": "ed25519",
        "signed_payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "signature_base64": base64.b64encode(TEST_RECEIPT_PRIVATE.sign(canonical)).decode(),
    }
    receipt_path = path.with_name(path.stem + "-wrapper-receipt.json")
    write_json(receipt_path, receipt)
    return receipt_path


def source_inputs(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    humans = [
        {
            "document_id": f"human-{index:03d}",
            "text": f"Visible human document {index}. This is unique public reference prose.",
            "eligible": True,
            "eligibility_basis": "public-fineweb-visible-v2",
            "exclusion_flags": [],
            "split": "train" if index < 128 else "dev",
        }
        for index in range(192)
    ]
    human_path = source_root / "humans.jsonl"
    write_jsonl(human_path, humans)
    prompts = []
    for index in range(64):
        reference = f"Prompt-matched visible human reference {index}."
        prompts.append(
            {
                "prompt_id": f"human-{index + 128:03d}",
                "full_brief": f"Write the independently frozen brief number {index}.",
                "reference_text": reference,
                "reference_fingerprint": sha(reference),
                "split": "quality_visible_human",
            }
        )
    prompt_path = source_root / "prompts.jsonl"
    write_jsonl(prompt_path, prompts)
    control = [
        {
            "prompt_id": row["prompt_id"],
            "training_seed": 11,
            "sampling_seed": 101,
            "text": f"Matched control text {index} with stable words.",
            "checkpoint_sha256": sha("control-checkpoint"),
            "generation_contract_sha256": sha("generation-contract"),
            "decoding_policy_sha256": sha("decoding-policy"),
        }
        for index, row in enumerate(prompts)
    ]
    control_path = source_root / "control.jsonl"
    write_jsonl(control_path, control)
    ledger_path = source_root / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    control_config = source_root / "control-config.json"
    write_json(control_config, {"arm": "A0", "frozen": True})
    control_manifest = source_root / "control-run-manifest.json"
    control_receipt = write_generation_manifest(
        control_manifest,
        control_path,
        control_config,
        ledger_path,
        arm="A0",
        checkpoint=sha("control-checkpoint"),
    )

    # Reproduce the production selection ranking so the supplied embedding
    # bundle covers exactly the selected 192 IDs. With exactly 192 source rows,
    # this is the same ID set regardless of order.
    vectors = []
    for index, row in enumerate(humans):
        angle = index / 17.0
        vectors.append(
            {
                "document_id": row["document_id"],
                "embedding": [
                    float(np.sin(angle)),
                    float(np.cos(angle)),
                    float(index % 7) / 7.0,
                    float(index % 11) / 11.0,
                ],
            }
        )
    preprocessing = {"text": "utf8_input_verbatim", "normalize_embeddings": True}
    embeddings = {
        "artifact_schema": EMBEDDING_SCHEMA,
        "status": "materialized",
        "embedder_id": "independent-dev-embedder-test",
        "embedder_revision": "test-revision-immutable",
        "embedder_sha256": sha("test-model-directory"),
        "preprocessing": preprocessing,
        "preprocessing_sha256": hashlib.sha256(
            json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": vectors,
    }
    embedding_path = source_root / "human-embeddings.json"
    write_json(embedding_path, embeddings)
    assumptions = {
        "artifact_schema": POWER_ASSUMPTIONS_SCHEMA,
        "trials": 1000,
        "seed": 4101,
        "minimally_important_effects": {
            "mmd": 0.03,
            "auc": 0.15,
            "repetition": 0.20,
            "coverage": 0.95,
        },
        "pilot_scales": {
            "mmd_standard_error": 0.01,
            "auc_standard_error": 0.05,
            "human_repetition_rate": 0.10,
            "repetition_margin": 0.15,
        },
        "multiplicity": "holm",
    }
    assumption_path = source_root / "power.json"
    write_json(assumption_path, assumptions)
    decision = {
        "artifact_schema": DECISION_SCHEMA,
        "status": "frozen",
        "evidence_class": "prospective_screen",
        "permutation_draws": 10_000,
        "authorship_uncertainty_refits": 100,
        "authorship_fold_seeds": [701, 702, 703],
        "seeds": {"permutation": 41, "bootstrap": 42, "authorship_split": 43},
        "thresholds": {
            "candidate_minus_control_mmd_max": -0.001,
            "paired_mmd_p_max": 0.05,
            "repetition_noninferiority_margin": 0.15,
            "authorship_separability_improvement_min": 0.01,
            "quality_win_rate_min": 0.50,
        },
    }
    decision_path = source_root / "decision.json"
    write_json(decision_path, decision)
    lock_path, metric_path = source_root / "uv.lock", source_root / "measurement_v2.py"
    lock_path.write_text("locked synthetic dependencies\n")
    metric_path.write_text("# frozen synthetic metric image\n")
    history_path = source_root / "history.txt"
    history_path.write_text("immutable historical bytes\n")
    history_sha = hashlib.sha256(history_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(f"history.txt\0{history_sha}\n".encode()).hexdigest()
    inventory = {
        "artifact_schema": "dftr.measurement.historical_inventory.v1",
        "artifact_sets": [
            {
                "name": "synthetic-history",
                "include_globs": ["history.txt"],
                "file_count": 1,
                "manifest_sha256": manifest_sha,
            }
        ],
    }
    inventory_path = source_root / "inventory.json"
    write_json(inventory_path, inventory)
    private_path, blind_private = (
        tmp_path / "operator-private.json",
        tmp_path / "blind-private.json",
    )
    trust_path = source_root / "trusted.json"
    generate_operator_key(private_path, trust_path, key_id="operator-key")
    generate_operator_key(blind_private, trust_path, key_id="blind-key")
    return {
        "source_root": source_root,
        "humans": human_path,
        "prompts": prompt_path,
        "control": control_path,
        "control_manifest": control_manifest,
        "control_config": control_config,
        "control_receipt": control_receipt,
        "ledger": ledger_path,
        "embeddings": embedding_path,
        "power": assumption_path,
        "decision": decision_path,
        "lock": lock_path,
        "metric": metric_path,
        "inventory": inventory_path,
        "private": private_path,
        "blind_private": blind_private,
        "trust": trust_path,
    }


def freeze(tmp_path: Path):
    inputs = source_inputs(tmp_path)
    root = tmp_path / "artifact"
    result = freeze_operator_bundle(
        artifact_root=root,
        human_source=inputs["humans"],
        prompt_briefs=inputs["prompts"],
        control_outputs=inputs["control"],
        control_generation_manifest=inputs["control_manifest"],
        control_generation_config=inputs["control_config"],
        generation_ledger=inputs["ledger"],
        control_wrapper_receipt=inputs["control_receipt"],
        human_embeddings=inputs["embeddings"],
        power_assumptions=inputs["power"],
        decision_contract=inputs["decision"],
        dependency_lock=inputs["lock"],
        metric_code=inputs["metric"],
        private_key=inputs["private"],
        trusted_keys=inputs["trust"],
        historical_inventory=inputs["inventory"],
        repo_root=inputs["source_root"],
        control_checkpoint_sha256=sha("control-checkpoint"),
        decoding_policy_sha256=sha("decoding-policy"),
        generation_contract_sha256=sha("generation-contract"),
        operator="operator-test",
        reviewed_at="2026-07-17T00:00:00Z",
    )
    return root, inputs, result


def test_real_input_shaped_bundle_freezes_and_invokes_protocol_validator(tmp_path):
    root, _inputs, result = freeze(tmp_path)
    protocol = json.loads((root / "measurement_protocol_v2.json").read_text())
    trusted = json.loads((root / "trusted_operator_keys_v2.json").read_text())
    assert result["status"] == "protocol_ready_blind_attestation_pending"
    assert (
        protocol_readiness(protocol, artifact_root=root, trusted_public_keys=trusted)[
            "status"
        ]
        == "ready"
    )
    assert len(json.loads((root / "human_panels.json").read_text())["panels"]) == 3
    assert len((root / "matched_control_outputs.jsonl").read_text().splitlines()) == 64
    assert (
        json.loads((root / "power_plan.json").read_text())["all_targets_pass"] is True
    )
    assert (
        json.loads((root / "blind_test_manifest_v2.candidate.json").read_text())[
            "status"
        ]
        == "not_run"
    )


def test_missing_humans_and_incomplete_control_grid_fail_closed(tmp_path):
    inputs = source_inputs(tmp_path)
    short = tmp_path / "short-humans.jsonl"
    short.write_text("\n".join(inputs["humans"].read_text().splitlines()[:191]) + "\n")
    with pytest.raises(MeasurementV2Error, match="at least 192"):
        freeze_operator_bundle(
            artifact_root=tmp_path / "short-artifact",
            human_source=short,
            prompt_briefs=inputs["prompts"],
            control_outputs=inputs["control"],
            control_generation_manifest=inputs["control_manifest"],
            control_generation_config=inputs["control_config"],
            generation_ledger=inputs["ledger"],
            control_wrapper_receipt=inputs["control_receipt"],
            human_embeddings=inputs["embeddings"],
            power_assumptions=inputs["power"],
            decision_contract=inputs["decision"],
            dependency_lock=inputs["lock"],
            metric_code=inputs["metric"],
            private_key=inputs["private"],
            trusted_keys=inputs["trust"],
            historical_inventory=inputs["inventory"],
            repo_root=inputs["source_root"],
            control_checkpoint_sha256=sha("control-checkpoint"),
            decoding_policy_sha256=sha("decoding-policy"),
            generation_contract_sha256=sha("generation-contract"),
            operator="operator-test",
            reviewed_at="2026-07-17T00:00:00Z",
        )

    incomplete = tmp_path / "incomplete-control.jsonl"
    incomplete.write_text(
        "\n".join(inputs["control"].read_text().splitlines()[:-1]) + "\n"
    )
    with pytest.raises(MeasurementV2Error, match="exact 64-prompt"):
        # A fresh root is required because materialization never overwrites.
        freeze_operator_bundle(
            artifact_root=tmp_path / "incomplete-artifact",
            human_source=inputs["humans"],
            prompt_briefs=inputs["prompts"],
            control_outputs=incomplete,
            control_generation_manifest=inputs["control_manifest"],
            control_generation_config=inputs["control_config"],
            generation_ledger=inputs["ledger"],
            control_wrapper_receipt=inputs["control_receipt"],
            human_embeddings=inputs["embeddings"],
            power_assumptions=inputs["power"],
            decision_contract=inputs["decision"],
            dependency_lock=inputs["lock"],
            metric_code=inputs["metric"],
            private_key=inputs["private"],
            trusted_keys=inputs["trust"],
            historical_inventory=inputs["inventory"],
            repo_root=inputs["source_root"],
            control_checkpoint_sha256=sha("control-checkpoint"),
            decoding_policy_sha256=sha("decoding-policy"),
            generation_contract_sha256=sha("generation-contract"),
            operator="operator-test",
            reviewed_at="2026-07-17T00:00:00Z",
        )


def test_underpowered_assumptions_write_fail_closed_status(tmp_path):
    inputs = source_inputs(tmp_path)
    assumptions = json.loads(inputs["power"].read_text())
    assumptions["minimally_important_effects"]["mmd"] = 1e-8
    write_json(inputs["power"], assumptions)
    root = tmp_path / "underpowered-artifact"
    with pytest.raises(MeasurementV2Error, match="did not qualify"):
        freeze_operator_bundle(
            artifact_root=root,
            human_source=inputs["humans"],
            prompt_briefs=inputs["prompts"],
            control_outputs=inputs["control"],
            control_generation_manifest=inputs["control_manifest"],
            control_generation_config=inputs["control_config"],
            generation_ledger=inputs["ledger"],
            control_wrapper_receipt=inputs["control_receipt"],
            human_embeddings=inputs["embeddings"],
            power_assumptions=inputs["power"],
            decision_contract=inputs["decision"],
            dependency_lock=inputs["lock"],
            metric_code=inputs["metric"],
            private_key=inputs["private"],
            trusted_keys=inputs["trust"],
            historical_inventory=inputs["inventory"],
            repo_root=inputs["source_root"],
            control_checkpoint_sha256=sha("control-checkpoint"),
            decoding_policy_sha256=sha("decoding-policy"),
            generation_contract_sha256=sha("generation-contract"),
            operator="operator-test",
            reviewed_at="2026-07-17T00:00:00Z",
        )
    assert (
        json.loads((root / "materialization_status.json").read_text())["status"]
        == "fail_closed"
    )
    assert not (root / "measurement_protocol_v2.json").exists()


def test_candidate_blind_template_cannot_attest(tmp_path):
    root, inputs, _result = freeze(tmp_path)
    with pytest.raises(MeasurementV2Error, match="signature"):
        attest_operator_bundle(
            artifact_root=root,
            blind_manifest=root / "blind_test_manifest_v2.candidate.json",
            repo_root=inputs["source_root"],
            operator="operator-test",
            attested_at="2026-07-17T00:00:00Z",
        )


def test_freeze_rejects_relabelled_control_text_not_bound_by_run_manifest(tmp_path):
    inputs = source_inputs(tmp_path)
    forged = tmp_path / "forged-control.jsonl"
    rows = _jsonl(inputs["control"])
    for index, row in enumerate(rows):
        row["text"] = f"Relabelled text that was never generated {index}."
    write_jsonl(forged, rows)
    with pytest.raises(MeasurementV2Error, match="manifest.*output bytes"):
        freeze_operator_bundle(
            artifact_root=tmp_path / "forged-artifact",
            human_source=inputs["humans"],
            prompt_briefs=inputs["prompts"],
            control_outputs=forged,
            control_generation_manifest=inputs["control_manifest"],
            control_generation_config=inputs["control_config"],
            generation_ledger=inputs["ledger"],
            control_wrapper_receipt=inputs["control_receipt"],
            human_embeddings=inputs["embeddings"],
            power_assumptions=inputs["power"],
            decision_contract=inputs["decision"],
            dependency_lock=inputs["lock"],
            metric_code=inputs["metric"],
            private_key=inputs["private"],
            trusted_keys=inputs["trust"],
            historical_inventory=inputs["inventory"],
            repo_root=inputs["source_root"],
            control_checkpoint_sha256=sha("control-checkpoint"),
            decoding_policy_sha256=sha("decoding-policy"),
            generation_contract_sha256=sha("generation-contract"),
            operator="operator-test",
            reviewed_at="2026-07-17T00:00:00Z",
        )


def test_score_rejects_candidate_grid_before_any_metric_work(tmp_path):
    root, inputs, _result = freeze(tmp_path)
    candidate = tmp_path / "candidate.jsonl"
    rows = json.loads("[" + ",".join(inputs["control"].read_text().splitlines()) + "]")[
        :-1
    ]
    for row in rows:
        row["checkpoint_sha256"] = sha("candidate-checkpoint")
    write_jsonl(candidate, rows)
    candidate_manifest = tmp_path / "candidate-run-manifest.json"
    candidate_config = tmp_path / "candidate-config.json"
    write_json(candidate_config, {"arm": "A64", "frozen": True})
    candidate_receipt = write_generation_manifest(
        candidate_manifest,
        candidate,
        candidate_config,
        inputs["ledger"],
        arm="A64",
        checkpoint=sha("candidate-checkpoint"),
    )
    score_embeddings = tmp_path / "unused-score-embeddings.json"
    write_json(score_embeddings, {"status": "unused"})
    with pytest.raises(MeasurementV2Error, match="exact 64-prompt"):
        score_candidate_bundle(
            artifact_root=root,
            candidate_outputs=candidate,
            candidate_generation_manifest=candidate_manifest,
            candidate_generation_config=candidate_config,
            generation_ledger=inputs["ledger"],
            candidate_wrapper_receipt=candidate_receipt,
            score_embeddings=score_embeddings,
            candidate_checkpoint_sha256=sha("candidate-checkpoint"),
            private_key=inputs["private"],
        )


def test_hard_gate_manifest_requires_exact_precomputed_evidence(tmp_path):
    bare = tmp_path / "bare-decisions.json"
    write_json(bare, {name: "pass" for name in REQUIRED_HARD_GATE_SCHEMAS})
    with pytest.raises(MeasurementV2Error, match="evidence path"):
        _load_hard_gate_sources(bare)

    sources = {}
    for name, schema in REQUIRED_HARD_GATE_SCHEMAS.items():
        evidence = tmp_path / f"observed-{name}.json"
        write_json(
            evidence,
            {"artifact_schema": schema, "name": name, "decision": "pass"},
        )
        sources[name] = str(evidence)
    manifest = tmp_path / "gate-evidence-manifest.json"
    write_json(manifest, sources)
    assert set(_load_hard_gate_sources(manifest)) == set(REQUIRED_HARD_GATE_SCHEMAS)


def test_complete_candidate_grid_emits_valid_nonpromoting_report(tmp_path, monkeypatch):
    root, inputs, _result = freeze(tmp_path)
    candidate = tmp_path / "candidate.jsonl"
    candidate_rows = []
    for index, row in enumerate(_jsonl(inputs["control"])):
        candidate_rows.append(
            {
                **row,
                "text": f"Candidate text {index} with stable prose.",
                "checkpoint_sha256": sha("candidate-checkpoint"),
            }
        )
    write_jsonl(candidate, candidate_rows)
    candidate_manifest = tmp_path / "candidate-run-manifest.json"
    candidate_config = tmp_path / "candidate-config.json"
    write_json(candidate_config, {"arm": "A64", "frozen": True})
    candidate_receipt = write_generation_manifest(
        candidate_manifest,
        candidate,
        candidate_config,
        inputs["ledger"],
        arm="A64",
        checkpoint=sha("candidate-checkpoint"),
    )
    human_meta = json.loads(inputs["embeddings"].read_text())
    score_vectors = []
    for index, row in enumerate(candidate_rows):
        vector = [float(index) / 64.0, 0.5, 0.25, 0.125]
        score_vectors.append(
            {"document_id": f"candidate:{row['prompt_id']}", "embedding": vector}
        )
        score_vectors.append(
            {"document_id": f"control:{row['prompt_id']}", "embedding": vector}
        )
    score_bundle = {
        key: human_meta[key]
        for key in (
            "artifact_schema",
            "status",
            "embedder_id",
            "embedder_revision",
            "embedder_sha256",
            "preprocessing",
            "preprocessing_sha256",
        )
    }
    score_bundle["rows"] = score_vectors
    score_embeddings = tmp_path / "score-embeddings.json"
    write_json(score_embeddings, score_bundle)

    import harness.measurement_v2_operator as operator

    monkeypatch.setattr(
        operator,
        "common_kernel_report",
        lambda *args, **kwargs: {
            "documents_per_cell": 64,
            "human_documents_per_panel": 64,
            "bandwidth_sha256": sha("value-bandwidth"),
            "candidate_mmd2_unbiased": 0.02,
            "control_mmd2_unbiased": 0.02,
            "human_floor_mmd2_unbiased": 0.0,
            "candidate_minus_control": 0.0,
            "paired_candidate_control_p": 1.0,
        },
    )
    monkeypatch.setattr(
        operator,
        "grouped_authorship_auc",
        lambda *args, **kwargs: {
            "status": "ready",
            "auc": 0.7,
            "separability": 0.2,
            "interval": {"low": 0.6, "high": 0.8},
            "effective_clusters": 64,
            "fit_count": 10,
        },
    )
    result = score_candidate_bundle(
        artifact_root=root,
        candidate_outputs=candidate,
        candidate_generation_manifest=candidate_manifest,
        candidate_generation_config=candidate_config,
        generation_ledger=inputs["ledger"],
        candidate_wrapper_receipt=candidate_receipt,
        score_embeddings=score_embeddings,
        candidate_checkpoint_sha256=sha("candidate-checkpoint"),
        private_key=inputs["private"],
    )
    report = json.loads((root / "measurement_report_v2.json").read_text())
    assert result["status"] == "pass"
    assert result["promotion_eligible"] is False
    assert report["promotion"] == {"eligible": False}
    assert report["quality"]["status"] == "not_measured"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
