from __future__ import annotations

import ast
import base64
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HARNESS_SRC = Path(__file__).resolve().parents[2] / "harness" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from experiments import runner
from experiments.m2 import dft as dft_module
from experiments.m2.dft import (
    BASE_MODEL,
    BASE_REVISION,
    DFT_SCHEMA,
    DFT_STEP,
    FULL_BRIEF_SCHEMA,
    FULL_BRIEF_SERIALIZER_SHA256,
    M2ConfigError,
    _render_prompt,
    _sample_raw_policy,
    _save_training_checkpoint,
    _sequence_log_probs,
    _verify_a64_readiness,
    _verify_resume_artifact,
    _verify_inputs,
    canonical_hash,
    deterministic_batches,
    deterministic_schedule,
    method_contract_payload,
    matched_exposure_payload,
    mmd_leave_one_out_baselines,
    mmd_score_rewards,
    per_sample_score_loss,
    repeated_ngram_fraction,
    run_dft,
    score_function_loss,
    training_factual_adherence_sentinels,
    validate_dft_config,
)
from experiments.m2.prepare_dft import preparation_contract_payload
from experiments.m2.representation import (
    TRAINING_BANDWIDTH_DERIVATION,
    representation_execution_payload,
)
from experiments.m1.contracts import file_sha256
from experiments.m1.workflow import _render_prompt as render_m1_prompt
from infra.backend.policy import PolicyError, canonical_hash as policy_hash, validate_launch
from harness.measurement_v2 import REQUIRED_BLIND_GROUPS


SHA = "a" * 64


def _measurement_fixture_module():
    path = Path(__file__).resolve().parents[2] / "harness" / "tests" / "test_measurement_v2_bindings.py"
    spec = importlib.util.spec_from_file_location("dft_measurement_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_config() -> dict:
    config = {
        "artifact_schema": DFT_SCHEMA,
        "run": {
            "comparison_id": "M2-score-function-MMD-A0-vs-A64-v1",
            "arm": "A0-vs-A64",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 11,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 120},
        "model": {"base": BASE_MODEL, "revision": BASE_REVISION, "torch_dtype": "bfloat16"},
        "initial_adapter": {
            "path": "/checkpoints/runs/source/seed-11",
            "adapter_model_sha256": SHA,
            "adapter_config_sha256": SHA,
            "file_manifest_sha256": SHA,
        },
        "data": {
            "rollout_path": "/checkpoints/data/train_rollouts.jsonl",
            "rollout_sha256": SHA,
            "sft_anchor_path": "/checkpoints/data/train_anchor.jsonl",
            "sft_anchor_sha256": SHA,
            "human_targets_path": "/checkpoints/data/train_humans.jsonl",
            "human_targets_sha256": SHA,
            "completion_field": "completion",
            "human_text_field": "completion",
            "prompt_format": "USER:\n{brief}\nASSISTANT:",
            "prompt_schema_version": FULL_BRIEF_SCHEMA,
            "prompt_serializer_sha256": FULL_BRIEF_SERIALIZER_SHA256,
        },
        "representation": {
            "model": BASE_MODEL,
            "revision": BASE_REVISION,
            "layer": -1,
            "pooling": "attention_masked_mean",
            "normalize": True,
            "role": "training_only_not_measurement_v2",
            "batch_size": 4,
            "max_tokens": 256,
        },
        "kernel": {
            "bandwidths_path": "/checkpoints/data/training_bandwidths.json",
            "bandwidths_sha256": SHA,
            "source": "training_humans_only",
        },
        "runtime": {
            "torch_version": "2.9.1",
            "transformers_version": "4.57.6",
            "peft_version": "0.18.0",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "training": {
            "steps": 8,
            "rollout_batch_size": 4,
            "sft_batch_size": 2,
            "learning_rate": 1e-5,
            "max_input_tokens": 1024,
            "generated_tokens": 64,
            "sampling_distribution": "raw_policy_categorical",
            "kl_coefficient": 0.01,
            "sft_coefficient": 0.1,
            "gradient_clip_norm": 1.0,
            "checkpoint_every": 4,
        },
        "arms": [
            {"id": "A0", "mmd_coefficient": 0.0},
            {"id": "A64", "mmd_coefficient": 0.1},
        ],
        "stop": {
            "max_kl": 0.2,
            "min_unique_fraction": 0.5,
            "max_repeated_trigram_fraction": 0.5,
            "min_outline_fact_recall": 0.0,
            "max_unsupported_claim_rate": 1.0,
            "max_mean_abs_target_error": 1000.0,
        },
        "readiness_trust": {
            "trusted_public_keys_path": "/checkpoints/measurement/trusted-keys.json",
            "trusted_public_keys_sha256": SHA,
            "protocol_signer_key_id": "operator-key",
            "blind_signer_key_id": "blind-key",
        },
        "resume": {"A0": None, "A64": None},
        "execution": {"arm": "A0"},
        "workflow": {
            "protocol_version": DFT_SCHEMA,
            "step": DFT_STEP,
            "method_contract_sha256": "0" * 64,
        },
    }
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    return config


def test_strict_prospective_config_accepts_only_matched_a0_a64_contract():
    config = valid_config()
    assert validate_dft_config(config) is config

    config["arms"][0]["mmd_coefficient"] = 0.01
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    with pytest.raises(M2ConfigError, match="A0=0"):
        validate_dft_config(config)


def test_a0_and_a64_configs_differ_only_in_selector_and_share_method_contract():
    a0 = valid_config()
    a64 = copy.deepcopy(a0)
    a64["execution"]["arm"] = "A64"
    assert validate_dft_config(a0) is a0
    assert validate_dft_config(a64) is a64
    differing_top_level = {key for key in a0 if a0[key] != a64[key]}
    assert differing_top_level == {"execution"}
    assert method_contract_payload(a0) == method_contract_payload(a64)
    assert a0["workflow"]["method_contract_sha256"] == a64["workflow"][
        "method_contract_sha256"
    ]
    assert matched_exposure_payload(a0) == matched_exposure_payload(a64)


def test_method_contract_hash_and_training_only_paths_fail_closed():
    config = valid_config()
    config["training"]["learning_rate"] = 2e-5
    with pytest.raises(M2ConfigError, match="method contract hash mismatch"):
        validate_dft_config(config)

    config = valid_config()
    config["training"]["sampling_distribution"] = "top_p"
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    with pytest.raises(M2ConfigError, match="raw on-policy"):
        validate_dft_config(config)


def test_prompt_serializer_is_exactly_the_source_sft_full_brief_contract():
    record = {
        "user_prompt": "Explain the launch.",
        "use_case": "news",
        "style_kind": "reported",
        "style": "concise",
        "detail_mode": "grounded",
        "target_length": 180,
        "em_dashes_allowed": False,
        "outline": [{"section": "Launch", "supported_facts": ["It launched."], "quotations": []}],
    }
    config = valid_config()
    assert _render_prompt(record, config) == render_m1_prompt(
        record, config["data"]["prompt_format"], FULL_BRIEF_SCHEMA
    )


def test_source_adapter_data_and_human_only_bandwidth_hashes_fail_closed(
    monkeypatch, tmp_path
):
    adapter = tmp_path / "source-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": BASE_MODEL,
                "task_type": "CAUSAL_LM",
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    data_paths = {}
    for name in ("rollout", "anchor", "humans"):
        path = tmp_path / f"{name}.jsonl"
        rows = ['{"user_prompt":"brief","completion":"human text"}\n']
        if name == "humans":
            rows.append('{"user_prompt":"brief two","completion":"human text two"}\n')
        path.write_text("".join(rows), encoding="utf-8")
        data_paths[name] = path
    bandwidth_path = tmp_path / "bandwidths.json"

    config = valid_config()
    config["runtime"] = {
        "torch_version": torch.__version__,
        "transformers_version": "test-transformers",
        "peft_version": "test-peft",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test-transformers"))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(__version__="test-peft"))
    config["initial_adapter"].update(
        path=str(adapter),
        adapter_model_sha256=file_sha256(adapter / "adapter_model.safetensors"),
        adapter_config_sha256=file_sha256(adapter / "adapter_config.json"),
        file_manifest_sha256=canonical_hash(
            {
                item.relative_to(adapter).as_posix(): file_sha256(item)
                for item in sorted(adapter.rglob("*"))
                if item.is_file()
            }
        ),
    )
    config["data"].update(
        rollout_path=str(data_paths["rollout"]),
        rollout_sha256=file_sha256(data_paths["rollout"]),
        sft_anchor_path=str(data_paths["anchor"]),
        sft_anchor_sha256=file_sha256(data_paths["anchor"]),
        human_targets_path=str(data_paths["humans"]),
        human_targets_sha256=file_sha256(data_paths["humans"]),
    )
    producer_config = {
        "artifact_schema": "dftr.m2.prepare_training_bandwidths.v1",
        "run": {
            "comparison_id": "test-bandwidth-prepare", "arm": "training-bandwidths",
            "budget_class": "smoke", "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"], "seed": 0,
        },
        "compute": {"gpu": config["compute"]["gpu"], "gpus": 1, "timeout_min": 20},
        "model": config["model"],
        "initial_adapter": config["initial_adapter"],
        "data": {
            "human_targets_path": config["data"]["human_targets_path"],
            "human_targets_sha256": config["data"]["human_targets_sha256"],
            "human_text_field": config["data"]["human_text_field"],
        },
        "representation": config["representation"],
        "derivation": TRAINING_BANDWIDTH_DERIVATION,
        "runtime": config["runtime"],
        "output": {"filename": "training_bandwidths.json", "overwrite": False},
        "workflow": {
            "protocol_version": "dftr.m2.prepare_training_bandwidths.v1",
            "step": "prepare_dft", "preparation_contract_sha256": "0" * 64,
        },
    }
    producer_config["workflow"]["preparation_contract_sha256"] = canonical_hash(
        preparation_contract_payload(producer_config)
    )
    median_distance = 2.0
    values = [median_distance * scale**2 for scale in TRAINING_BANDWIDTH_DERIVATION["scales"]]
    bandwidth_payload = {
        "artifact_schema": "dftr.m2.training_bandwidths.v2",
        "source": "training_humans_only",
        "human_targets_sha256": config["data"]["human_targets_sha256"],
        "human_text_sequence_sha256": canonical_hash(["human text", "human text two"]),
        "representation_contract_sha256": canonical_hash(config["representation"]),
        "representation_execution_contract_sha256": canonical_hash(
            representation_execution_payload(config)
        ),
        "tokenizer_file_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "source_adapter_file_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "source_adapter_model_sha256": config["initial_adapter"]["adapter_model_sha256"],
        "source_adapter_config_sha256": config["initial_adapter"]["adapter_config_sha256"],
        "preparation_contract_sha256": producer_config["workflow"]["preparation_contract_sha256"],
        "preparation_contract": preparation_contract_payload(producer_config),
        "producer_run_id": "prepare-test", "producer_git_sha": "b" * 40,
        "producer_config_sha256": canonical_hash(producer_config),
        "producer_config": producer_config,
        "model_base": config["model"]["base"], "model_revision": config["model"]["revision"],
        "observed_runtime": {
            key: config["runtime"][key]
            for key in ("torch_version", "transformers_version", "peft_version")
        },
        "gpu": config["compute"]["gpu"], "observed_device_name": "test-gpu",
        "derivation": TRAINING_BANDWIDTH_DERIVATION,
        "human_document_count": 2, "embedding_dimension": 2,
        "total_unordered_pair_count": 1, "positive_pair_distance_count": 1,
        "zero_distance_count": 0,
        "median_positive_squared_distance": median_distance,
        "embedding_matrix_sha256": "c" * 64, "positive_distances_sha256": "d" * 64,
        "parameterization": "squared_distance_over_2_sigma_squared",
        "values": values, "values_sha256": canonical_hash(values),
    }
    bandwidth_path.write_text(json.dumps(bandwidth_payload), encoding="utf-8")
    config["kernel"].update(
        bandwidths_path=str(bandwidth_path),
        bandwidths_sha256=file_sha256(bandwidth_path),
    )
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    validate_dft_config(config)
    rollout, anchor, humans, bandwidths = _verify_inputs(config)
    assert len(rollout) == len(anchor) == 1
    assert len(humans) == 2
    assert bandwidths == values

    for field, replacement in (
        ("artifact_schema", "dftr.m2.training_bandwidths.v1"),
        ("values_sha256", "e" * 64),
        ("preparation_contract_sha256", "f" * 64),
        ("representation_execution_contract_sha256", "1" * 64),
    ):
        tampered = copy.deepcopy(bandwidth_payload)
        tampered[field] = replacement
        bandwidth_path.write_text(json.dumps(tampered), encoding="utf-8")
        config["kernel"]["bandwidths_sha256"] = file_sha256(bandwidth_path)
        with pytest.raises(M2ConfigError, match="bandwidth"):
            _verify_inputs(config)
    bandwidth_path.write_text(json.dumps(bandwidth_payload), encoding="utf-8")
    config["kernel"]["bandwidths_sha256"] = file_sha256(bandwidth_path)

    bandwidth_payload["human_targets_sha256"] = "b" * 64
    bandwidth_path.write_text(json.dumps(bandwidth_payload), encoding="utf-8")
    config["kernel"]["bandwidths_sha256"] = file_sha256(bandwidth_path)
    with pytest.raises(M2ConfigError, match="training humans only"):
        _verify_inputs(config)

    config = valid_config()
    config["data"]["human_targets_path"] = "/repo/harness/measurement_v2/humans.jsonl"
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    with pytest.raises(M2ConfigError, match="training-only"):
        validate_dft_config(config)


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_sha256(path)


def _materialize_real_a64_readiness(
    tmp_path: Path, config: dict
) -> tuple[Path, dict, Ed25519PrivateKey]:
    fixture = _measurement_fixture_module()
    measurement_root = tmp_path / "measurement"
    measurement_root.mkdir()
    protocol, operator_key, trusted = fixture.synthetic_evidence(measurement_root)
    blind_key = Ed25519PrivateKey.generate()
    blind_public = blind_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trusted["blind-key"] = base64.b64encode(blind_public).decode("ascii")
    trust_path = tmp_path / "trusted-keys.json"
    trust_sha = _write_json(trust_path, trusted)
    config["readiness_trust"].update(
        trusted_public_keys_path=str(trust_path),
        trusted_public_keys_sha256=trust_sha,
    )
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )

    checkpoint_dir = tmp_path / "a0-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"real A0 adapter bytes")
    (checkpoint_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    adapter_sha = file_sha256(checkpoint_dir / "adapter_model.safetensors")
    file_map = {
        item.relative_to(checkpoint_dir).as_posix(): file_sha256(item)
        for item in sorted(checkpoint_dir.rglob("*"))
        if item.is_file()
    }
    checkpoint = {
        "artifact_schema": "dftr.m2.adapter_native_checkpoint.v1",
        "arm": "A0",
        "status": "completed",
        "adapter_native": True,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_adapter_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "source_adapter_model_sha256": config["initial_adapter"]["adapter_model_sha256"],
        "source_adapter_config_sha256": config["initial_adapter"]["adapter_config_sha256"],
        "git_sha": "b" * 40,
        "checkpoint_dir": str(checkpoint_dir),
        "file_sha256": file_map,
        "file_map_excludes": ["checkpoint_manifest.json"],
        "generated_tokens": config["training"]["steps"]
        * config["training"]["rollout_batch_size"] * 64,
        "steps": config["training"]["steps"],
        "resumed_from_step": 0,
        "trainable_parameter_count": 1,
        "mmd_coefficient": 0.0,
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
    }
    checkpoint_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_sha = _write_json(checkpoint_path, checkpoint)

    output_path = measurement_root / "control-outputs.jsonl"
    output_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in output_rows:
        row["checkpoint_sha256"] = adapter_sha
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    output_sha = file_sha256(output_path)
    baseline_path = measurement_root / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["checkpoint_sha256"] = adapter_sha
    baseline["output_manifest_sha256"] = output_sha
    baseline_sha = _write_json(baseline_path, baseline)
    calibration_path = measurement_root / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["matched_baseline_sha256"] = baseline_sha
    calibration_sha = _write_json(calibration_path, calibration)
    power_path = measurement_root / "power.json"
    power = json.loads(power_path.read_text(encoding="utf-8"))
    power["matched_baseline_sha256"] = baseline_sha
    power["calibration_sha256"] = calibration_sha
    power_sha = _write_json(power_path, power)
    protocol["hashes"].update(
        matched_baseline_sha256=baseline_sha,
        matched_baseline_outputs_sha256=output_sha,
        calibration_sha256=calibration_sha,
        power_plan_sha256=power_sha,
    )
    protocol["artifact_bindings"]["matched_baseline"]["sha256"] = baseline_sha
    protocol["artifact_bindings"]["matched_baseline_outputs"]["sha256"] = output_sha
    protocol["artifact_bindings"]["calibration"]["sha256"] = calibration_sha
    protocol["artifact_bindings"]["power_plan"]["sha256"] = power_sha
    protocol["matched_design"]["control_checkpoint_sha256"] = adapter_sha
    protocol["matched_design"]["control_output_manifest_sha256"] = output_sha
    protocol.pop("operator_signature")
    fixture.sign(protocol, operator_key)
    protocol_path = measurement_root / "measurement_protocol.json"
    protocol_sha = _write_json(protocol_path, protocol)

    blind = {
        "artifact_schema": "dftr.measurement.blind_test_manifest.v2",
        "status": "qualified",
        "protocol_sha256": canonical_hash(protocol),
        "tests": [{"name": name, "status": "pass"} for name in sorted(REQUIRED_BLIND_GROUPS)],
        "evaluator_commit": protocol["hashes"]["metric_code_sha256"],
        "dependency_lock_sha256": protocol["hashes"]["dependency_lock_sha256"],
        "fixture_pack_sha256": "placeholder",
        "no_sealed_imitation": True,
        "signer_identity": "independent-blind-tester",
        "tested_at": "2026-07-17T00:00:00Z",
        "runtime_versions": {"python": "synthetic-real-contract"},
    }
    fixture_path = tmp_path / "blind-fixtures.tar"
    fixture_path.write_bytes(b"independent blind fixture pack")
    blind["fixture_pack_sha256"] = file_sha256(fixture_path)
    fixture.sign(blind, blind_key, key_id="blind-key")
    blind_path = tmp_path / "blind-qualified.json"
    blind_sha = _write_json(blind_path, blind)
    readiness = {
        "artifact_schema": "dftr.m2.a64_readiness.v1",
        "status": "ready",
        "comparison_id": config["run"]["comparison_id"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "a0_checkpoint_manifest": {
            "path": str(checkpoint_path), "sha256": checkpoint_sha,
            "adapter_model_sha256": adapter_sha,
        },
        "a0_generation_manifest": {
            "path": str(baseline_path), "sha256": baseline_sha,
            "output_path": str(output_path), "output_sha256": output_sha,
        },
        "measurement_protocol": {
            "path": str(protocol_path), "sha256": protocol_sha,
            "artifact_root": str(measurement_root),
        },
        "blind_qualification": {
            "path": str(blind_path), "sha256": blind_sha,
            "operator": "independent-blind-tester",
            "fixture_pack_path": str(fixture_path),
            "fixture_pack_sha256": file_sha256(fixture_path),
        },
        "trusted_public_keys": {"path": str(trust_path), "sha256": trust_sha},
    }
    readiness_path = tmp_path / "readiness.json"
    _write_json(readiness_path, readiness)
    return readiness_path, readiness, blind_key


def test_a64_readiness_accepts_real_signed_protocol_and_qualified_blind_evidence(
    monkeypatch, tmp_path
):
    config = valid_config()
    config["execution"]["arm"] = "A64"
    readiness_path, readiness, _ = _materialize_real_a64_readiness(tmp_path, config)
    readiness_sha = file_sha256(readiness_path)
    monkeypatch.setenv("DFTR_M2_A64_READINESS_MANIFEST", str(readiness_path))
    monkeypatch.setenv("DFTR_M2_A64_READINESS_SHA256", readiness_sha)
    assert _verify_a64_readiness(config) == readiness_sha

    readiness["comparison_id"] = "substitute"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    monkeypatch.setenv("DFTR_M2_A64_READINESS_SHA256", file_sha256(readiness_path))
    with pytest.raises(M2ConfigError, match="frozen comparison"):
        _verify_a64_readiness(config)


@pytest.mark.parametrize("forgery", ["adapter_bytes", "output_bytes", "blind_claim"])
def test_a64_readiness_rejects_forged_underlying_evidence(monkeypatch, tmp_path, forgery):
    config = valid_config()
    config["execution"]["arm"] = "A64"
    readiness_path, readiness, _ = _materialize_real_a64_readiness(tmp_path, config)
    if forgery == "adapter_bytes":
        Path(readiness["a0_checkpoint_manifest"]["path"]).parent.joinpath(
            "adapter_model.safetensors"
        ).write_bytes(b"forged adapter")
    elif forgery == "output_bytes":
        Path(readiness["a0_generation_manifest"]["output_path"]).write_text(
            '{"forged":true}\n', encoding="utf-8"
        )
    else:
        blind_path = Path(readiness["blind_qualification"]["path"])
        blind = json.loads(blind_path.read_text(encoding="utf-8"))
        blind["no_sealed_imitation"] = False
        _write_json(blind_path, blind)
        readiness["blind_qualification"]["sha256"] = file_sha256(blind_path)
        _write_json(readiness_path, readiness)
    monkeypatch.setenv("DFTR_M2_A64_READINESS_MANIFEST", str(readiness_path))
    monkeypatch.setenv("DFTR_M2_A64_READINESS_SHA256", file_sha256(readiness_path))
    with pytest.raises(M2ConfigError, match="byte|file map|hash mismatch|signature|qualification"):
        _verify_a64_readiness(config)


def test_a64_readiness_rejects_self_declared_minimal_json(monkeypatch, tmp_path):
    config = valid_config()
    config["execution"]["arm"] = "A64"
    fake = {
        "artifact_schema": "dftr.m2.a64_readiness.v1",
        "status": "ready",
        "comparison_id": config["run"]["comparison_id"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "a0_checkpoint_manifest": {},
        "a0_generation_manifest": {},
        "measurement_protocol": {},
        "blind_qualification": {
            "artifact_schema": "dftr.measurement.blind_test_manifest.v2",
            "status": "pass",
        },
    }
    path = tmp_path / "self-declared.json"
    _write_json(path, fake)
    monkeypatch.setenv("DFTR_M2_A64_READINESS_MANIFEST", str(path))
    monkeypatch.setenv("DFTR_M2_A64_READINESS_SHA256", file_sha256(path))
    with pytest.raises(M2ConfigError, match="schema mismatch"):
        _verify_a64_readiness(config)


def test_a64_readiness_rejects_submitter_supplied_trust_store(monkeypatch, tmp_path):
    config = valid_config()
    config["execution"]["arm"] = "A64"
    readiness_path, readiness, _ = _materialize_real_a64_readiness(tmp_path, config)
    attacker_trust = tmp_path / "attacker-trust.json"
    _write_json(attacker_trust, {"attacker-one": "A" * 44, "attacker-two": "B" * 44})
    readiness["trusted_public_keys"] = {
        "path": str(attacker_trust), "sha256": file_sha256(attacker_trust)
    }
    _write_json(readiness_path, readiness)
    monkeypatch.setenv("DFTR_M2_A64_READINESS_MANIFEST", str(readiness_path))
    monkeypatch.setenv("DFTR_M2_A64_READINESS_SHA256", file_sha256(readiness_path))
    with pytest.raises(M2ConfigError, match="frozen method contract"):
        _verify_a64_readiness(config)


def test_a64_readiness_rejects_same_public_key_under_two_ids(monkeypatch, tmp_path):
    config = valid_config()
    config["execution"]["arm"] = "A64"
    readiness_path, readiness, _ = _materialize_real_a64_readiness(tmp_path, config)
    trust_path = Path(readiness["trusted_public_keys"]["path"])
    trusted = json.loads(trust_path.read_text(encoding="utf-8"))
    trusted["blind-key"] = base64.b64decode(trusted["operator-key"]).hex()
    trust_sha = _write_json(trust_path, trusted)
    config["readiness_trust"]["trusted_public_keys_sha256"] = trust_sha
    method_sha = canonical_hash(method_contract_payload(config))
    config["workflow"]["method_contract_sha256"] = method_sha
    checkpoint_path = Path(readiness["a0_checkpoint_manifest"]["path"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["method_contract_sha256"] = method_sha
    readiness["a0_checkpoint_manifest"]["sha256"] = _write_json(
        checkpoint_path, checkpoint
    )
    readiness["method_contract_sha256"] = method_sha
    readiness["trusted_public_keys"]["sha256"] = trust_sha
    _write_json(readiness_path, readiness)
    monkeypatch.setenv("DFTR_M2_A64_READINESS_MANIFEST", str(readiness_path))
    monkeypatch.setenv("DFTR_M2_A64_READINESS_SHA256", file_sha256(readiness_path))
    with pytest.raises(M2ConfigError, match="trust identities"):
        _verify_a64_readiness(config)


@pytest.mark.parametrize("group_mutation", ["extra", "failed"])
def test_a64_readiness_rejects_nonexact_signed_blind_groups(
    monkeypatch, tmp_path, group_mutation
):
    config = valid_config()
    config["execution"]["arm"] = "A64"
    readiness_path, readiness, blind_key = _materialize_real_a64_readiness(tmp_path, config)
    blind_path = Path(readiness["blind_qualification"]["path"])
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    blind.pop("operator_signature")
    if group_mutation == "extra":
        blind["tests"].append({"name": "attacker-extra", "status": "pass"})
    else:
        blind["tests"][0]["status"] = "fail"
    _measurement_fixture_module().sign(blind, blind_key, key_id="blind-key")
    _write_json(blind_path, blind)
    readiness["blind_qualification"]["sha256"] = file_sha256(blind_path)
    _write_json(readiness_path, readiness)
    monkeypatch.setenv("DFTR_M2_A64_READINESS_MANIFEST", str(readiness_path))
    monkeypatch.setenv("DFTR_M2_A64_READINESS_SHA256", file_sha256(readiness_path))
    with pytest.raises(M2ConfigError, match="exact passing group|frozen group set"):
        _verify_a64_readiness(config)


def test_separate_jobs_account_only_executed_arm_and_preserve_matched_exposure(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        dft_module,
        "_verify_inputs",
        lambda _config: ([{}] * 4, [{}] * 2, [{}, {}], [1.0]),
    )
    monkeypatch.setattr(
        dft_module,
        "_verify_a64_readiness",
        lambda value: "3" * 64 if value["execution"]["arm"] == "A64" else None,
    )
    monkeypatch.setattr(
        dft_module,
        "build_run_paths",
        lambda _config, run_id: (
            tmp_path / run_id / "output",
            tmp_path / run_id / "checkpoint",
        ),
    )

    def fake_run_arm(config, arm, *_args):
        expected = config["training"]["steps"] * config["training"]["rollout_batch_size"] * 64
        return {"arm": arm["id"], "generated_tokens": expected}

    monkeypatch.setattr(dft_module, "_run_arm", fake_run_arm)
    results = {}
    for arm_id in ("A0", "A64"):
        config = valid_config()
        config["execution"]["arm"] = arm_id
        results[arm_id] = run_dft(config, f"run-{arm_id}")
        assert results[arm_id]["executed_arm"] == arm_id
        assert results[arm_id]["arm"]["arm"] == arm_id
        assert results[arm_id]["token_accounting"] == {
            "executed_arm": arm_id,
            "generated_tokens": 2048,
            "total_tokens": 2048,
        }
    assert results["A0"]["method_contract_sha256"] == results["A64"][
        "method_contract_sha256"
    ]
    assert results["A0"]["matched_exposure_contract_sha256"] == results["A64"][
        "matched_exposure_contract_sha256"
    ]


def _kernel(left: torch.Tensor, right: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return torch.exp(-torch.sum((left - right) ** 2) / (2.0 * bandwidth))


def test_vectorized_mmd_rewards_and_true_leave_one_out_controls_match_loops():
    generated = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    humans = torch.tensor([[0.5, 0.5], [1.5, 1.0], [-0.5, 1.0]])
    bandwidth = 1.7
    observed_rewards = mmd_score_rewards(generated, humans, [bandwidth])
    expected_rewards = []
    for i in range(len(generated)):
        generated_mean = sum(
            _kernel(generated[i], generated[j], bandwidth)
            for j in range(len(generated))
            if j != i
        ) / (len(generated) - 1)
        human_mean = sum(_kernel(generated[i], row, bandwidth) for row in humans) / len(humans)
        expected_rewards.append(-(2 * generated_mean - 2 * human_mean))
    assert torch.allclose(observed_rewards, torch.stack(expected_rewards), atol=1e-6)

    controls = mmd_leave_one_out_baselines(generated, humans, [bandwidth])
    mutated = generated.clone()
    mutated[2] = torch.tensor([100.0, -100.0])
    mutated_controls = mmd_leave_one_out_baselines(mutated, humans, [bandwidth])
    assert controls[2] == pytest.approx(mutated_controls[2], abs=1e-7)
    assert not torch.allclose(controls, mutated_controls)


def test_score_function_gradient_is_finite_and_zero_coefficient_matches_a0():
    log_probs = torch.tensor([-5.0, -7.0, -6.0], requires_grad=True)
    advantages = torch.tensor([0.25, -0.5, 0.75])
    loss = score_function_loss(log_probs, advantages, 0.2)
    loss.backward()
    assert torch.all(torch.isfinite(log_probs.grad))
    assert torch.allclose(log_probs.grad, -0.2 * advantages / len(advantages))

    a0_log_probs = log_probs.detach().clone().requires_grad_(True)
    a0_loss = score_function_loss(a0_log_probs, advantages, 0.0)
    a0_loss.backward()
    assert a0_loss.item() == 0.0
    assert torch.equal(a0_log_probs.grad, torch.zeros_like(a0_log_probs))


def test_zero_mmd_coefficient_preserves_the_full_a0_optimizer_update():
    control = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    zero_treatment = torch.nn.Parameter(control.detach().clone())
    control_optimizer = torch.optim.AdamW([control], lr=1e-2)
    treatment_optimizer = torch.optim.AdamW([zero_treatment], lr=1e-2)
    advantages = torch.tensor([0.7, -0.1])

    control_loss = control.pow(2).sum()
    treatment_loss = zero_treatment.pow(2).sum() + score_function_loss(
        zero_treatment, advantages, 0.0
    )
    control_loss.backward()
    treatment_loss.backward()
    control_optimizer.step()
    treatment_optimizer.step()
    assert torch.equal(control, zero_treatment)
    assert control_optimizer.state_dict()["state"].keys() == treatment_optimizer.state_dict()[
        "state"
    ].keys()


def test_streamed_per_sample_score_terms_equal_the_batched_objective_gradient():
    batched_log_probs = torch.tensor([-4.0, -5.0, -6.0], requires_grad=True)
    streamed_log_probs = batched_log_probs.detach().clone().requires_grad_(True)
    advantages = torch.tensor([0.2, -0.4, 0.7])
    log_ratios = torch.tensor([0.01, -0.02, 0.03])
    batched = score_function_loss(batched_log_probs, advantages, 0.15) + 0.02 * (
        (log_ratios + 1.0 / 64) * batched_log_probs
    ).mean()
    streamed = sum(
        per_sample_score_loss(
            streamed_log_probs[index], advantages[index], log_ratios[index],
            mmd_coefficient=0.15, kl_coefficient=0.02, batch_size=3,
        )
        for index in range(3)
    )
    assert streamed.item() == pytest.approx(batched.item())
    assert torch.allclose(
        torch.autograd.grad(streamed, streamed_log_probs)[0],
        torch.autograd.grad(batched, batched_log_probs)[0],
        atol=1e-9,
    )


def test_enumerated_score_gradient_matches_mmd_objective_and_finite_difference():
    outcomes = torch.tensor([[0.0, 0.0], [1.5, -0.5]], dtype=torch.double)
    humans = torch.tensor([[0.25, 0.5], [1.0, 1.0]], dtype=torch.double)
    bandwidths = [0.8]

    def expected_objective(theta_value):
        logits = torch.stack((theta_value, theta_value.new_zeros(())))
        probabilities = torch.softmax(logits, dim=0)
        total = theta_value.new_zeros(())
        for choices in itertools.product(range(2), repeat=3):
            generated = outcomes[list(choices)]
            xx = torch.exp(-torch.cdist(generated, generated).pow(2) / 1.6)
            xy = torch.exp(-torch.cdist(generated, humans).pow(2) / 1.6)
            objective = (xx.sum() - torch.diagonal(xx).sum()) / 6.0 - 2.0 * xy.mean()
            probability = torch.stack([probabilities[index] for index in choices]).prod()
            total = total + probability * objective
        return total

    theta = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    exact_gradient = torch.autograd.grad(expected_objective(theta), theta)[0]
    logits = torch.stack((theta, theta.new_zeros(())))
    probabilities = torch.softmax(logits, dim=0)
    surrogate = theta.new_zeros(())
    for choices in itertools.product(range(2), repeat=3):
        generated = outcomes[list(choices)]
        rewards = mmd_score_rewards(generated, humans, bandwidths)
        controls = mmd_leave_one_out_baselines(generated, humans, bandwidths)
        log_probs = torch.stack([torch.log_softmax(logits, dim=0)[index] for index in choices])
        joint_probability = torch.stack([probabilities[index] for index in choices]).prod()
        surrogate = surrogate + joint_probability.detach() * score_function_loss(
            log_probs, rewards - controls, 1.0
        )
    estimator_gradient = torch.autograd.grad(surrogate, theta)[0]
    epsilon = 1e-5
    finite_difference = (
        expected_objective(torch.tensor(0.3 + epsilon, dtype=torch.double))
        - expected_objective(torch.tensor(0.3 - epsilon, dtype=torch.double))
    ) / (2 * epsilon)
    assert estimator_gradient.item() == pytest.approx(exact_gradient.item(), abs=1e-9)
    assert estimator_gradient.item() == pytest.approx(finite_difference.item(), abs=1e-7)


def test_sequence_log_probability_preserves_left_padding_mask_and_scores_only_rollout():
    class Model:
        def __call__(self, *, input_ids, attention_mask, return_dict):
            self.attention_mask = attention_mask
            return SimpleNamespace(
                logits=torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
            )

    model = Model()
    sequences = torch.tensor([[0, 4, 5, 6, 7]])
    prompt_mask = torch.tensor([[0, 1, 1]])
    result = _sequence_log_probs(model, sequences, prompt_mask, prompt_width=3)
    assert torch.equal(model.attention_mask, torch.tensor([[0, 1, 1, 1, 1]]))
    assert result.item() == pytest.approx(-2 * torch.log(torch.tensor(8.0)).item())


def test_raw_policy_sampler_ignores_eos_stopping_and_uses_no_generation_warpers():
    class Model:
        def __call__(
            self, *, input_ids, attention_mask, past_key_values, use_cache, return_dict
        ):
            logits = torch.full((*input_ids.shape, 5), -1000.0)
            logits[:, -1, 2] = 1000.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    prompt = torch.tensor([[0, 4]])
    mask = torch.tensor([[0, 1]])
    sampled = _sample_raw_policy(Model(), prompt, mask, new_tokens=4)
    assert sampled.tolist() == [[0, 4, 2, 2, 2, 2]]


def test_optimizer_and_rng_state_resume_reproduces_uninterrupted_updates(tmp_path):
    class TinyPolicy(torch.nn.Linear):
        def save_pretrained(self, target, **_kwargs):
            torch.save(self.state_dict(), target / "adapter_model.safetensors")
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")

    def advance(model, optimizer, count):
        for _ in range(count):
            sample = torch.randn(3)
            scale = random.random()
            loss = (model(sample).sum() * scale).pow(2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    torch.manual_seed(7)
    initial = TinyPolicy(3, 2)
    initial_state = copy.deepcopy(initial.state_dict())

    continuous = TinyPolicy(3, 2)
    continuous.load_state_dict(initial_state)
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=1e-3)
    random.seed(19)
    torch.manual_seed(23)
    advance(continuous, continuous_optimizer, 4)

    interrupted = TinyPolicy(3, 2)
    interrupted.load_state_dict(initial_state)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1e-3)
    random.seed(19)
    torch.manual_seed(23)
    advance(interrupted, interrupted_optimizer, 2)
    config = valid_config()
    target = tmp_path / "step-2"
    logs = [{"step": 0}, {"step": 1}]
    _save_training_checkpoint(
        interrupted, interrupted_optimizer, target, "A0", 2, logs,
        2 * config["training"]["rollout_batch_size"] * 64, config,
    )
    descriptor = {
        "path": str(target),
        "adapter_model_sha256": file_sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(target / "adapter_config.json"),
        "training_state_sha256": file_sha256(target / "training_state.pt"),
        "file_manifest_sha256": canonical_hash(
            {
                item.relative_to(target).as_posix(): file_sha256(item)
                for item in sorted(target.rglob("*")) if item.is_file()
            }
        ),
    }
    state = _verify_resume_artifact(descriptor, "A0", config)
    resumed = TinyPolicy(3, 2)
    resumed.load_state_dict(torch.load(target / "adapter_model.safetensors", weights_only=True))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(state["optimizer_state"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["cpu_rng_state"])
    advance(resumed, resumed_optimizer, 2)
    for expected, observed in zip(continuous.parameters(), resumed.parameters()):
        assert torch.equal(expected, observed)


def test_seeded_accounting_and_repetition_sentinel_are_deterministic():
    first = deterministic_schedule(5, 17, 11)
    assert first == deterministic_schedule(5, 17, 11)
    assert first != deterministic_schedule(5, 17, 12)
    batches = deterministic_batches(5, 4, 8, 11)
    assert batches == deterministic_batches(5, 4, 8, 11)
    assert all(len(set(batch)) == 4 for batch in batches)
    assert repeated_ngram_fraction([1, 2, 3, 1, 2, 3]) == pytest.approx(0.25)


def test_training_only_factual_and_adherence_sentinels_are_emitted():
    values = training_factual_adherence_sentinels(
        ["Acme launched the Atlas system today."],
        [{
            "target_length": 6,
            "outline": [{
                "supported_facts": ["Acme launched the Atlas system today"],
                "quotations": [],
            }],
        }],
    )
    assert set(values) == {
        "outline_fact_recall", "unsupported_claim_rate", "mean_abs_target_error"
    }
    assert values["outline_fact_recall"] == 1.0
    assert values["mean_abs_target_error"] == 0.0


def test_runner_dispatches_dft_protocol_without_falling_through(monkeypatch, capsys):
    config = valid_config()
    monkeypatch.setattr(runner, "_load_config", lambda _path: config)
    monkeypatch.setattr(runner, "run_dft", lambda value, run_id: {"kind": "dft", "run_id": run_id})
    monkeypatch.setattr(runner, "run_m1", lambda *_args: pytest.fail("M1 path selected"))
    monkeypatch.setattr(runner, "run_smoke", lambda *_args: pytest.fail("smoke path selected"))
    assert runner.main(["--config", "prospective.json", "--run-id", "run-1"]) == 0
    assert '"kind": "dft"' in capsys.readouterr().out


def test_runner_rejects_train_dft_protocol_substitution(monkeypatch):
    config = valid_config()
    config["workflow"]["protocol_version"] = "dftr.m2.substitute.v1"
    monkeypatch.setattr(runner, "_load_config", lambda _path: config)
    with pytest.raises(ValueError, match="frozen M2 score-function MMD"):
        runner.main(["--config", "substitute.json", "--run-id", "run-1"])


def test_gateway_rejects_dft_protocol_substitution_before_reservation():
    config = valid_config()
    payload = {
        "run_id": "dftr-dft-test",
        "config": config,
        "config_hash": policy_hash(config),
        "git_sha": "b" * 40,
        "budget_class": "screen",
        "preregistration": {
            "kind": "prereg",
            "comparison": config["run"]["comparison_id"],
            "status": "open",
        },
    }
    assert validate_launch(payload).task_kind == "experiment"
    config["workflow"]["protocol_version"] = "dftr.m2.substitute.v1"
    payload["config_hash"] = policy_hash(config)
    with pytest.raises(PolicyError, match="frozen M2 DFT"):
        validate_launch(payload)


def test_gateway_requires_wrapper_readiness_only_for_later_a64_job(monkeypatch, tmp_path):
    config = valid_config()
    config["execution"]["arm"] = "A64"
    payload = {
        "run_id": "dftr-a64-test",
        "config": config,
        "config_hash": policy_hash(config),
        "git_sha": "b" * 40,
        "budget_class": "screen",
        "preregistration": {
            "kind": "prereg",
            "comparison": config["run"]["comparison_id"],
            "status": "open",
        },
    }
    with pytest.raises(PolicyError, match="wrapper readiness"):
        validate_launch(payload)
    payload["dft_a64_readiness"] = {
        "kind": "dft_a64_readiness",
        "status": "ready",
        "comparison": config["run"]["comparison_id"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "manifest_path": "/checkpoints/readiness/a64.json",
        "manifest_sha256": "c" * 64,
    }
    monkeypatch.setenv("DFTR_M2_TRUSTED_KEYS_SHA256", SHA)
    assert validate_launch(payload).task_kind == "experiment"

    with pytest.raises(PolicyError, match="independently configured Modal"):
        validate_launch(payload, backend="local")

    config["arms"][1]["mmd_coefficient"] = 0.2
    payload["config_hash"] = policy_hash(config)
    with pytest.raises(PolicyError, match="method contract hash mismatch"):
        validate_launch(payload)

    a0 = valid_config()
    payload["config"] = a0
    payload["config_hash"] = policy_hash(a0)
    with pytest.raises(PolicyError, match="A0 cannot consume"):
        validate_launch(payload)


def test_client_rejects_local_a64_even_with_submitter_trust_env(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(str(root / "infra" / "gpu"))
    client["_validate_submit"].__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg", "status": "open", "comparison": comparison
    }
    monkeypatch.setenv("DFTR_GPU_BACKEND", "local")
    monkeypatch.setenv("DFTR_M2_TRUSTED_KEYS_SHA256", SHA)
    config = valid_config()
    config["execution"]["arm"] = "A64"
    with pytest.raises(SystemExit):
        client["_validate_submit"](config, "screen")


def test_training_module_does_not_import_harness_or_measurement_v2():
    source_path = Path(__file__).resolve().parents[1] / "m2" / "dft.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name == "harness" or name.startswith("harness.") for name in imports)
    assert not any("measurement_v2" in name for name in imports)
