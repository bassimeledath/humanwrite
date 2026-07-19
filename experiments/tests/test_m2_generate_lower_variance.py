from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.m2.generate_lower_variance import (
    GenerationConfigError,
    _checkpoint_file_map,
    _require_empty_wrapper_checkpoint_dir,
    canonical_hash,
    decoding_policy_payload,
    generation_contract_payload,
    validate_generation_config,
)
from experiments.m2.materialize_scale_ladder_generation_configs import (
    COMPARISON_ID,
    materialize,
)
from infra.backend.policy import PolicyError, validate_launch


ROOT = Path(__file__).resolve().parents[2]


def config() -> dict:
    value = {
        "artifact_schema": "dftr.m2.lower_variance_generation.v1",
        "run": {
            "comparison_id": COMPARISON_ID,
            "arm": "SFT-generation",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 101,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 120},
        "model": {
            "base": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "torch_dtype": "bfloat16",
        },
        "checkpoint": {
            "path": "/checkpoints/runs/dftr-sft/SFT",
            "manifest_sha256": "a" * 64,
            "adapter_model_sha256": "b" * 64,
            "arm": "SFT",
            "method_contract_sha256": "c" * 64,
        },
        "data": {
            "prompt_briefs_path": "/checkpoints/data/m2-scale-ladder-v1/scale-dev-panels/prompt_briefs-128.jsonl",
            "prompt_briefs_sha256": "d" * 64,
            "prompt_format": "USER:\n{brief}\nASSISTANT:",
            "prompt_schema_version": "dft.full-brief.tokens.v1",
            "prompt_serializer_sha256": "1f92174518dfac375abbbbcf4ceba0659b726cabb215e0561a9fbffc4036b4a1",
        },
        "sampling": {
            "training_seed": 11,
            "sampling_seed": 101,
            "seed_scope": "single_global_rng_stream",
            "prompt_order": "sorted_prompt_id",
            "distribution": "raw_policy_categorical",
            "batch_size": 4,
            "new_tokens": 128,
            "max_input_tokens": 1024,
            "decode": {"skip_special_tokens": True},
        },
        "runtime": {
            "torch_version": "2.13.0+cu130",
            "transformers_version": "4.57.6",
            "peft_version": "0.19.1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "output": {"filename": "outputs.jsonl", "overwrite": False},
        "workflow": {
            "protocol_version": "dftr.m2.lower_variance_generation.v1",
            "step": "generate_lower_variance",
            "generation_contract_sha256": "",
            "decoding_policy_sha256": "",
        },
    }
    value["workflow"]["generation_contract_sha256"] = canonical_hash(
        generation_contract_payload(value)
    )
    value["workflow"]["decoding_policy_sha256"] = canonical_hash(
        decoding_policy_payload(value)
    )
    return value


def payload(value: dict) -> dict:
    return {
        "config": value,
        "config_hash": canonical_hash(value),
        "git_sha": "a" * 40,
        "budget_class": "screen",
        "preregistration": {
            "kind": "prereg",
            "comparison": value["run"]["comparison_id"],
            "status": "open",
        },
    }


def test_frozen_generation_config_validates() -> None:
    value = config()
    assert validate_generation_config(value) is value
    assert value["workflow"]["generation_contract_sha256"] == canonical_hash(
        generation_contract_payload(value)
    )
    assert value["workflow"]["decoding_policy_sha256"] == canonical_hash(
        decoding_policy_payload(value)
    )


def test_generation_contract_binds_prompt_bytes() -> None:
    value = config()
    original = canonical_hash(generation_contract_payload(value))
    value["data"]["prompt_briefs_sha256"] = "f" * 64
    assert canonical_hash(generation_contract_payload(value)) != original


def test_checkpoint_file_map_rejects_symlinks_and_detects_mutation(tmp_path: Path) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    original = _checkpoint_file_map(tmp_path)
    (tmp_path / "adapter_config.json").write_text('{"changed":true}', encoding="utf-8")
    assert _checkpoint_file_map(tmp_path) != original
    (tmp_path / "escape").symlink_to(tmp_path / "adapter_config.json")
    with pytest.raises(GenerationConfigError, match="symlink"):
        _checkpoint_file_map(tmp_path)


def test_wrapper_checkpoint_dir_allows_worker_log_only(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "worker.log").write_text("wrapper output\n", encoding="utf-8")
    _require_empty_wrapper_checkpoint_dir(root)
    (root / "outputs.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(
        GenerationConfigError, match="empty wrapper checkpoint directory"
    ):
        _require_empty_wrapper_checkpoint_dir(root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["sampling"].update({"new_tokens": 64}),
        lambda value: value["sampling"].update({"distribution": "top_p"}),
        lambda value: value["sampling"].update({"sampling_seed": 102}),
        lambda value: value["checkpoint"].update({"arm": "MMD_WITNESS"}),
        lambda value: value["data"].update({"prompt_briefs_path": "relative.jsonl"}),
    ],
)
def test_generation_config_fails_closed(mutate) -> None:
    value = config()
    mutate(value)
    with pytest.raises(GenerationConfigError):
        validate_generation_config(value)


def test_gateway_accepts_only_modal_volume_generation() -> None:
    value = config()
    policy = validate_launch(payload(value), backend="modal")
    assert policy.task_kind == "experiment"
    changed = copy.deepcopy(value)
    changed["checkpoint"]["path"] = "/tmp/checkpoint"
    with pytest.raises(PolicyError, match="generate_lower_variance"):
        validate_launch(payload(changed), backend="modal")


def test_gateway_rejects_local_generation() -> None:
    with pytest.raises(PolicyError, match="generate_lower_variance"):
        validate_launch(payload(config()), backend="local")


def test_gateway_rejects_sampling_drift_before_worker() -> None:
    value = config()
    value["sampling"]["new_tokens"] = 127
    with pytest.raises(PolicyError, match="generate_lower_variance"):
        validate_launch(payload(value), backend="modal")


def test_materializer_binds_completed_statuses(tmp_path: Path) -> None:
    sft_status = {
        "status": "completed",
        "comparison": "M2-scale-ladder-4b-4096-v1",
        "workflow_step": "train_lower_variance",
        "arm_executed_arm": "SFT",
        "arm_checkpoint_dir_path": "/checkpoints/runs/dftr-sft/SFT",
        "arm_checkpoint_manifest_sha256": "a" * 64,
        "arm_adapter_model_sha256": "b" * 64,
        "arm_method_contract_sha256": "c" * 64,
    }
    mmd_status = {
        **sft_status,
        "arm_executed_arm": "MMD_WITNESS",
        "arm_checkpoint_dir_path": "/checkpoints/runs/dftr-mmd/MMD_WITNESS",
        "arm_checkpoint_manifest_sha256": "d" * 64,
        "arm_adapter_model_sha256": "e" * 64,
        "arm_method_contract_sha256": "f" * 64,
    }
    prompt_status = {
        "status": "completed",
        "records_failed": 0,
        "records_processed": 3,
        "output_uri": "modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/scale-dev-panels/prompt_briefs-128.jsonl",
        "output_sha256": "1" * 64,
    }
    sft_path = tmp_path / "sft.json"
    mmd_path = tmp_path / "mmd.json"
    prompt_path = tmp_path / "prompt.json"
    sft_path.write_text(json.dumps(sft_status), encoding="utf-8")
    mmd_path.write_text(json.dumps(mmd_status), encoding="utf-8")
    prompt_path.write_text(json.dumps(prompt_status), encoding="utf-8")
    result = materialize(
        sft_status_path=sft_path,
        mmd_status_path=mmd_path,
        prompt_status_path=prompt_path,
        output_dir=tmp_path / "configs",
    )
    assert result["comparison_id"] == COMPARISON_ID
    sft_config = json.loads(
        json.dumps(
            yaml_safe_load((tmp_path / "configs" / "m2_scale_ladder_4b_4096_sft_generation_v1.yaml").read_text(encoding="utf-8"))
        )
    )
    assert sft_config["run"]["arm"] == "SFT-generation"
    assert sft_config["checkpoint"]["path"] == "/checkpoints/runs/dftr-sft/SFT"
    assert sft_config["data"]["prompt_briefs_path"].endswith("prompt_briefs-128.jsonl")


def yaml_safe_load(text: str) -> dict:
    import yaml

    return yaml.safe_load(text)
