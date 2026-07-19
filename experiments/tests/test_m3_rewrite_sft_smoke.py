from __future__ import annotations

import copy

import pytest

from experiments.m2.representation import canonical_hash
from experiments.m3.rewrite_sft_smoke import (
    M3_REWRITE_SFT_SMOKE_SCHEMA,
    M3_REWRITE_SFT_SMOKE_STEP,
    M3RewriteSFTSmokeError,
    deterministic_epoch_indices,
    method_contract_payload,
    prepare_example,
    validate_m3_rewrite_sft_smoke_config,
)
from backend.policy import canonical_hash as policy_hash, validate_launch


def valid_config() -> dict:
    config = {
        "artifact_schema": M3_REWRITE_SFT_SMOKE_SCHEMA,
        "run": {
            "comparison_id": "M3-rewrite-sft-14b-mechanical-smoke-v1",
            "arm": "SFT14-mechanical-smoke",
            "budget_class": "smoke",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 1701,
        },
        "compute": {"gpu": "H100", "gpus": 1, "timeout_min": 20},
        "model": {
            "base": "Qwen/Qwen3-14B",
            "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
            "torch_dtype": "bfloat16",
        },
        "data": {
            "source_briefs_path": "/checkpoints/data/m3/source.jsonl",
            "source_briefs_sha256": "a" * 64,
            "rewrite_tasks_path": "/checkpoints/data/m3/rewrite.jsonl",
            "rewrite_tasks_sha256": "b" * 64,
            "source_records": 128,
            "rewrite_records": 96,
        },
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "optimizer_steps": 16,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "max_prompt_tokens": 768,
            "max_completion_tokens": 383,
            "max_sequence_tokens": 1024,
            "checkpoint_every": 8,
            "schedule": "single_seeded_epoch.v1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "workflow": {
            "protocol_version": M3_REWRITE_SFT_SMOKE_SCHEMA,
            "step": M3_REWRITE_SFT_SMOKE_STEP,
            "method_contract_sha256": "0" * 64,
        },
    }
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    return config


def test_smoke_config_and_schedule_are_frozen() -> None:
    config = valid_config()
    assert validate_m3_rewrite_sft_smoke_config(config) is config
    schedule = deterministic_epoch_indices(128, 1701)
    assert len(schedule) == 128
    assert sorted(schedule) == list(range(128))


def test_smoke_config_rejects_method_drift() -> None:
    config = copy.deepcopy(valid_config())
    config["lora"]["r"] = 64
    with pytest.raises(M3RewriteSFTSmokeError, match="LoRA contract"):
        validate_m3_rewrite_sft_smoke_config(config)


def test_gateway_accepts_only_human_approved_frozen_14b_smoke() -> None:
    config = valid_config()
    payload = {
        "run_id": "dftr-m3-smoke-test",
        "config": config,
        "config_hash": policy_hash(config),
        "git_sha": "a" * 40,
        "budget_class": "smoke",
        "preregistration": {
            "kind": "prereg",
            "comparison": config["run"]["comparison_id"],
            "status": "open",
        },
        "human_scaleup_approved": True,
    }
    policy = validate_launch(payload)
    assert policy.gpu == "H100"
    assert 0 < policy.worst_case_cost_usd < 2

    payload["human_scaleup_approved"] = False
    with pytest.raises(Exception, match="human approval"):
        validate_launch(payload)


class FakeTokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, **_kwargs):
        return f"USER: {messages[0]['content']} ASSISTANT:"

    def encode(self, text, **_kwargs):
        return list(range(1, len(text.split()) + 1))


def test_prepare_example_masks_prompt_and_counts_completion() -> None:
    row = {"prompt": "Rewrite this source naturally.", "completion": "A concise human answer."}
    batch = prepare_example(FakeTokenizer(), row, valid_config())
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["completion_tokens"] == 5
    assert (batch["labels"] == -100).sum().item() == 6
    assert batch["input_ids"][0, -1].item() == 9
