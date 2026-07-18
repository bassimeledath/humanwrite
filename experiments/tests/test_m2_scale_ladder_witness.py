from __future__ import annotations

import copy

import pytest

from experiments.m2.lower_variance_train import (
    BASE_MODEL,
    BASE_REVISION,
    CONFIRMATION_GENERATION_CONTRACT,
    FULL_BRIEF_SCHEMA,
    FULL_BRIEF_SERIALIZER_SHA256,
    canonical_hash,
)
from experiments.m2.scale_ladder_witness import (
    SCALE_LADDER_WITNESS_SCHEMA,
    SCALE_LADDER_WITNESS_STEP,
    ScaleLadderWitnessError,
    validate_scale_ladder_witness_config,
    witness_contract_payload,
)


def config() -> dict:
    value = {
        "artifact_schema": SCALE_LADDER_WITNESS_SCHEMA,
        "run": {
            "comparison_id": "M2-scale-ladder-witness-4096-v1",
            "arm": "SFT-baseline-witness-4096",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 41001,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 90},
        "model": {
            "base": BASE_MODEL,
            "revision": BASE_REVISION,
            "torch_dtype": "bfloat16",
        },
        "initial_adapter": {
            "path": "/checkpoints/runs/source/seed-11",
            "adapter_model_sha256": "a" * 64,
            "adapter_config_sha256": "b" * 64,
            "file_manifest_sha256": "c" * 64,
        },
        "data": {
            "briefs_path": "/checkpoints/data/train-briefs-4096.jsonl",
            "briefs_sha256": "d" * 64,
            "expected_documents": 4096,
            "output_dir": "/checkpoints/data/witness-4096-v1",
            "prompt_format": "USER:\n{brief}\nASSISTANT:",
            "prompt_schema_version": FULL_BRIEF_SCHEMA,
            "prompt_serializer_sha256": FULL_BRIEF_SERIALIZER_SHA256,
        },
        "generation": dict(CONFIRMATION_GENERATION_CONTRACT),
        "runtime": {
            "torch_version": "2.13.0+cu130",
            "transformers_version": "4.57.6",
            "peft_version": "0.19.1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "workflow": {
            "protocol_version": SCALE_LADDER_WITNESS_SCHEMA,
            "step": SCALE_LADDER_WITNESS_STEP,
            "witness_contract_sha256": "0" * 64,
        },
    }
    value["workflow"]["witness_contract_sha256"] = canonical_hash(
        witness_contract_payload(value)
    )
    return value


def test_scale_ladder_witness_config_is_exact_and_hash_bound():
    value = config()
    assert validate_scale_ladder_witness_config(value) is value
    changed = copy.deepcopy(value)
    changed["data"]["expected_documents"] = 4095
    with pytest.raises(ScaleLadderWitnessError):
        validate_scale_ladder_witness_config(changed)

