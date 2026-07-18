"""Bind the completed baseline panel into the three full 1,024-brief configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.m2.lower_variance_train import (
    ARM_IDS,
    BASE_MODEL,
    BASE_REVISION,
    EPOCH_SCHEDULE,
    FULL_BRIEF_SCHEMA,
    FULL_BRIEF_SERIALIZER_SHA256,
    GENERATION_CONTRACT,
    LOWER_VARIANCE_SCHEMA,
    LOWER_VARIANCE_STEP,
    canonical_hash,
    method_contract_payload,
    validate_lower_variance_config,
)


ANCHOR_PATH = "/checkpoints/data/m2-lower-variance-v1/train-briefs-1024.jsonl"
ANCHOR_SHA256 = "419f927ac52cfd2ee6a4420638a14730ebce045a7191dce0314508c0356bc632"
INITIAL_ADAPTER = {
    "path": "/checkpoints/runs/dftr-1784216516-91130dd3/seed-11",
    "adapter_model_sha256": "b7b590ca0d40b8b51951d44beb2e7928fccabbd6a2a7290f47e927da0fb81178",
    "adapter_config_sha256": "9a72a9527c48cc2acf703f40047f1a6dec59e92fdc7021d59523ba5ed6fb965c",
    "file_manifest_sha256": "bd9843c196f6685c3e3c83f829726b09e3c87188d07ccbe66613616cd032d643",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(
    witness_manifest_path: Path,
    vocabulary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    witness = json.loads(witness_manifest_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    if (
        witness.get("artifact_schema")
        != "dftr.m2.lower_variance_baseline_witness.v1"
        or witness.get("scientific_role") != "training_only_not_evaluation"
        or witness.get("documents") != 1024
        or witness.get("briefs_sha256") != ANCHOR_SHA256
        or witness.get("generation_contract") != GENERATION_CONTRACT
        or witness.get("generation_contract_sha256") != canonical_hash(GENERATION_CONTRACT)
    ):
        raise ValueError("baseline witness manifest does not match the frozen full screen")
    output_path = str(witness.get("output_path") or "")
    output_sha256 = str(witness.get("output_sha256") or "")
    if (
        not output_path.startswith("/checkpoints/")
        or len(output_sha256) != 64
        or any(character not in "0123456789abcdef" for character in output_sha256)
    ):
        raise ValueError("baseline witness output binding is invalid")
    token_ids = vocabulary.get("frequent_token_ids")
    if (
        vocabulary.get("artifact_schema") != "dftr.m2.frequent_token_vocabulary.v1"
        or vocabulary.get("source_sha256") != ANCHOR_SHA256
        or not isinstance(token_ids, list)
        or len(token_ids) != 512
    ):
        raise ValueError("frequent-token vocabulary is not the frozen 512-token artifact")

    token_coefficient = 0.2
    base: dict[str, Any] = {
        "artifact_schema": LOWER_VARIANCE_SCHEMA,
        "run": {
            "comparison_id": "M2-lower-variance-4b-full1024-v1",
            "arm": "SFT-vs-TOKEN_MOMENT-vs-MMD_WITNESS",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 11,
        },
        "compute": {"gpu": "H100", "gpus": 1, "timeout_min": 120},
        "model": {
            "base": BASE_MODEL,
            "revision": BASE_REVISION,
            "torch_dtype": "bfloat16",
        },
        "initial_adapter": dict(INITIAL_ADAPTER),
        "data": {
            "anchor_path": ANCHOR_PATH,
            "anchor_sha256": ANCHOR_SHA256,
            "witness_generated_path": output_path,
            "witness_generated_sha256": output_sha256,
            "witness_generation_contract_sha256": canonical_hash(GENERATION_CONTRACT),
            "completion_field": "completion",
            "generated_text_field": "generated_completion",
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
            "role": "lower_variance_training_only_not_measurement_v3",
            "batch_size": 4,
            "max_tokens": 256,
        },
        "objectives": {
            "token_moments": {
                "coefficient": token_coefficient,
                "first_moment_weight": 1.0,
                "second_moment_weight": 0.5,
                "frequent_token_ids": token_ids,
                "vocabulary_source_sha256": _file_sha256(vocabulary_path),
            },
            "mmd_witness": {
                "bandwidths": [0.5, 1.0, 2.0],
                "temperature": 0.7,
                "weighting": "softmax_mean_one.v1",
                "human_self_kernel": "leave_one_out",
            },
        },
        "generation": dict(GENERATION_CONTRACT),
        "runtime": {
            "torch_version": "2.13.0+cu130",
            "transformers_version": "4.57.6",
            "peft_version": "0.19.1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "training": {
            "steps": 512,
            "batch_size": 2,
            "learning_rate": 0.00001,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "max_input_tokens": 1024,
            "checkpoint_every": 64,
            "schedule": EPOCH_SCHEDULE,
        },
        "arms": [
            {
                "id": "SFT",
                "sft_weighting": "uniform",
                "token_moment_coefficient": 0.0,
            },
            {
                "id": "TOKEN_MOMENT",
                "sft_weighting": "uniform",
                "token_moment_coefficient": token_coefficient,
            },
            {
                "id": "MMD_WITNESS",
                "sft_weighting": "mmd_witness",
                "token_moment_coefficient": 0.0,
            },
        ],
        "resume": {arm_id: None for arm_id in ARM_IDS},
        "execution": {"arm": "SFT"},
        "workflow": {
            "protocol_version": LOWER_VARIANCE_SCHEMA,
            "step": LOWER_VARIANCE_STEP,
            "method_contract_sha256": "0" * 64,
        },
    }
    base["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(base)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "comparison_id": base["run"]["comparison_id"],
        "method_contract_sha256": base["workflow"]["method_contract_sha256"],
        "witness_generated_sha256": output_sha256,
        "configs": {},
    }
    for arm_id in ARM_IDS:
        config = json.loads(json.dumps(base))
        config["execution"]["arm"] = arm_id
        validate_lower_variance_config(config)
        path = output_dir / f"m2_lower_variance_4b_full1024_{arm_id.casefold()}_v1.yaml"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result["configs"][arm_id] = {
            "path": str(path),
            "config_sha256": canonical_hash(config),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--witness-manifest", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.witness_manifest.resolve(),
                args.vocabulary.resolve(),
                args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
