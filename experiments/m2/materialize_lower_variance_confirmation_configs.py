"""Bind the 128-token witness into the matched 4B confirmation configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.m2.lower_variance_train import (
    BASE_MODEL,
    BASE_REVISION,
    CONFIRMATION_GENERATION_CONTRACT,
    EPOCH_SCHEDULE,
    FULL_BRIEF_SCHEMA,
    FULL_BRIEF_SERIALIZER_SHA256,
    LOWER_VARIANCE_CONFIRMATION_SCHEMA,
    LOWER_VARIANCE_STEP,
    canonical_hash,
    method_contract_payload,
    validate_lower_variance_config,
)
from experiments.m2.materialize_lower_variance_full_configs import (
    ANCHOR_PATH,
    ANCHOR_SHA256,
    INITIAL_ADAPTER,
)


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
        != "dftr.m2.lower_variance_baseline_witness.v2"
        or witness.get("scientific_role") != "training_only_not_evaluation"
        or witness.get("documents") != 1024
        or witness.get("briefs_sha256") != ANCHOR_SHA256
        or witness.get("generation_contract") != CONFIRMATION_GENERATION_CONTRACT
        or witness.get("generation_contract_sha256")
        != canonical_hash(CONFIRMATION_GENERATION_CONTRACT)
    ):
        raise ValueError("128-token witness manifest does not match confirmation")
    output_path = str(witness.get("output_path") or "")
    output_sha256 = str(witness.get("output_sha256") or "")
    if not output_path.startswith("/checkpoints/") or len(output_sha256) != 64:
        raise ValueError("confirmation witness output binding is invalid")
    token_ids = vocabulary.get("frequent_token_ids")
    if (
        vocabulary.get("artifact_schema")
        != "dftr.m2.frequent_token_vocabulary.v1"
        or vocabulary.get("source_sha256") != ANCHOR_SHA256
        or not isinstance(token_ids, list)
        or len(token_ids) != 512
    ):
        raise ValueError("frequent-token vocabulary does not match training anchors")

    base: dict[str, Any] = {
        "artifact_schema": LOWER_VARIANCE_CONFIRMATION_SCHEMA,
        "run": {
            "comparison_id": "M2-mmd-witness-4b-confirmation-v1",
            "arm": "SFT-vs-MMD_WITNESS",
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
            "witness_generation_contract_sha256": canonical_hash(
                CONFIRMATION_GENERATION_CONTRACT
            ),
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
            "role": "lower_variance_training_only_not_measurement_v4",
            "batch_size": 4,
            "max_tokens": 256,
        },
        "objectives": {
            "token_moments": {
                "coefficient": 0.2,
                "first_moment_weight": 1.0,
                "second_moment_weight": 0.5,
                "frequent_token_ids": token_ids,
                "vocabulary_source_sha256": _file_sha256(vocabulary_path),
            },
            "mmd_witness": {
                "bandwidths": [0.5, 1.0, 2.0],
                "temperature": 0.035,
                "weighting": "softmax_mean_one.v1",
                "human_self_kernel": "leave_one_out",
            },
        },
        "generation": dict(CONFIRMATION_GENERATION_CONTRACT),
        "runtime": {
            "torch_version": "2.13.0+cu130",
            "transformers_version": "4.57.6",
            "peft_version": "0.19.1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "training": {
            "steps": 1024,
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
                "id": "MMD_WITNESS",
                "sft_weighting": "mmd_witness",
                "token_moment_coefficient": 0.0,
            },
        ],
        "resume": {"SFT": None, "MMD_WITNESS": None},
        "execution": {"arm": "SFT"},
        "workflow": {
            "protocol_version": LOWER_VARIANCE_CONFIRMATION_SCHEMA,
            "step": LOWER_VARIANCE_STEP,
            "method_contract_sha256": "0" * 64,
        },
    }
    base["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(base)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"configs": {}, "method_contract_sha256": base["workflow"]["method_contract_sha256"]}
    for arm_id in ("SFT", "MMD_WITNESS"):
        config = json.loads(json.dumps(base))
        config["execution"]["arm"] = arm_id
        validate_lower_variance_config(config)
        path = output_dir / f"m2_confirmation_4b_{arm_id.casefold()}_v1.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result["configs"][arm_id] = {
            "path": str(path),
            "sha256": canonical_hash(config),
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
            materialize(args.witness_manifest, args.vocabulary, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
