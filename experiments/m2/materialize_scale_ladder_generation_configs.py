"""Bind completed 4K confirmation checkpoints into held-out generation configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.m2.generate_lower_variance import (
    GENERATION_SCHEMA,
    GENERATION_STEP,
    canonical_hash,
    decoding_policy_payload,
    generation_contract_payload,
    validate_generation_config,
)
from infra.backend.volume_paths import checkpoint_volume_path


COMPARISON_ID = "M2-scale-ladder-4b-4096-generation-v1"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _checkpoint_block(status: dict, *, expected_arm: str) -> dict:
    if (
        status.get("status") != "completed"
        or status.get("comparison") != "M2-scale-ladder-4b-4096-v1"
        or status.get("workflow_step") != "train_lower_variance"
        or status.get("arm_executed_arm") != expected_arm
    ):
        raise ValueError("completed lower-variance status does not bind the expected 4K arm")
    return {
        "path": status["arm_checkpoint_dir_path"],
        "manifest_sha256": status["arm_checkpoint_manifest_sha256"],
        "adapter_model_sha256": status["arm_adapter_model_sha256"],
        "arm": expected_arm,
        "method_contract_sha256": status["arm_method_contract_sha256"],
    }


def _prompt_block(prompt_status: dict) -> dict:
    if (
        prompt_status.get("status") != "completed"
        or prompt_status.get("records_failed") != 0
        or prompt_status.get("records_processed") != 3
    ):
        raise ValueError("prompt-brief status does not bind the completed 128-prompt panel")
    return {
        "prompt_briefs_path": str(
            checkpoint_volume_path(str(prompt_status["output_uri"]))
        ),
        "prompt_briefs_sha256": prompt_status["output_sha256"],
        "prompt_format": "USER:\n{brief}\nASSISTANT:",
        "prompt_schema_version": "dft.full-brief.tokens.v1",
        "prompt_serializer_sha256": "1f92174518dfac375abbbbcf4ceba0659b726cabb215e0561a9fbffc4036b4a1",
    }


def _base_config(*, prompt_status: dict) -> dict:
    return {
        "artifact_schema": GENERATION_SCHEMA,
        "run": {
            "comparison_id": COMPARISON_ID,
            "arm": "",
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
        "checkpoint": {},
        "data": _prompt_block(prompt_status),
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
            "protocol_version": GENERATION_SCHEMA,
            "step": GENERATION_STEP,
            "generation_contract_sha256": "",
            "decoding_policy_sha256": "",
        },
    }


def _finalize(config: dict) -> dict:
    config["workflow"]["generation_contract_sha256"] = canonical_hash(
        generation_contract_payload(config)
    )
    config["workflow"]["decoding_policy_sha256"] = canonical_hash(
        decoding_policy_payload(config)
    )
    validate_generation_config(config)
    return config


def materialize(
    *,
    sft_status_path: Path,
    mmd_status_path: Path,
    prompt_status_path: Path,
    output_dir: Path,
) -> dict:
    sft_status = _load_json(sft_status_path)
    mmd_status = _load_json(mmd_status_path)
    prompt_status = _load_json(prompt_status_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"comparison_id": COMPARISON_ID, "configs": {}}
    for arm_id, status in (
        ("SFT", sft_status),
        ("MMD_WITNESS", mmd_status),
    ):
        config = _base_config(prompt_status=prompt_status)
        config["checkpoint"] = _checkpoint_block(status, expected_arm=arm_id)
        config["run"]["arm"] = f"{arm_id}-generation"
        _finalize(config)
        filename = f"m2_scale_ladder_4b_4096_{arm_id.casefold()}_generation_v1.yaml"
        path = output_dir / filename
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result["configs"][arm_id] = {
            "path": str(path),
            "sha256": canonical_hash(config),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-status", type=Path, required=True)
    parser.add_argument("--mmd-status", type=Path, required=True)
    parser.add_argument("--prompt-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                sft_status_path=args.sft_status,
                mmd_status_path=args.mmd_status,
                prompt_status_path=args.prompt_status,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
