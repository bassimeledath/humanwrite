"""Bind a completed 4K brief artifact into the frozen baseline-witness job."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.m2.lower_variance_train import (
    BASE_MODEL,
    BASE_REVISION,
    CONFIRMATION_GENERATION_CONTRACT,
    FULL_BRIEF_SCHEMA,
    FULL_BRIEF_SERIALIZER_SHA256,
    canonical_hash,
)
from experiments.m2.materialize_lower_variance_full_configs import INITIAL_ADAPTER
from experiments.m2.scale_ladder_witness import (
    SCALE_LADDER_WITNESS_SCHEMA,
    SCALE_LADDER_WITNESS_STEP,
    validate_scale_ladder_witness_config,
    witness_contract_payload,
)


def materialize(briefs_sha256: str, output_path: Path) -> dict:
    config = {
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
        "initial_adapter": dict(INITIAL_ADAPTER),
        "data": {
            "briefs_path": "/checkpoints/data/m2-scale-ladder-v1/train-briefs-4096-token-normalized-v1.jsonl",
            "briefs_sha256": briefs_sha256,
            "expected_documents": 4096,
            "output_dir": "/checkpoints/data/m2-scale-ladder-v1/witness-4096-v1",
            "prompt_format": "USER:\n{brief}\nASSISTANT:",
            "prompt_schema_version": FULL_BRIEF_SCHEMA,
            "prompt_serializer_sha256": FULL_BRIEF_SERIALIZER_SHA256,
            "generation_batch_size": 8,
            "prompt_max_length": 1024,
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
    config["workflow"]["witness_contract_sha256"] = canonical_hash(
        witness_contract_payload(config)
    )
    validate_scale_ladder_witness_config(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--briefs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.briefs_sha256, args.output)


if __name__ == "__main__":
    main()
