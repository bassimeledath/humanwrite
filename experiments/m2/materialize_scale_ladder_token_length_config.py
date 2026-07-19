"""Materialize the immutable 4K token-length normalization job."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from experiments.m2.lower_variance_train import BASE_MODEL, BASE_REVISION, canonical_hash
from experiments.m2.materialize_lower_variance_full_configs import INITIAL_ADAPTER
from experiments.m2.scale_ladder_token_lengths import (
    TOKEN_LENGTH_SCHEMA, TOKEN_LENGTH_STEP, token_length_contract_payload,
    validate_token_length_config,
)


def materialize(output: Path) -> dict:
    config = {
        "artifact_schema": TOKEN_LENGTH_SCHEMA,
        "run": {
            "comparison_id": "M2-scale-ladder-token-lengths-4096-v1",
            "arm": "exact-token-lengths-4096",
            "budget_class": "smoke",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 0,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 20},
        "model": {"base": BASE_MODEL, "revision": BASE_REVISION, "torch_dtype": "bfloat16"},
        "initial_adapter": dict(INITIAL_ADAPTER),
        "data": {
            "source_path": "/checkpoints/data/m2-scale-ladder-v1/scale-train-prefixes/clean-train-4096.jsonl",
            "source_sha256": "3c1c4340a97f68d58a873b4085ac7bc307c8f4593ffd6840aabea976bf219eef",
            "brief_path": "/checkpoints/data/m2-scale-ladder-v1/train-briefs-4096.jsonl",
            "brief_sha256": "0c35745e5a352a63fef17bee246e2c1822cf54a609e0c41e05327754db135d47",
            "output_path": "/checkpoints/data/m2-scale-ladder-v1/train-briefs-4096-token-normalized-v1.jsonl",
            "manifest_path": "/checkpoints/data/m2-scale-ladder-v1/token-length-normalization-v1.json",
            "expected_documents": 4096,
        },
        "runtime": {"transformers_version": "4.57.6", "tokenizer_local_files_only": True},
        "workflow": {
            "protocol_version": TOKEN_LENGTH_SCHEMA,
            "step": TOKEN_LENGTH_STEP,
            "contract_sha256": "0" * 64,
        },
    }
    config["workflow"]["contract_sha256"] = canonical_hash(token_length_contract_payload(config))
    validate_token_length_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.output)


if __name__ == "__main__":
    main()
