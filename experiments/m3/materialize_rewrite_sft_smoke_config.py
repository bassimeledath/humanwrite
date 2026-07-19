"""Bind a completed rewrite-task artifact into the frozen 14B H100 smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import yaml

from experiments.m2.representation import canonical_hash
from experiments.m3.rewrite_sft_smoke import (
    M3_REWRITE_SFT_SMOKE_SCHEMA,
    M3_REWRITE_SFT_SMOKE_STEP,
    method_contract_payload,
    validate_m3_rewrite_sft_smoke_config,
)


SOURCE_PATH = "/checkpoints/data/m2-scale-ladder-v1/train-briefs-4096-token-normalized-v1.jsonl"
SOURCE_SHA256 = "723ebf559a4139c49454f5898a0e51120cdf424bd3cd12e39466c6758d25217b"
REWRITE_PATH = "/checkpoints/data/m3-rewriting-14b-v1/rewrite-tasks-96-smoke-v2.jsonl"
DEFAULT_OUTPUT = Path("configs/m3/m3_rewrite_sft_14b_mechanical_smoke_v1.yaml")


def build_config(rewrite_tasks_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", rewrite_tasks_sha256):
        raise ValueError("rewrite task SHA must be lowercase SHA-256")
    config: dict[str, Any] = {
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
            "source_briefs_path": SOURCE_PATH,
            "source_briefs_sha256": SOURCE_SHA256,
            "rewrite_tasks_path": REWRITE_PATH,
            "rewrite_tasks_sha256": rewrite_tasks_sha256,
            "source_records": 128,
            "rewrite_records": 96,
        },
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
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
            "max_prompt_tokens": 640,
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
    return validate_m3_rewrite_sft_smoke_config(config)


def materialize(rewrite_tasks_sha256: str, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config = build_config(rewrite_tasks_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite-tasks-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = materialize(args.rewrite_tasks_sha256, args.output)
    print(canonical_hash(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
