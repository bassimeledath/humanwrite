"""Materialize the SHA-bound provider config for fresh M3 evaluation inputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


OUTPUT = Path("configs/m3/m3_fresh_eval_rewrite_inputs_224_v1.yaml")
INPUT_URI = (
    "modal-volume://humanwrite-checkpoints/data/m3-rewriting-14b-v1/"
    "fresh-eval-clean-pool-640.jsonl"
)
OUTPUT_URI = (
    "modal-volume://humanwrite-checkpoints/data/m3-rewriting-14b-v1/"
    "fresh-eval-api-rewrite-inputs-224-v1.jsonl"
)


def config(input_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("input SHA must be lowercase SHA-256")
    return {
        "run": {
            "comparison_id": "M3-rewriting-14b-fresh-eval-inputs-v1",
            "arm": "cross-provider-public-eval-input-construction",
            "budget_class": "promo",
            "task_kind": "rewrite_synthesis",
        },
        "compute": {"gpus": 1, "timeout_min": 480},
        "data": {
            "input_uri": INPUT_URI,
            "output_uri": OUTPUT_URI,
            "input_sha256": input_sha256,
            "max_records": 640,
            "target_records": 224,
        },
        "api": {
            "protocol": "humanwrite.m3.eval_rewrite_inputs.v1",
            "generator_models": [
                "google/gemini-3.1-flash-lite",
                "anthropic/claude-haiku-4.5",
            ],
            "verifier_by_generator": {
                "google/gemini-3.1-flash-lite": "qwen/qwen3-32b",
                "anthropic/claude-haiku-4.5": "qwen/qwen3-32b",
            },
            "max_cost_usd": 6.0,
            "max_attempts": 4,
            "semantic_similarity_min": 0.90,
            "concurrency": 16,
        },
        "tokenizer": {
            "model": "Qwen/Qwen3-14B",
            "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = config(args.input_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
