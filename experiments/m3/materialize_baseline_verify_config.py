"""Materialize the SHA-bound M3 baseline-draft verification config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


OUTPUT = Path("configs/m3/m3_baseline_verify_819_v1.yaml")
CANDIDATE_URI = (
    "modal-volume://humanwrite-checkpoints/data/m3-rewriting-14b-v1/"
    "baseline-draft-candidates-4096-v1.jsonl"
)
VERIFIED_URI = (
    "modal-volume://humanwrite-checkpoints/data/m3-rewriting-14b-v1/"
    "baseline-draft-rewrites-819-v1.jsonl"
)


def config(candidate_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
        raise ValueError("candidate SHA must be lowercase SHA-256")
    return {
        "run": {
            "comparison_id": "M3-rewriting-14b-4096-scientific-screen-v1",
            "arm": "base-draft-verification",
            "budget_class": "promo",
            "task_kind": "rewrite_synthesis",
        },
        "compute": {"gpus": 1, "timeout_min": 480},
        "data": {
            "input_uri": CANDIDATE_URI,
            "output_uri": VERIFIED_URI,
            "input_sha256": candidate_sha256,
            "max_records": 819,
            "target_records": 819,
        },
        "api": {
            "protocol": "humanwrite.m3.baseline_draft_verification.v1",
            "verifier_model": "qwen/qwen3-32b",
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
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = config(args.candidate_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
