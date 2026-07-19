"""Materialize the SHA-bound cleaner config for the fresh M3 public eval pool."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


OUTPUT = Path("configs/m3/m3_fresh_eval_clean_640_v1.yaml")
INPUT_URI = (
    "modal-volume://humanwrite-checkpoints/data/m3-rewriting-14b-v1/"
    "fresh-eval-raw-pool-1200.jsonl"
)
OUTPUT_URI = (
    "modal-volume://humanwrite-checkpoints/data/m3-rewriting-14b-v1/"
    "fresh-eval-clean-pool-640.jsonl"
)


def config(input_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("input SHA must be lowercase SHA-256")
    return {
        "run": {
            "comparison_id": "M3-rewriting-14b-fresh-eval-clean-v1",
            "arm": "qwen32b-line-cleaned-public-eval-pool",
            "budget_class": "promo",
            "task_kind": "document_cleaning",
        },
        "compute": {"gpus": 1, "timeout_min": 480},
        "data": {
            "input_uri": INPUT_URI,
            "output_uri": OUTPUT_URI,
            "input_sha256": input_sha256,
            "max_records": 1200,
            "target_records": 640,
        },
        "api": {"model": "qwen/qwen3-32b", "max_cost_usd": 3.0},
        "quality": {"min_word_count": 80, "max_word_count": 220},
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
