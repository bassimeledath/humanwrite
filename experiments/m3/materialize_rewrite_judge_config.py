"""Materialize the SHA-bound blinded judge config for the M3 rewrite screen."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from data.m3_rewrite_judge import (
    DIMENSIONS,
    MODELS,
    PROTOCOL,
    RANDOMIZATION,
    RESPONSE_CONTRACT,
)


DEFAULT_OUTPUT = Path("configs/m3/m3_rewrite_pairwise_judge_v1.yaml")
ROOT_URI = "modal-volume://humanwrite-checkpoints/data/m3-rewriting-14b-v1"


def _sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("artifact SHA must be lowercase SHA-256")
    return value


def config(
    *, panel_sha256: str, sft_sha256: str, treatment_sha256: str
) -> dict[str, Any]:
    return {
        "artifact_schema": PROTOCOL,
        "run": {
            "comparison_id": "M3-rewriting-14b-4k-pairwise-judge-v1",
            "arm": "HUMANWRITE14-vs-SFT14-two-family-blinded-judge",
            "budget_class": "promo",
            "task_kind": "rewrite_judging",
        },
        "compute": {"gpus": 1, "timeout_min": 120},
        "data": {
            "panel_uri": f"{ROOT_URI}/fresh-rewrite-eval-panel-256-v1.jsonl",
            "panel_sha256": _sha(panel_sha256),
            "panel_records": 256,
            "sft_uri": f"{ROOT_URI}/evaluation/sft14-outputs-256-v1.jsonl",
            "sft_sha256": _sha(sft_sha256),
            "treatment_uri": f"{ROOT_URI}/evaluation/humanwrite14-outputs-256-v1.jsonl",
            "treatment_sha256": _sha(treatment_sha256),
            "output_uri": f"{ROOT_URI}/evaluation/pairwise-judge-results-v1.jsonl",
            "manifest_uri": f"{ROOT_URI}/evaluation/pairwise-judge-manifest-v1.json",
        },
        "judge": {
            "protocol": PROTOCOL,
            "models": list(MODELS),
            "dimensions": dict(DIMENSIONS),
            "randomization": dict(RANDOMIZATION),
            "temperature": 0.0,
            "max_completion_tokens": 512,
            "response_contract": RESPONSE_CONTRACT,
            "concurrency": 16,
            "retry_attempts": 3,
            "max_cost_usd": 10.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--sft-sha256", required=True)
    parser.add_argument("--treatment-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = config(
        panel_sha256=args.panel_sha256,
        sft_sha256=args.sft_sha256,
        treatment_sha256=args.treatment_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
