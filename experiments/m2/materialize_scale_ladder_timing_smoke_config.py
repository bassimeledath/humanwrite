"""Derive a bounded 64-step MMD-witness timing smoke from a frozen full config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.m2.lower_variance_train import (
    LOWER_VARIANCE_STEP,
    canonical_hash,
    method_contract_payload,
    validate_lower_variance_config,
)


def materialize(source_path: Path, output_path: Path) -> dict:
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if (config.get("execution") or {}).get("arm") != "MMD_WITNESS":
        raise ValueError("timing smoke must derive from the frozen MMD_WITNESS config")
    config["run"].update(
        comparison_id="M2-scale-ladder-4b-4096-timing-smoke-l40s-v1",
        budget_class="smoke",
    )
    config["compute"] = {"gpu": "L40S", "gpus": 1, "timeout_min": 20}
    config["training"].update(
        steps=64,
        checkpoint_every=64,
        schedule="python_random_sample_without_replacement.v1",
    )
    config["resume"] = {"SFT": None, "MMD_WITNESS": None}
    config["workflow"]["step"] = LOWER_VARIANCE_STEP
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    validate_lower_variance_config(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)


if __name__ == "__main__":
    main()
