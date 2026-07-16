from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PilotTrainingConfigError(ValueError):
    pass


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_training_config(config_path: str | Path) -> dict[str, Any]:
    operator_path = _resolve(config_path)
    operator = json.loads(operator_path.read_text(encoding="utf-8"))
    validation_path = _resolve(operator["validation_path"])
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("artifact_schema") != "dftr.realdata_pilot_validation.v1":
        raise PilotTrainingConfigError("unexpected pilot validation schema")
    source = validation.get("source") or {}
    briefs = validation.get("briefs") or {}
    if int(source.get("train", {}).get("count", 0)) != 256:
        raise PilotTrainingConfigError("validated training source must contain 256 records")
    if int(source.get("dev", {}).get("count", 0)) != 64:
        raise PilotTrainingConfigError("validated dev source must contain 64 records")
    for split in ("train", "dev"):
        if briefs.get(split, {}).get("source_split_hash") != source.get(split, {}).get("split_hash"):
            raise PilotTrainingConfigError(f"validated {split} source/brief split hash mismatch")
        if int(briefs.get(split, {}).get("count", 0)) != int(source[split]["count"]):
            raise PilotTrainingConfigError(f"validated {split} source/brief count mismatch")
    fixed_path = _resolve(operator["fixed_manifest_path"])
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    fixed = {
        "artifact_schema": "dftr.realdata_pilot_fixed_inputs.v1",
        "train_count": 256,
        "dev_count": 64,
        "train_path": str(operator["train_briefs_volume_path"]),
        "dev_path": str(operator["dev_briefs_volume_path"]),
        "train_briefs_sha256": str(briefs["train"]["briefs_sha256"]),
        "dev_briefs_sha256": str(briefs["dev"]["briefs_sha256"]),
        "train_split_hash": str(source["train"]["split_hash"]),
        "dev_split_hash": str(source["dev"]["split_hash"]),
        "source_manifest_sha256": str(validation["source_manifest_sha256"]),
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "prompt_format": "USER: {user_prompt}\nASSISTANT:",
        "max_input_tokens": 1024,
        "max_new_tokens": 384,
        "training_seeds": [11, 29, 47],
        "sampling_seeds": [101, 202, 303],
        "provenance_note": (
            "Validated 256/64 real-FineWeb recovery pilot; distinct from visible Tier-1 "
            "and sealed evaluator data; not promotion evidence."
        ),
    }
    fixed_path.write_text(json.dumps(fixed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fixed_sha = hashlib.sha256(fixed_path.read_bytes()).hexdigest()
    training = {
        "run": {
            "comparison_id": "M1-realdata-pilot-sft-qwen3-1p7b-v1",
            "arm": "SFT",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seeds": [11, 29, 47],
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 120},
        "model": {
            "base": "Qwen/Qwen3-1.7B",
            "revision": str(operator["model_revision"]),
            "requested_revision": "main",
            "lora": {
                "rank": 64,
                "alpha": 128,
                "dropout": 0.0,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            },
        },
        "training": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1.0,
            "learning_rate": 0.0002,
            "logging_steps": 10,
        },
        "data": {
            "train_split_hash": fixed["train_split_hash"],
            "dev_split_hash": fixed["dev_split_hash"],
            "train_path": fixed["train_path"],
            "dev_path": fixed["dev_path"],
            "prompt_format": fixed["prompt_format"],
            "max_input_tokens": fixed["max_input_tokens"],
            "max_new_tokens": fixed["max_new_tokens"],
        },
        "workflow": {
            "protocol_version": "m1.realdata-pilot.v1",
            "step": "train_sft",
            "fixed_manifest": _display(fixed_path),
            "fixed_manifest_sha256": fixed_sha,
        },
    }
    training_path = _resolve(operator["training_config_path"])
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(
        yaml.safe_dump(training, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "artifact_schema": "dftr.realdata_pilot_training_config_index.v1",
        "fixed_manifest_path": _display(fixed_path),
        "fixed_manifest_sha256": fixed_sha,
        "training_config_path": _display(training_path),
        "training_config_hash": _canonical_hash(training),
        "validated_train_count": 256,
        "validated_dev_count": 64,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the hash-bound real-data pilot SFT config")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(build_training_config(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
