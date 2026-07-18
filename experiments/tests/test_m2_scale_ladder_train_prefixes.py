from __future__ import annotations

import copy

import pytest

from experiments.m2.scale_ladder_train_prefixes import (
    SCALE_LADDER_TRAIN_PREFIX_SCHEMA,
    SCALE_LADDER_TRAIN_PREFIX_STEP,
    ScaleLadderTrainPrefixError,
    canonical_hash,
    train_prefix_contract_payload,
    validate_scale_ladder_train_prefix_config,
)


SHA = "a" * 64


def _config() -> dict:
    config = {
        "artifact_schema": SCALE_LADDER_TRAIN_PREFIX_SCHEMA,
        "run": {
            "comparison_id": "M2-scale-ladder-freeze-train-prefixes-v1",
            "arm": "freeze-scale-train-prefixes",
            "budget_class": "smoke",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 0,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 20},
        "data": {
            "raw_train_uri": "modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/raw-train-pool-26000.jsonl",
            "raw_train_sha256": SHA,
            "clean_train_uri": "modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/clean-train-16384.jsonl",
            "source_manifest_uri": "modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/raw-source-manifest.json",
            "output_dir_uri": "modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/scale-train-prefixes",
            "expected_clean_records": 16384,
        },
        "workflow": {
            "protocol_version": SCALE_LADDER_TRAIN_PREFIX_SCHEMA,
            "step": SCALE_LADDER_TRAIN_PREFIX_STEP,
            "train_prefix_contract_sha256": "",
        },
    }
    config["workflow"]["train_prefix_contract_sha256"] = canonical_hash(
        train_prefix_contract_payload(config)
    )
    return config


def test_scale_ladder_train_prefix_config_is_hash_bound_and_exact():
    config = _config()
    assert validate_scale_ladder_train_prefix_config(config) is config

    wrong = copy.deepcopy(config)
    wrong["data"]["expected_clean_records"] = 4096
    wrong["workflow"]["train_prefix_contract_sha256"] = canonical_hash(
        train_prefix_contract_payload(wrong)
    )
    with pytest.raises(ScaleLadderTrainPrefixError, match="16384"):
        validate_scale_ladder_train_prefix_config(wrong)

    wrong = copy.deepcopy(config)
    wrong["workflow"]["step"] = "freeze_scale_dev_panel"
    wrong["workflow"]["train_prefix_contract_sha256"] = canonical_hash(
        train_prefix_contract_payload(wrong)
    )
    with pytest.raises(ScaleLadderTrainPrefixError, match="workflow contract"):
        validate_scale_ladder_train_prefix_config(wrong)
