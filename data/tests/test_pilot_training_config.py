from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from data.pilot_training_config import PilotTrainingConfigError, build_training_config


def _config(tmp_path: Path, *, train_count=256) -> Path:
    validation = {
        "artifact_schema": "dftr.realdata_pilot_validation.v1",
        "source_manifest_sha256": "a" * 64,
        "source": {
            "train": {"count": train_count, "split_hash": "b" * 64},
            "dev": {"count": 64, "split_hash": "c" * 64},
        },
        "briefs": {
            "train": {"count": train_count, "source_split_hash": "b" * 64, "briefs_sha256": "d" * 64},
            "dev": {"count": 64, "source_split_hash": "c" * 64, "briefs_sha256": "e" * 64},
        },
    }
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    config = {
        "validation_path": str(validation_path),
        "fixed_manifest_path": str(tmp_path / "fixed.json"),
        "training_config_path": str(tmp_path / "train.yaml"),
        "train_briefs_volume_path": "/checkpoints/data/pilot/train-briefs.jsonl",
        "dev_briefs_volume_path": "/checkpoints/data/pilot/dev-briefs.jsonl",
        "model_revision": "f" * 40,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_build_training_config_binds_validation_manifest_and_model(tmp_path):
    result = build_training_config(_config(tmp_path))
    fixed = json.loads((tmp_path / "fixed.json").read_text())
    training = yaml.safe_load((tmp_path / "train.yaml").read_text())
    assert result["validated_train_count"] == 256
    assert fixed["train_briefs_sha256"] == "d" * 64
    assert training["workflow"]["fixed_manifest_sha256"] == result["fixed_manifest_sha256"]
    assert training["model"]["base"] == "Qwen/Qwen3-1.7B"
    assert training["run"]["seeds"] == [11, 29, 47]


def test_build_training_config_rejects_nonpilot_cardinality(tmp_path):
    with pytest.raises(PilotTrainingConfigError, match="256 records"):
        build_training_config(_config(tmp_path, train_count=255))
