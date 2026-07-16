from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from data.pilot_synthesis_configs import PilotSynthesisConfigError, build_configs


def _config(tmp_path: Path, *, train_count=256) -> Path:
    manifest = {
        "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
        "source": {"revision": "a" * 40},
        "counts": {"train_count": train_count, "dev_count": 64},
        "train": {
            "uri": "modal-volume://humanwrite-checkpoints/data/pilot/train.jsonl",
            "sha256": "b" * 64,
            "split_hash": "c" * 64,
        },
        "dev": {
            "uri": "modal-volume://humanwrite-checkpoints/data/pilot/dev.jsonl",
            "sha256": "d" * 64,
            "split_hash": "e" * 64,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "source_manifest_path": str(manifest_path),
        "output_dir": str(tmp_path / "configs"),
        "model": "openai/gpt-5-mini",
        "timeout_min": 120,
        "train_max_cost_usd": 5.0,
        "dev_max_cost_usd": 2.0,
        "train_output_uri": "modal-volume://humanwrite-checkpoints/data/pilot/train-briefs.jsonl",
        "dev_output_uri": "modal-volume://humanwrite-checkpoints/data/pilot/dev-briefs.jsonl",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_build_configs_binds_each_source_split_and_cost_cap(tmp_path):
    result = build_configs(_config(tmp_path))
    assert result["splits"]["train"]["record_count"] == 256
    assert result["splits"]["dev"]["max_cost_usd"] == 2.0
    train = yaml.safe_load((tmp_path / "configs/m1_realdata_pilot_briefs_train_v1.yaml").read_text())
    dev = yaml.safe_load((tmp_path / "configs/m1_realdata_pilot_briefs_dev_v1.yaml").read_text())
    assert train["data"]["input_sha256"] == "b" * 64
    assert train["data"]["expected_empty_outline_count"] == 64
    assert dev["data"]["expected_empty_outline_count"] == 16
    assert train["api"]["model"] == "openai/gpt-5-mini"


def test_build_configs_rejects_missing_source_cardinality(tmp_path):
    with pytest.raises(PilotSynthesisConfigError, match="complete train metadata"):
        build_configs(_config(tmp_path, train_count=0))
