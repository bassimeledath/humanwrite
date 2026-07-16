from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from data.pilot_sampler_config import PilotSamplerConfigError, build_sampler_config


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _config(tmp_path: Path, *, distinct_adapters=True) -> Path:
    dev = [{"fingerprint": f"dev-{index:02}", "split": "dev"} for index in range(64)]
    dev_path = tmp_path / "dev.jsonl"
    dev_sha = _write_jsonl(dev_path, dev)
    fixed = {
        "artifact_schema": "dftr.realdata_pilot_fixed_inputs.v1",
        "train_path": "/checkpoints/data/pilot/train.jsonl",
        "dev_path": "/checkpoints/data/pilot/dev.jsonl",
        "train_split_hash": "a" * 64,
        "dev_split_hash": "b" * 64,
        "dev_briefs_sha256": dev_sha,
        "prompt_format": "USER: {user_prompt}\nASSISTANT:",
        "max_input_tokens": 1024,
        "max_new_tokens": 384,
    }
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
    checkpoints = {
        "protocol_version": "m1.checkpoints.v1",
        "model_base": "Qwen/Qwen3-1.7B",
        "model_revision": "c" * 40,
        "checkpoints": [
            {
                "seed": seed,
                "train_tokens": 100,
                "checkpoint_files": {
                    "adapter_model.safetensors": (str(index) * 64 if distinct_adapters else "0" * 64)
                },
            }
            for index, seed in enumerate((11, 29, 47), 1)
        ],
    }
    checkpoint_path = tmp_path / "checkpoints.json"
    checkpoint_path.write_text(json.dumps(checkpoints), encoding="utf-8")
    operator = {
        "fixed_manifest_path": str(fixed_path),
        "dev_briefs_local_path": str(dev_path),
        "checkpoint_manifest_local_path": str(checkpoint_path),
        "checkpoint_manifest_volume_path": "/checkpoints/runs/run/checkpoints_manifest.json",
        "model_revision": "c" * 40,
        "subset_manifest_path": str(tmp_path / "subset.json"),
        "sampler_config_path": str(tmp_path / "sampler.yaml"),
    }
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(operator), encoding="utf-8")
    return config_path


def test_build_directional_sampler_config_is_fixed_and_bounded(tmp_path):
    result = build_sampler_config(_config(tmp_path))
    config = yaml.safe_load((tmp_path / "sampler.yaml").read_text())
    subset = json.loads((tmp_path / "subset.json").read_text())
    assert result["expected_documents"] == 144
    assert subset["count"] == 16
    assert config["sampling"]["stage"] == "directional_default"
    assert len(config["sampling"]["dev_subset_fingerprints"]) == 16
    assert config["sampling"]["seeds"] == [101, 202, 303]


def test_build_directional_sampler_rejects_identical_adapters(tmp_path):
    with pytest.raises(PilotSamplerConfigError, match="must differ"):
        build_sampler_config(_config(tmp_path, distinct_adapters=False))
