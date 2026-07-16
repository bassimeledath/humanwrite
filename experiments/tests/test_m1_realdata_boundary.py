from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.m1.contracts import M1ConfigError, file_sha256
from experiments.m1.workflow import _load_fixed_manifest, _load_training_records


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _split_hash(rows: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(row["fingerprint"] for row in rows)).encode()
    ).hexdigest()


def _config(tmp_path: Path) -> tuple[dict, Path]:
    train = [
        {"fingerprint": f"train-{index:03}", "split": "train", "user_prompt": "p", "completion": "c"}
        for index in range(256)
    ]
    dev = [
        {"fingerprint": f"dev-{index:03}", "split": "dev", "user_prompt": "p", "completion": "c"}
        for index in range(64)
    ]
    train_path, dev_path = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train_sha, dev_sha = _write_jsonl(train_path, train), _write_jsonl(dev_path, dev)
    manifest = {
        "artifact_schema": "dftr.realdata_pilot_fixed_inputs.v1",
        "train_count": 256,
        "dev_count": 64,
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "train_briefs_sha256": train_sha,
        "dev_briefs_sha256": dev_sha,
        "train_split_hash": _split_hash(train),
        "dev_split_hash": _split_hash(dev),
        "prompt_format": "USER: {user_prompt}\nASSISTANT:",
        "max_input_tokens": 1024,
        "max_new_tokens": 384,
        "training_seeds": [11, 29, 47],
    }
    manifest_path = tmp_path / "fixed.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "workflow": {
            "protocol_version": "m1.realdata-pilot.v1",
            "fixed_manifest": str(manifest_path),
            "fixed_manifest_sha256": file_sha256(manifest_path),
        },
        "data": {
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "train_split_hash": manifest["train_split_hash"],
            "dev_split_hash": manifest["dev_split_hash"],
        },
    }
    return config, train_path


def test_realdata_protocol_binds_manifest_and_brief_bytes(tmp_path):
    config, _ = _config(tmp_path)
    fixed = _load_fixed_manifest(config)
    train, dev = _load_training_records(config, fixed)
    assert len(train) == 256
    assert len(dev) == 64


def test_realdata_protocol_rejects_manifest_hash_mismatch(tmp_path):
    config, _ = _config(tmp_path)
    config["workflow"]["fixed_manifest_sha256"] = "0" * 64
    with pytest.raises(M1ConfigError, match="manifest SHA-256"):
        _load_fixed_manifest(config)


def test_realdata_protocol_rejects_mutated_brief_bytes(tmp_path):
    config, train_path = _config(tmp_path)
    fixed = _load_fixed_manifest(config)
    train_path.write_text(train_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(M1ConfigError, match="briefs SHA-256"):
        _load_training_records(config, fixed)
