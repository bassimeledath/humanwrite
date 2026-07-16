from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.m1.contracts import M1ConfigError, file_sha256
from experiments.m1.workflow import (
    FULL_BRIEF_SCHEMA,
    _directional_dev_subset,
    _load_fixed_manifest,
    _load_sampler_grid,
    _load_training_records,
    _render_prompt,
)


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


def test_adherence_protocol_requires_adherence_manifest_schema(tmp_path):
    config, _ = _config(tmp_path)
    manifest_path = Path(config["workflow"]["fixed_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_schema"] = "dftr.realdata_adherence_fixed_inputs.v1"
    manifest["prompt_schema_version"] = FULL_BRIEF_SCHEMA
    manifest["prompt_format"] = "USER:\n{brief}\nASSISTANT:"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    config["workflow"]["protocol_version"] = "m1.realdata-adherence.v1"
    config["workflow"]["fixed_manifest_sha256"] = file_sha256(manifest_path)
    assert _load_fixed_manifest(config)["prompt_schema_version"] == FULL_BRIEF_SCHEMA

    manifest["artifact_schema"] = "dftr.realdata_pilot_fixed_inputs.v1"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    config["workflow"]["fixed_manifest_sha256"] = file_sha256(manifest_path)
    with pytest.raises(M1ConfigError, match="manifest schema"):
        _load_fixed_manifest(config)


def test_directional_pilot_allows_only_one_default_sampler(tmp_path):
    grid = {
        "default_sampler_id": "default_t1.0_p1.0",
        "points": [
            {"id": "default_t1.0_p1.0", "temperature": 1.0, "top_p": 1.0, "do_sample": True}
        ],
    }
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid), encoding="utf-8")
    config = {
        "workflow": {"protocol_version": "m1.realdata-pilot.v1"},
        "sampling": {"stage": "directional_default", "sampler_grid": str(grid_path)},
    }
    assert _load_sampler_grid(config)["default_sampler_id"] == "default_t1.0_p1.0"

    config["sampling"]["stage"] = "full"
    with pytest.raises(M1ConfigError, match="exactly 5"):
        _load_sampler_grid(config)


def test_directional_pilot_binds_exact_16_record_subset():
    records = [{"fingerprint": f"dev-{index:02}"} for index in range(64)]
    fingerprints = [record["fingerprint"] for record in records[:16]]
    subset_hash = hashlib.sha256("\n".join(sorted(fingerprints)).encode()).hexdigest()
    config = {
        "sampling": {
            "dev_subset_fingerprints": fingerprints,
            "dev_subset_hash": subset_hash,
        }
    }
    assert _directional_dev_subset(config, records) == records[:16]
    config["sampling"]["dev_subset_hash"] = "0" * 64
    with pytest.raises(M1ConfigError, match="subset hash"):
        _directional_dev_subset(config, records)


def test_full_brief_prompt_includes_every_conditioning_field():
    record = {
        "user_prompt": "Write an update about harbor expansion.",
        "use_case": "news",
        "style_kind": "reported",
        "style": "neutral, sourced",
        "detail_mode": "strict",
        "target_length": 600,
        "em_dashes_allowed": False,
        "outline": [{"section": "Approval", "supported_facts": ["Approved in June"], "quotations": []}],
    }
    rendered = _render_prompt(record, "USER:\n{brief}\nASSISTANT:", FULL_BRIEF_SCHEMA)
    for expected in (
        record["user_prompt"], "news", "reported", "neutral, sourced", "strict",
        "600 words", "Em dashes allowed: no", "Approved in June",
    ):
        assert expected in rendered
    mutated = dict(record, outline=[])
    assert _render_prompt(mutated, "{brief}", FULL_BRIEF_SCHEMA) != rendered


def test_adherence_sampler_allows_one_directional_point(tmp_path):
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps({
        "points": [{"id": "default", "temperature": 1.0, "top_p": 1.0}],
    }), encoding="utf-8")
    config = {
        "workflow": {"protocol_version": "m1.realdata-adherence.v1"},
        "sampling": {"stage": "adherence_directional", "sampler_grid": str(grid_path)},
    }
    assert len(_load_sampler_grid(config)["points"]) == 1
