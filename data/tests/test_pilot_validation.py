from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from data.pilot_validation import PilotValidationError, validate_pilot
from data.pipeline import split_hash
from infra.backend.brief_contract import exact_empty_outline_ids


def _canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(_canonical(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _source(split: str, index: int) -> dict:
    completion = f"The pilot launched for domain {split}-{index}."
    return {
        "completion": completion,
        "domain": f"{split}{index}.example",
        "fineweb_id": f"{split}-{index}",
        "fingerprint": hashlib.sha256(completion.encode()).hexdigest(),
        "source_config": "CC-MAIN-2024-10",
        "source_revision": "a" * 40,
        "split": split,
        "url": f"https://{split}{index}.example/article",
        "word_count": 7,
    }


def _brief(source: dict, empty: bool) -> dict:
    row = dict(source)
    row.update(
        {
            "generation_mode": "generate",
            "user_prompt": "Write an update.",
            "use_case": "news",
            "style_kind": "reported",
            "style": "neutral",
            "detail_mode": "strict",
            "target_length": 20,
            "em_dashes_allowed": False,
            "outline": [] if empty else [
                {
                    "section": "Update",
                    "supported_facts": [source["completion"]],
                    "quotations": [source["completion"]],
                }
            ],
        }
    )
    return row


def _fixture(tmp_path: Path) -> Path:
    train = [_source("train", index) for index in range(8)]
    dev = [_source("dev", index) for index in range(4)]
    train_path, dev_path = tmp_path / "train-source.jsonl", tmp_path / "dev-source.jsonl"
    train_sha = _write_jsonl(train_path, train)
    dev_sha = _write_jsonl(dev_path, dev)
    manifest = {
        "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
        "source": {
            "dataset_id": "HuggingFaceFW/fineweb",
            "dataset_config": "CC-MAIN-2024-10",
            "revision": "a" * 40,
            "split": "train",
            "files": ["data/CC-MAIN-2024-10/000_00000.parquet"],
        },
        "counts": {"train_count": 8, "dev_count": 4, "unique_domain_count": 12},
    }
    for split, rows, sha in (("train", train, train_sha), ("dev", dev, dev_sha)):
        manifest[split] = {
            "fingerprints": [row["fingerprint"] for row in rows],
            "domains": [row["domain"] for row in rows],
            "sha256": sha,
            "split_hash": split_hash(rows),
        }
        empty = exact_empty_outline_ids(rows)
        _write_jsonl(
            tmp_path / f"{split}-briefs.jsonl",
            [_brief(row, row["fingerprint"] in empty) for row in rows],
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "source_manifest_path": str(manifest_path),
        "train_source_path": str(train_path),
        "dev_source_path": str(dev_path),
        "train_briefs_path": str(tmp_path / "train-briefs.jsonl"),
        "dev_briefs_path": str(tmp_path / "dev-briefs.jsonl"),
        "expected_source": manifest["source"],
        "excluded_fingerprints": [],
        "excluded_domains": [],
        "output_path": str(tmp_path / "validation.json"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_validate_pilot_binds_source_and_briefs(tmp_path):
    result = validate_pilot(_fixture(tmp_path))
    assert result["source"]["train"]["count"] == 8
    assert result["briefs"]["train"]["empty_outline_count"] == 2
    assert result["briefs"]["dev"]["empty_outline_count"] == 1


def test_validate_pilot_can_reuse_frozen_source_config(tmp_path):
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    source_config_path = tmp_path / "source.yaml"
    source_config_path.write_text(
        yaml.safe_dump(
            {
                "source": config.pop("expected_source"),
                "exclusions": {
                    "fingerprints": config.pop("excluded_fingerprints"),
                    "domains": config.pop("excluded_domains"),
                },
            }
        ),
        encoding="utf-8",
    )
    config["source_config_path"] = str(source_config_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert validate_pilot(config_path)["source"]["train"]["count"] == 8


def test_validate_pilot_rejects_mutated_completion(tmp_path):
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    brief_path = Path(config["train_briefs_path"])
    rows = [json.loads(line) for line in brief_path.read_text().splitlines()]
    rows[0]["completion"] = "mutated"
    _write_jsonl(brief_path, rows)
    with pytest.raises(PilotValidationError, match="changed source field"):
        validate_pilot(config_path)


def test_validate_pilot_rejects_wrong_empty_outline_assignment(tmp_path):
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    brief_path = Path(config["dev_briefs_path"])
    rows = [json.loads(line) for line in brief_path.read_text().splitlines()]
    rows[0]["outline"] = []
    _write_jsonl(brief_path, rows)
    with pytest.raises((PilotValidationError, ValueError)):
        validate_pilot(config_path)
