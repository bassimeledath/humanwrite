from __future__ import annotations

import hashlib
import json

import pytest

from data.pilot_corpus import PilotCorpusError, materialize, select_corpus


def _row(index: int, *, domain: str | None = None) -> dict:
    words = " ".join(f"word{offset}" for offset in range(90))
    return {
        "id": f"row-{index}",
        "url": f"https://{domain or f'domain{index}.example'}/article",
        "text": f"Real pilot document {index}. {words}.",
    }


def _config(tmp_path, *, corpus_size=8, dev_count=2, pool_size=10, scan_limit=12):
    excluded = tmp_path / "excluded.json"
    excluded.write_text(
        json.dumps({"fingerprints": [], "domains": ["excluded.example"]}),
        encoding="utf-8",
    )
    config = {
        "source": {
            "dataset_id": "example/fineweb",
            "dataset_config": "snapshot",
            "revision": "a" * 40,
            "split": "train",
        },
        "selection": {
            "corpus_size": corpus_size,
            "dev_count": dev_count,
            "eligible_pool_size": pool_size,
            "max_non_latin_letter_ratio": 0.02,
            "max_word_count": 120,
            "min_word_count": 80,
            "selection_seed": "pilot-selection-v1",
            "split_seed": "pilot-split-v1",
            "stream_scan_limit": scan_limit,
        },
        "exclude_manifests": [str(excluded)],
        "output": {
            "train_source_path": str(tmp_path / "train.jsonl"),
            "dev_source_path": str(tmp_path / "dev.jsonl"),
            "manifest_path": str(tmp_path / "manifest.json"),
        },
        "policy": {"hidden_test_materialized": False},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def test_selection_is_deterministic_and_split_disjoint(tmp_path):
    _, config = _config(tmp_path)
    rows = [_row(index) for index in range(12)]
    first, counts = select_corpus(rows, config)
    second, _ = select_corpus(rows, config)
    assert first == second
    assert counts["train_count"] == 6
    assert counts["dev_count"] == 2
    train_fingerprints = {row["fingerprint"] for row in first["train"]}
    dev_fingerprints = {row["fingerprint"] for row in first["dev"]}
    assert train_fingerprints.isdisjoint(dev_fingerprints)
    assert len({row["domain"] for split in first.values() for row in split}) == 8


def test_excluded_domain_is_never_selected(tmp_path):
    _, config = _config(tmp_path, pool_size=9, scan_limit=12)
    rows = [_row(0, domain="excluded.example")] + [_row(index) for index in range(1, 12)]
    splits, _ = select_corpus(rows, config)
    assert "excluded.example" not in {row["domain"] for split in splits.values() for row in split}


def test_materialize_writes_hash_bound_outputs(tmp_path):
    config_path, _ = _config(tmp_path)
    manifest = materialize(config_path, rows=[_row(index) for index in range(12)])
    for split in ("train", "dev"):
        path = tmp_path / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[split]["sha256"]
    assert manifest["counts"]["unique_domain_count"] == 8
    assert manifest["policy"]["hidden_test_materialized"] is False


def test_distinct_domain_shortage_fails_closed(tmp_path):
    _, config = _config(tmp_path)
    rows = [_row(index, domain="one.example") for index in range(12)]
    with pytest.raises(PilotCorpusError, match="distinct-domain records"):
        select_corpus(rows, config)
