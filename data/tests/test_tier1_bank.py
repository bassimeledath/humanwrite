from __future__ import annotations

import json

import pytest

from data.tier1_bank import Tier1BankError, materialize, select_bank


def _row(index: int, domain: str | None = None) -> dict:
    words = " ".join(f"word{offset}" for offset in range(45))
    return {
        "id": f"row-{index}",
        "url": f"https://{domain or f'domain{index}.example'}/article",
        "text": f"Visible human document {index}. {words}.",
    }


def _config(tmp_path, *, bank_size=4, pool_size=6, scan_limit=8):
    excluded = tmp_path / "excluded.json"
    excluded.write_text(json.dumps({"fingerprints": []}), encoding="utf-8")
    config = {
        "source": {
            "dataset_id": "example/fineweb",
            "dataset_config": "snapshot",
            "revision": "a" * 40,
            "split": "train",
        },
        "selection": {
            "bank_size": bank_size,
            "eligible_pool_size": pool_size,
            "max_non_latin_letter_ratio": 0.02,
            "max_word_count": 80,
            "min_word_count": 40,
            "require_distinct_domains": True,
            "seed_label": "fixed-seed",
            "stream_scan_limit": scan_limit,
        },
        "exclude_manifests": [str(excluded)],
        "output": {
            "bank_path": str(tmp_path / "bank.jsonl"),
            "manifest_path": str(tmp_path / "manifest.json"),
        },
        "policy": {"agent_visible": True, "hidden_test_materialized": False},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def test_selection_is_deterministic_unique_and_domain_diverse(tmp_path):
    _, config = _config(tmp_path)
    rows = [_row(index) for index in range(8)]
    first, counts = select_bank(rows, config)
    second, _ = select_bank(rows, config)
    assert first == second
    assert len(first) == 4
    assert len({row["fingerprint"] for row in first}) == 4
    assert len({row["domain"] for row in first}) == 4
    assert counts == {
        "bank_size": 4,
        "eligible_unique_count": 6,
        "excluded_fingerprint_count": 0,
        "scanned_count": 6,
        "unique_domain_count": 4,
    }


def test_materialize_writes_hash_bound_bank_and_manifest(tmp_path):
    config_path, _ = _config(tmp_path)
    manifest = materialize(config_path, rows=[_row(index) for index in range(8)])
    bank = tmp_path / "bank.jsonl"
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == on_disk
    assert manifest["counts"]["bank_size"] == 4
    assert len(bank.read_text(encoding="utf-8").splitlines()) == 4
    assert manifest["policy"]["hidden_test_materialized"] is False


def test_distinct_domain_requirement_fails_closed(tmp_path):
    _, config = _config(tmp_path)
    rows = [_row(index, domain="one.example") for index in range(8)]
    with pytest.raises(Tier1BankError, match="distinct-domain requirement failed"):
        select_bank(rows, config)
