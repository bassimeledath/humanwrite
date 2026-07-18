from __future__ import annotations

from backend.source_materializer import SourceMaterializationError, materialize_rows

import pytest


def _row(index: int, domain: str | None = None) -> dict:
    return {
        "id": f"row-{index}",
        "url": f"https://{domain or f'domain{index}.example'}/article",
        "text": f"Pilot document {index}.\n\n" + " ".join(f"word{offset}" for offset in range(90)),
    }


def _config() -> dict:
    return {
        "source": {
            "dataset_id": "HuggingFaceFW/fineweb",
            "dataset_config": "CC-MAIN-2024-10",
            "revision": "a" * 40,
            "split": "train",
            "files": ["data/CC-MAIN-2024-10/000_00000.parquet"],
        },
        "selection": {
            "corpus_size": 8,
            "dev_count": 2,
            "eligible_pool_size": 10,
            "max_records_per_domain": 1,
            "stream_scan_limit": 12,
            "min_word_count": 80,
            "max_word_count": 120,
            "max_non_latin_letter_ratio": 0.02,
            "selection_seed": "selection",
            "split_seed": "split",
        },
        "exclusions": {"domains": ["excluded.example"], "fingerprints": []},
        "policy": {"hidden_test_materialized": False},
    }


def test_materialization_is_deterministic_hash_bound_and_disjoint():
    config = _config()
    rows = [_row(0, "excluded.example")] + [_row(index) for index in range(1, 13)]
    first_payloads, first = materialize_rows(rows, config)
    second_payloads, second = materialize_rows(rows, config)
    assert first_payloads == second_payloads
    assert first == second
    assert first["counts"]["train_count"] == 6
    assert first["counts"]["dev_count"] == 2
    assert set(first["train"]["fingerprints"]).isdisjoint(first["dev"]["fingerprints"])
    assert "excluded.example" not in first["domains"]


def test_materialization_fails_when_domains_are_not_distinct():
    config = _config()
    with pytest.raises(SourceMaterializationError, match="distinct-domain"):
        materialize_rows([_row(index, "one.example") for index in range(12)], config)


def test_materialization_rejects_nonpositive_domain_cap():
    config = _config()
    config["selection"]["max_records_per_domain"] = 0
    with pytest.raises(SourceMaterializationError, match="must be positive"):
        materialize_rows([_row(index) for index in range(12)], config)
