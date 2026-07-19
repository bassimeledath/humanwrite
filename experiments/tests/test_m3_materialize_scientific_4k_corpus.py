from __future__ import annotations

import json

import pytest

from experiments.m3.materialize_scientific_4k_corpus import (
    M3MaterializationError,
    file_sha256,
    load_bound_jsonl,
)


def test_bound_jsonl_loader_requires_exact_bytes(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"fingerprint": "a" * 64}) + "\n", encoding="utf-8")
    digest = file_sha256(path)
    assert load_bound_jsonl(path, digest) == [{"fingerprint": "a" * 64}]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(M3MaterializationError, match="hash mismatch"):
        load_bound_jsonl(path, digest)


def test_bound_jsonl_loader_rejects_non_sha(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(M3MaterializationError, match="SHA-256"):
        load_bound_jsonl(path, "not-a-sha")
