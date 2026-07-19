from __future__ import annotations

from collections import Counter
import hashlib

from data.m3_eval_panel import CATEGORY_COUNTS, API_CATEGORIES, eval_panel_manifest


def source(index: int) -> dict:
    completion = f"Document {index} reports the result on 2026-07-{index % 28 + 1:02d}. " + "word " * 90
    return {
        "fingerprint": hashlib.sha256(f"eval-{index}".encode()).hexdigest(),
        "source_fingerprint": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
        "domain": f"domain-{index}.example",
        "completion": completion,
    }


def test_fresh_eval_manifest_has_exact_categories_and_provider_balance() -> None:
    rows = eval_panel_manifest([source(index) for index in range(640)])
    assert len(rows) == 256
    assert Counter(row["category"] for row in rows) == CATEGORY_COUNTS
    for category in API_CATEGORIES:
        providers = Counter(
            row["generator_model"] for row in rows if row["category"] == category
        )
        assert sorted(providers.values()) == [CATEGORY_COUNTS[category] // 2] * 2
