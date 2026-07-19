from __future__ import annotations

import hashlib

from data.lower_variance_briefs import (
    OUTLINE_MODEL,
    QWEN_MODEL,
    deterministic_empty_outline_ids,
    merge_brief,
)
from experiments.m2.scale_ladder_token_lengths import normalize_rows


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _source(index: int) -> dict:
    completion = f"Orchard report {index}. Workers picked twelve crates and stored the apples overnight."
    return {
        "completion": completion,
        "domain": f"source-{index}.example",
        "fineweb_id": f"source-{index}",
        "fingerprint": _sha(completion),
        "source_config": "CC-MAIN-2024-10",
        "source_revision": "a" * 40,
        "split": "train",
        "url": f"https://source-{index}.example",
        "word_count": len(completion.split()),
        "source_word_count": len(completion.split()),
        "source_fingerprint": _sha(f"raw-{index}"),
        "cleaning_model": QWEN_MODEL,
        "cleaning_mode": "ordered_original_line_subset.v1",
    }


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return list(range(len(text.split()) + 3))


def test_4k_normalization_replaces_every_guess_and_preserves_empty_outline_assignment():
    sources = [_source(index) for index in range(4096)]
    empty_ids = deterministic_empty_outline_ids(sources)
    briefs = []
    for source in sources:
        empty = source["fingerprint"] in empty_ids
        briefs.append(
            merge_brief(
                source=source,
                qwen_metadata={
                    "document_fingerprint": source["fingerprint"],
                    "user_prompt": "Write an orchard report.",
                    "use_case": "news",
                    "style_kind": "reported",
                    "style": "clear and factual",
                    "detail_mode": "strict",
                    "target_length": 999,
                    "target_length_unit": "tokens",
                    "em_dashes_allowed": False,
                },
                outline_response={
                    "document_fingerprint": source["fingerprint"],
                    "outline": [] if empty else [{
                        "section": "Harvest",
                        "supported_facts": ["Workers picked twelve crates"],
                        "quotations": [],
                    }],
                },
                force_empty_outline=empty,
                qwen_model=QWEN_MODEL,
                outline_model=OUTLINE_MODEL,
            )
        )
    normalized, stats = normalize_rows(sources, briefs, FakeTokenizer())
    assert stats["rows"] == 4096
    assert stats["changed_rows"] == 4096
    assert stats["all_lengths_exact"] is True
    assert all(row["target_length"] == len(row["completion"].split()) + 3 for row in normalized)
    assert {row["fingerprint"] for row in normalized if not row["outline"]} == empty_ids
