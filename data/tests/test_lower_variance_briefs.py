from __future__ import annotations

import hashlib

import pytest

from data.lower_variance_briefs import (
    MAX_TARGET_LENGTH_TOKENS,
    OUTLINE_MODEL,
    QWEN_MODEL,
    TARGET_LENGTH_UNIT,
    LowerVarianceBriefError,
    assemble_briefs,
    deterministic_empty_outline_ids,
    merge_brief,
    outline_response_schema,
    qwen_metadata_response_schema,
    validate_assembled_brief,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(index: int, split: str = "train") -> dict:
    completion = (
        f"{split.title()} orchard harvest report {index}\n\n"
        f"Workers picked twelve crates on day {index}.\n"
        "Cool storage kept the apples firm overnight."
    )
    return {
        "completion": completion,
        "domain": f"{split}-{index}.example",
        "fineweb_id": f"{split}-{index}",
        "fingerprint": _sha(completion),
        "source_config": "CC-MAIN-2024-10",
        "source_revision": "a" * 40,
        "split": split,
        "url": f"https://{split}-{index}.example/report",
        "word_count": len(completion.split()),
        "source_fingerprint": _sha(f"raw-{split}-{index}"),
        "cleaning_model": QWEN_MODEL,
        "cleaning_mode": "ordered_original_line_subset.v1",
    }


def _qwen(source: dict) -> dict:
    return {
        "document_fingerprint": source["fingerprint"],
        "user_prompt": "Write a concise orchard harvest report.",
        "use_case": "news",
        "style_kind": "reported",
        "style": "clear, factual",
        "detail_mode": "strict",
        "target_length": 96,
        "target_length_unit": TARGET_LENGTH_UNIT,
        "em_dashes_allowed": False,
    }


def _outline(source: dict, *, empty: bool) -> dict:
    return {
        "document_fingerprint": source["fingerprint"],
        "outline": []
        if empty
        else [
            {
                "section": "Harvest results",
                "supported_facts": [
                    f"Workers picked twelve crates on day {source['fineweb_id'].split('-')[-1]}."
                ],
                "quotations": ["Cool storage kept the apples firm overnight."],
            }
        ],
    }


def _responses(sources: list[dict]):
    empty_ids = deterministic_empty_outline_ids(sources)
    return (
        [_qwen(source) for source in sources],
        [_outline(source, empty=source["fingerprint"] in empty_ids) for source in sources],
        empty_ids,
    )


def test_schema_separates_provider_responsibilities_and_token_unit():
    qwen = qwen_metadata_response_schema()
    outline = outline_response_schema(force_empty_outline=False)
    empty = outline_response_schema(force_empty_outline=True)

    assert "outline" not in qwen["properties"]
    assert set(outline["properties"]) == {"document_fingerprint", "outline"}
    assert qwen["properties"]["target_length_unit"]["const"] == "tokens"
    assert qwen["properties"]["target_length"]["maximum"] == MAX_TARGET_LENGTH_TOKENS
    assert outline["properties"]["outline"]["minItems"] == 1
    assert empty["properties"]["outline"]["maxItems"] == 0


def test_assembly_is_order_independent_exactly_quarter_empty_and_preserves_source():
    sources = [_source(index, "train") for index in range(8)] + [
        _source(index, "dev") for index in range(4)
    ]
    qwen_rows, outline_rows, empty_ids = _responses(sources)
    first = assemble_briefs(
        sources=reversed(sources),
        qwen_metadata_rows=qwen_rows,
        outline_rows=reversed(outline_rows),
    )
    second = assemble_briefs(
        sources=sources,
        qwen_metadata_rows=reversed(qwen_rows),
        outline_rows=outline_rows,
    )

    assert first == second
    assert len(empty_ids) == 3
    assert {row["fingerprint"] for row in first if not row["outline"]} == set(empty_ids)
    assert sum(not row["outline"] for row in first if row["split"] == "train") == 2
    assert sum(not row["outline"] for row in first if row["split"] == "dev") == 1
    source_by_id = {row["fingerprint"]: row for row in sources}
    for row in first:
        source = source_by_id[row["fingerprint"]]
        assert all(row[field] == value for field, value in source.items())
        assert row["brief_metadata_model"] == QWEN_MODEL
        assert row["outline_model"] == OUTLINE_MODEL
        assert row["target_length_unit"] == "tokens"
        assert row["generation_mode"] == "generate"


def test_exact_quarter_requires_each_split_count_divisible_by_four():
    with pytest.raises(LowerVarianceBriefError, match="divisible by four"):
        deterministic_empty_outline_ids([_source(index) for index in range(5)])


def test_qwen_cannot_supply_outline_or_omit_explicit_length_unit():
    source = _source(0)
    metadata = {**_qwen(source), "outline": []}
    with pytest.raises(LowerVarianceBriefError, match="exactly the frozen"):
        merge_brief(
            source=source,
            qwen_metadata=metadata,
            outline_response=_outline(source, empty=True),
            force_empty_outline=True,
        )

    metadata = _qwen(source)
    metadata["target_length_unit"] = "words"
    with pytest.raises(LowerVarianceBriefError, match="must be tokens"):
        merge_brief(
            source=source,
            qwen_metadata=metadata,
            outline_response=_outline(source, empty=True),
            force_empty_outline=True,
        )


@pytest.mark.parametrize("field", ["supported_facts", "quotations"])
def test_outline_grounding_requires_exact_source_substrings(field):
    source = _source(0)
    outline = _outline(source, empty=False)
    outline["outline"][0][field] = ["This claim was invented by the provider."]
    with pytest.raises(LowerVarianceBriefError, match="exact source substring"):
        merge_brief(
            source=source,
            qwen_metadata=_qwen(source),
            outline_response=outline,
            force_empty_outline=False,
        )


def test_deterministic_empty_assignment_is_enforced_not_repaired():
    sources = [_source(index) for index in range(4)]
    qwen_rows, outline_rows, empty_ids = _responses(sources)
    empty_id = next(iter(empty_ids))
    wrong = next(row for row in outline_rows if row["document_fingerprint"] == empty_id)
    source = next(row for row in sources if row["fingerprint"] == empty_id)
    wrong["outline"] = _outline(source, empty=False)["outline"]
    with pytest.raises(LowerVarianceBriefError, match="must have"):
        assemble_briefs(
            sources=sources,
            qwen_metadata_rows=qwen_rows,
            outline_rows=outline_rows,
        )


def test_provider_models_are_frozen():
    source = _source(0)
    with pytest.raises(LowerVarianceBriefError, match="brief metadata model"):
        merge_brief(
            source=source,
            qwen_metadata=_qwen(source),
            outline_response=_outline(source, empty=True),
            force_empty_outline=True,
            qwen_model="openai/gpt-5-mini",
        )
    with pytest.raises(LowerVarianceBriefError, match="outline model"):
        merge_brief(
            source=source,
            qwen_metadata=_qwen(source),
            outline_response=_outline(source, empty=True),
            force_empty_outline=True,
            outline_model="another/model",
        )


def test_source_provenance_and_provider_coverage_fail_closed():
    sources = [_source(index) for index in range(4)]
    qwen_rows, outline_rows, _ = _responses(sources)
    with pytest.raises(LowerVarianceBriefError, match="set mismatch"):
        assemble_briefs(
            sources=sources,
            qwen_metadata_rows=qwen_rows[:-1],
            outline_rows=outline_rows,
        )

    changed_source = dict(sources[0], completion=sources[0]["completion"] + " changed")
    with pytest.raises(LowerVarianceBriefError, match="does not bind completion"):
        merge_brief(
            source=changed_source,
            qwen_metadata=qwen_rows[0],
            outline_response=outline_rows[0],
            force_empty_outline=False,
        )


def test_persisted_brief_revalidation_detects_provenance_tampering():
    sources = [_source(index) for index in range(4)]
    qwen_rows, outline_rows, empty_ids = _responses(sources)
    rows = assemble_briefs(
        sources=sources,
        qwen_metadata_rows=qwen_rows,
        outline_rows=outline_rows,
    )
    row = rows[0]
    source = next(source for source in sources if source["fingerprint"] == row["fingerprint"])
    assert validate_assembled_brief(
        row,
        source=source,
        force_empty_outline=row["fingerprint"] in empty_ids,
    ) == row

    tampered = {**row, "split": "dev"}
    with pytest.raises(LowerVarianceBriefError, match="changed source provenance"):
        validate_assembled_brief(
            tampered,
            source=source,
            force_empty_outline=row["fingerprint"] in empty_ids,
        )

def test_user_prompt_must_be_document_grounded_and_not_meta():
    source = _source(0)
    for prompt in (
        "Write about semiconductor manufacturing.",
        "Convert the supplied document into a training brief about orchard harvest.",
    ):
        metadata = {**_qwen(source), "user_prompt": prompt}
        with pytest.raises(LowerVarianceBriefError, match="user_prompt"):
            merge_brief(
                source=source,
                qwen_metadata=metadata,
                outline_response=_outline(source, empty=True),
                force_empty_outline=True,
            )
