from __future__ import annotations

import pytest

from backend.brief_contract import (
    BriefContractError,
    brief_response_format,
    empty_brief_quotations,
    exact_empty_outline_ids,
    prompt_repair_response_format,
    validate_brief,
    validate_repaired_user_prompt,
)


def _record(index: int) -> dict:
    return {"fingerprint": f"fingerprint-{index}"}


def _brief() -> dict:
    return {
        "user_prompt": "Write a concise update.",
        "use_case": "news",
        "style_kind": "reported",
        "style": "neutral, sourced",
        "detail_mode": "strict",
        "target_length": 120,
        "em_dashes_allowed": False,
        "outline": [
            {
                "section": "Update",
                "supported_facts": ["The pilot launched."],
                "quotations": ["The pilot launched."],
            }
        ],
    }


def test_exact_empty_outline_assignment_is_deterministic_and_exact():
    records = [_record(index) for index in range(64)]
    first = exact_empty_outline_ids(records)
    second = exact_empty_outline_ids(reversed(records))
    assert first == second
    assert len(first) == 16


def test_provider_schema_is_strict_and_tracks_outline_assignment():
    regular = brief_response_format(force_empty_outline=False)
    empty = brief_response_format(force_empty_outline=True)
    assert regular["type"] == "json_schema"
    assert regular["json_schema"]["strict"] is True
    schema = regular["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "one or more" in schema["properties"]["outline"]["description"]
    empty_outline = empty["json_schema"]["schema"]["properties"]["outline"]
    assert "empty list" in empty_outline["description"]


def test_prompt_repair_schema_and_validation_reject_meta_prompts():
    schema = prompt_repair_response_format()["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["user_prompt"]
    assert validate_repaired_user_prompt(
        {"user_prompt": "Write a factual overview of Orlando Health MyChart."},
        source_text="Orlando Health MyChart provides secure access to health information.",
    ).startswith("Write a factual")
    with pytest.raises(BriefContractError, match="meta-instructions"):
        validate_repaired_user_prompt(
            {"user_prompt": "Convert the supplied human web document into a DFT training brief."},
            source_text="A document about housing.",
        )
    with pytest.raises(BriefContractError, match="source-grounded"):
        validate_repaired_user_prompt(
            {"user_prompt": "Write a thoughtful overview of lunar geology."},
            source_text="Orlando Health MyChart provides secure access to health information.",
        )


def test_quote_recovery_preserves_every_field_except_quotations():
    original = _brief()
    recovered = empty_brief_quotations(original)
    assert recovered["outline"][0]["quotations"] == []
    expected = _brief()
    expected["outline"][0]["quotations"] = []
    assert recovered == expected
    assert original["outline"][0]["quotations"] == ["The pilot launched."]


def test_brief_contract_validates_and_forces_empty_outline():
    brief = validate_brief(
        _brief(), source_text="The pilot launched.", force_empty_outline=False
    )
    assert brief["outline"][0]["section"] == "Update"
    empty = validate_brief(
        _brief(), source_text="The pilot launched.", force_empty_outline=True
    )
    assert empty["outline"] == []


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda value: value.update(target_length="120"), "positive integer"),
        (lambda value: value.update(detail_mode="unknown"), "strict or creative"),
        (lambda value: value["outline"][0].update(quotations=["Invented quote"]), "source text"),
        (lambda value: value.update(outline=[]), "empty outline"),
    ],
)
def test_brief_contract_fails_closed(mutation, error):
    brief = _brief()
    mutation(brief)
    with pytest.raises(BriefContractError, match=error):
        validate_brief(brief, source_text="The pilot launched.", force_empty_outline=False)
