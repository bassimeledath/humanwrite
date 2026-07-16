from __future__ import annotations

import pytest

from backend.brief_contract import BriefContractError, exact_empty_outline_ids, validate_brief


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
