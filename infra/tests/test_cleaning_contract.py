import pytest

from backend.cleaning_contract import (
    CleaningContractError,
    apply_line_selection,
    cleaning_response_format,
    numbered_cleaning_prompt,
)


def test_cleaner_keeps_exact_original_lines_without_rewriting():
    source = "HOME\nA first paragraph.\nA second paragraph.\nSubscribe"
    cleaned = apply_line_selection(
        {"kept_line_numbers": [2, 3]}, source_text=source
    )
    assert cleaned == "A first paragraph.\nA second paragraph."
    prompt = numbered_cleaning_prompt(source)
    assert "1: HOME" in prompt and "4: Subscribe" in prompt
    assert cleaning_response_format()["json_schema"]["strict"] is True


@pytest.mark.parametrize("selection", [[3, 2], [2, 2], [0], [5], [], [True]])
def test_cleaner_rejects_invalid_line_selections(selection):
    with pytest.raises(CleaningContractError):
        apply_line_selection(
            {"kept_line_numbers": selection}, source_text="one\ntwo\nthree\nfour"
        )
