"""Fixed-code contract for Qwen3-32B line-level FineWeb cleaning."""
from __future__ import annotations

from typing import Any


class CleaningContractError(ValueError):
    pass


def numbered_cleaning_prompt(
    text: str,
    *,
    min_word_count: int | None = None,
    max_word_count: int | None = None,
) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    numbered = "\n".join(f"{index + 1}: {line}" for index, line in enumerate(lines))
    length_instruction = ""
    if min_word_count is not None and max_word_count is not None:
        length_instruction = (
            f" Select a coherent main-prose excerpt totaling {min_word_count} to "
            f"{max_word_count} words when the document contains enough suitable prose. "
            "Prefer complete paragraphs and a natural stopping point; never retain boilerplate "
            "merely to satisfy the length range. Return an empty list if no qualifying excerpt exists."
        )
    return (
        "Identify the lines that belong to the main high-quality writing sample. "
        "Keep substantive article, essay, blog, or news prose. Remove navigation, "
        "addresses and contact blocks, cookie/privacy text, subscription prompts, "
        "commerce or review widgets, isolated image captions, unrelated recommendations, "
        "and incomplete boilerplate. Do not rewrite, summarize, merge, or reorder lines."
        + length_instruction
        + " "
        "Return only a JSON object containing kept_line_numbers, a strictly increasing "
        "list of the original 1-indexed line numbers.\n\nDOCUMENT LINES:\n" + numbered
    )


def cleaning_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "fineweb_line_cleaning",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "kept_line_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                },
                "required": ["kept_line_numbers"],
                "additionalProperties": False,
            },
        },
    }


def apply_line_selection(value: Any, *, source_text: str) -> str:
    if not isinstance(value, dict) or set(value) != {"kept_line_numbers"}:
        raise CleaningContractError("cleaning response must contain only kept_line_numbers")
    selected = value["kept_line_numbers"]
    if not isinstance(selected, list) or not selected:
        raise CleaningContractError("kept_line_numbers must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in selected):
        raise CleaningContractError("kept_line_numbers must contain integers")
    if selected != sorted(set(selected)):
        raise CleaningContractError("kept_line_numbers must be unique and increasing")
    lines = source_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if selected[0] < 1 or selected[-1] > len(lines):
        raise CleaningContractError("kept_line_numbers contains an out-of-range line")
    cleaned = "\n".join(lines[index - 1] for index in selected).strip()
    if not cleaned:
        raise CleaningContractError("cleaned document is empty")
    return cleaned
