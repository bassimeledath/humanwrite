"""Fixed-code validation for privileged brief synthesis outputs."""
from __future__ import annotations

import hashlib
from typing import Any, Iterable


class BriefContractError(ValueError):
    pass


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("fingerprint") or record.get("fineweb_id") or "").strip()


def exact_empty_outline_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    ids = [record_id(record) for record in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise BriefContractError("source records require unique fingerprint/fineweb_id values")
    ranked = sorted(
        ids,
        key=lambda value: (
            hashlib.sha256(f"empty-outline:{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    return set(ranked[: len(ranked) // 4])


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BriefContractError(f"{field} must be a non-empty string")
    return value.strip()


def validate_brief(
    value: Any,
    *,
    source_text: str,
    force_empty_outline: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BriefContractError("brief response must be a JSON object")
    result = dict(value)
    for field in ("user_prompt", "use_case", "style_kind", "style"):
        result[field] = _nonempty_string(result.get(field), field)
    detail_mode = _nonempty_string(result.get("detail_mode"), "detail_mode").casefold()
    if detail_mode not in {"strict", "creative"}:
        raise BriefContractError("detail_mode must be strict or creative")
    result["detail_mode"] = detail_mode
    target_length = result.get("target_length")
    if isinstance(target_length, bool) or not isinstance(target_length, int) or target_length <= 0:
        raise BriefContractError("target_length must be a positive integer")
    if not isinstance(result.get("em_dashes_allowed"), bool):
        raise BriefContractError("em_dashes_allowed must be boolean")
    outline = result.get("outline")
    if not isinstance(outline, list):
        raise BriefContractError("outline must be a list")
    if force_empty_outline:
        result["outline"] = []
        return result
    if not outline:
        raise BriefContractError("non-empty-outline record received an empty outline")
    validated_outline = []
    for index, item in enumerate(outline):
        if not isinstance(item, dict):
            raise BriefContractError(f"outline[{index}] must be an object")
        section = _nonempty_string(item.get("section"), f"outline[{index}].section")
        facts = item.get("supported_facts")
        quotations = item.get("quotations")
        if not isinstance(facts, list) or not facts:
            raise BriefContractError(f"outline[{index}].supported_facts must be non-empty")
        if not isinstance(quotations, list):
            raise BriefContractError(f"outline[{index}].quotations must be a list")
        clean_facts = [
            _nonempty_string(fact, f"outline[{index}].supported_facts") for fact in facts
        ]
        clean_quotes = [
            _nonempty_string(quote, f"outline[{index}].quotations") for quote in quotations
        ]
        for quote in clean_quotes:
            if quote not in source_text:
                raise BriefContractError(f"outline[{index}] quotation is not in source text")
        validated_outline.append(
            {"section": section, "supported_facts": clean_facts, "quotations": clean_quotes}
        )
    result["outline"] = validated_outline
    return result
