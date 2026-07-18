"""Pure contracts for assembling faithful briefs from clean human documents.

Qwen supplies the non-outline brief metadata and GPT-5-mini supplies only the
outline.  This module performs no provider calls; it validates and merges
already-returned structured values while preserving every source field.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping


QWEN_MODEL = "qwen/qwen3-32b"
OUTLINE_MODEL = "openai/gpt-5-mini"
TARGET_LENGTH_UNIT = "tokens"
MAX_TARGET_LENGTH_TOKENS = 4096
EMPTY_OUTLINE_SEED = "dftr-m2-lower-variance-empty-outline-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_QWEN_FIELDS = {
    "detail_mode",
    "document_fingerprint",
    "em_dashes_allowed",
    "style",
    "style_kind",
    "target_length",
    "target_length_unit",
    "use_case",
    "user_prompt",
}
_OUTLINE_FIELDS = {"document_fingerprint", "outline"}
_SOURCE_PROVENANCE_FIELDS = (
    "completion",
    "domain",
    "fineweb_id",
    "fingerprint",
    "source_config",
    "source_revision",
    "split",
    "url",
    "word_count",
)
_ASSEMBLED_FIELDS = (_QWEN_FIELDS - {"document_fingerprint"}) | {
    "brief_metadata_model",
    "generation_mode",
    "outline",
    "outline_model",
}
_PROMPT_META_PHRASES = (
    "dft",
    "training brief",
    "json object",
    "json schema",
    "supplied document",
    "source document",
)
_PROMPT_STOPWORDS = {
    "about",
    "article",
    "create",
    "document",
    "for",
    "from",
    "into",
    "piece",
    "that",
    "the",
    "this",
    "using",
    "with",
    "write",
}


class LowerVarianceBriefError(ValueError):
    """Raised when provider output cannot form a faithful clean-data brief."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LowerVarianceBriefError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise LowerVarianceBriefError(f"{field} must not have boundary whitespace")
    return value


def _fingerprint(value: Any, field: str) -> str:
    result = _nonempty_text(value, field)
    if _SHA256_RE.fullmatch(result) is None:
        raise LowerVarianceBriefError(f"{field} must be a lowercase SHA-256 value")
    return result


def _source_fingerprint(source: Mapping[str, Any], *, label: str) -> str:
    fingerprint = _fingerprint(source.get("fingerprint"), f"{label}.fingerprint")
    completion = _nonempty_text(source.get("completion"), f"{label}.completion")
    if _sha256_text(completion) != fingerprint:
        raise LowerVarianceBriefError(f"{label} fingerprint does not bind completion")
    return fingerprint


def _source_index(
    sources: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    indexed: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, list[str]] = {}
    for position, raw_source in enumerate(sources):
        source = dict(raw_source)
        label = f"source[{position}]"
        fingerprint = _source_fingerprint(source, label=label)
        if fingerprint in indexed:
            raise LowerVarianceBriefError("source fingerprints must be unique")
        for field in _SOURCE_PROVENANCE_FIELDS:
            if field not in source:
                raise LowerVarianceBriefError(f"{label} is missing provenance field {field}")
        split = _nonempty_text(source.get("split"), f"{label}.split")
        if split not in {"train", "dev"}:
            raise LowerVarianceBriefError("source split must be train or dev")
        collisions = sorted(_ASSEMBLED_FIELDS.intersection(source))
        if collisions:
            raise LowerVarianceBriefError(
                f"{label} already contains assembled fields: {', '.join(collisions)}"
            )
        indexed[fingerprint] = source
        split_ids.setdefault(split, []).append(fingerprint)
    if not indexed:
        raise LowerVarianceBriefError("at least one source record is required")
    return indexed, split_ids


def deterministic_empty_outline_ids(
    sources: Iterable[Mapping[str, Any]],
    *,
    seed: str = EMPTY_OUTLINE_SEED,
) -> frozenset[str]:
    """Select exactly one quarter of every source split, independent of order."""

    if not isinstance(seed, str) or not seed:
        raise LowerVarianceBriefError("empty-outline seed must be non-empty")
    _, split_ids = _source_index(sources)
    selected: set[str] = set()
    for split, fingerprints in sorted(split_ids.items()):
        if len(fingerprints) % 4:
            raise LowerVarianceBriefError(
                f"{split} source count must be divisible by four for exactly 25% empty outlines"
            )
        ranked = sorted(
            fingerprints,
            key=lambda fingerprint: (
                _sha256_text(f"{seed}:{split}:{fingerprint}"),
                fingerprint,
            ),
        )
        selected.update(ranked[: len(ranked) // 4])
    return frozenset(selected)


def qwen_metadata_response_schema() -> dict[str, Any]:
    """Strict schema for Qwen's non-outline portion of a brief."""

    properties: dict[str, Any] = {
        "document_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "user_prompt": {"type": "string"},
        "use_case": {"type": "string"},
        "style_kind": {"type": "string"},
        "style": {"type": "string"},
        "detail_mode": {"type": "string", "enum": ["strict", "creative"]},
        "target_length": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_TARGET_LENGTH_TOKENS,
        },
        "target_length_unit": {"type": "string", "const": TARGET_LENGTH_UNIT},
        "em_dashes_allowed": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


def outline_response_schema(*, force_empty_outline: bool) -> dict[str, Any]:
    """Strict schema for GPT-5-mini's outline-only response."""

    outline: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "supported_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "quotations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["section", "supported_facts", "quotations"],
            "additionalProperties": False,
        },
    }
    if force_empty_outline:
        outline.update({"maxItems": 0})
    else:
        outline.update({"minItems": 1})
    properties = {
        "document_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "outline": outline,
    }
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


def _validate_user_prompt(value: Any, *, source_text: str) -> str:
    prompt = _nonempty_text(value, "user_prompt")
    if len(prompt) > 1000:
        raise LowerVarianceBriefError("user_prompt exceeds 1000 characters")
    lowered = prompt.casefold()
    if any(phrase in lowered for phrase in _PROMPT_META_PHRASES):
        raise LowerVarianceBriefError("user_prompt contains synthesis meta-instructions")
    prompt_tokens = {
        token.casefold()
        for token in _TOKEN_RE.findall(prompt)
        if len(token) >= 4 and token.casefold() not in _PROMPT_STOPWORDS
    }
    source_tokens = {
        token.casefold() for token in _TOKEN_RE.findall(source_text) if len(token) >= 4
    }
    if not prompt_tokens or not prompt_tokens.intersection(source_tokens):
        raise LowerVarianceBriefError("user_prompt lacks source-grounded topic terms")
    return prompt


def validate_qwen_metadata(
    value: Any,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that Qwen supplied exactly the non-outline brief fields."""

    if not isinstance(value, Mapping) or set(value) != _QWEN_FIELDS:
        raise LowerVarianceBriefError(
            "Qwen metadata must contain exactly the frozen non-outline fields"
        )
    result = dict(value)
    source_fingerprint = _source_fingerprint(source, label="source")
    if _fingerprint(result["document_fingerprint"], "document_fingerprint") != source_fingerprint:
        raise LowerVarianceBriefError("Qwen metadata binds the wrong source fingerprint")
    source_text = str(source["completion"])
    result["user_prompt"] = _validate_user_prompt(
        result["user_prompt"], source_text=source_text
    )
    for field in ("use_case", "style_kind", "style"):
        result[field] = _nonempty_text(result[field], field)
    detail_mode = _nonempty_text(result["detail_mode"], "detail_mode").casefold()
    if detail_mode not in {"strict", "creative"}:
        raise LowerVarianceBriefError("detail_mode must be strict or creative")
    result["detail_mode"] = detail_mode
    target_length = result["target_length"]
    if (
        isinstance(target_length, bool)
        or not isinstance(target_length, int)
        or not 1 <= target_length <= MAX_TARGET_LENGTH_TOKENS
    ):
        raise LowerVarianceBriefError(
            f"target_length must be an integer from 1 to {MAX_TARGET_LENGTH_TOKENS}"
        )
    if result["target_length_unit"] != TARGET_LENGTH_UNIT:
        raise LowerVarianceBriefError("target_length_unit must be tokens")
    if not isinstance(result["em_dashes_allowed"], bool):
        raise LowerVarianceBriefError("em_dashes_allowed must be boolean")
    return result


def validate_outline(
    value: Any,
    *,
    source: Mapping[str, Any],
    force_empty_outline: bool,
) -> list[dict[str, Any]]:
    """Validate GPT's outline, requiring literal source support for facts/quotes."""

    if not isinstance(value, Mapping) or set(value) != _OUTLINE_FIELDS:
        raise LowerVarianceBriefError("outline response must contain only fingerprint and outline")
    source_fingerprint = _source_fingerprint(source, label="source")
    if _fingerprint(value["document_fingerprint"], "document_fingerprint") != source_fingerprint:
        raise LowerVarianceBriefError("outline response binds the wrong source fingerprint")
    outline = value["outline"]
    if not isinstance(outline, list):
        raise LowerVarianceBriefError("outline must be a list")
    if force_empty_outline:
        if outline:
            raise LowerVarianceBriefError("deterministic empty-outline record must have []")
        return []
    if not outline:
        raise LowerVarianceBriefError("non-empty-outline record requires at least one section")

    source_text = str(source["completion"])
    validated: list[dict[str, Any]] = []
    for index, raw_item in enumerate(outline):
        if not isinstance(raw_item, Mapping) or set(raw_item) != {
            "section",
            "supported_facts",
            "quotations",
        }:
            raise LowerVarianceBriefError(f"outline[{index}] has invalid fields")
        section = _nonempty_text(raw_item["section"], f"outline[{index}].section")
        raw_facts = raw_item["supported_facts"]
        raw_quotes = raw_item["quotations"]
        if not isinstance(raw_facts, list) or not raw_facts:
            raise LowerVarianceBriefError(
                f"outline[{index}].supported_facts must be non-empty"
            )
        if not isinstance(raw_quotes, list):
            raise LowerVarianceBriefError(f"outline[{index}].quotations must be a list")
        facts = [
            _nonempty_text(fact, f"outline[{index}].supported_facts")
            for fact in raw_facts
        ]
        quotes = [
            _nonempty_text(quote, f"outline[{index}].quotations")
            for quote in raw_quotes
        ]
        if len(facts) != len(set(facts)) or len(quotes) != len(set(quotes)):
            raise LowerVarianceBriefError(f"outline[{index}] contains duplicate grounding")
        if any(fact not in source_text for fact in facts):
            raise LowerVarianceBriefError(
                f"outline[{index}] supported fact is not an exact source substring"
            )
        if any(quote not in source_text for quote in quotes):
            raise LowerVarianceBriefError(
                f"outline[{index}] quotation is not an exact source substring"
            )
        validated.append(
            {"section": section, "supported_facts": facts, "quotations": quotes}
        )
    return validated


def merge_brief(
    *,
    source: Mapping[str, Any],
    qwen_metadata: Any,
    outline_response: Any,
    force_empty_outline: bool,
    qwen_model: str = QWEN_MODEL,
    outline_model: str = OUTLINE_MODEL,
) -> dict[str, Any]:
    """Merge one source with separately validated provider responsibilities."""

    if qwen_model != QWEN_MODEL:
        raise LowerVarianceBriefError(f"brief metadata model must be {QWEN_MODEL}")
    if outline_model != OUTLINE_MODEL:
        raise LowerVarianceBriefError(f"outline model must be {OUTLINE_MODEL}")
    clean_source = dict(source)
    _source_index([clean_source])
    metadata = validate_qwen_metadata(qwen_metadata, source=clean_source)
    outline = validate_outline(
        outline_response,
        source=clean_source,
        force_empty_outline=force_empty_outline,
    )
    metadata.pop("document_fingerprint")
    return {
        **clean_source,
        **metadata,
        "outline": outline,
        "generation_mode": "generate",
        "brief_metadata_model": qwen_model,
        "outline_model": outline_model,
    }


def validate_assembled_brief(
    value: Any,
    *,
    source: Mapping[str, Any],
    force_empty_outline: bool,
) -> dict[str, Any]:
    """Revalidate one persisted assembled brief against its clean source."""

    if not isinstance(value, Mapping):
        raise LowerVarianceBriefError("assembled brief must be an object")
    row = dict(value)
    clean_source = dict(source)
    _source_index([clean_source])
    expected_fields = set(clean_source) | _ASSEMBLED_FIELDS
    if set(row) != expected_fields:
        raise LowerVarianceBriefError("assembled brief has missing or unexpected fields")
    for field, source_value in clean_source.items():
        if row[field] != source_value:
            raise LowerVarianceBriefError(
                f"assembled brief changed source provenance field {field}"
            )
    if row["generation_mode"] != "generate":
        raise LowerVarianceBriefError("generation_mode must be generate")
    qwen_metadata = {
        field: row[field]
        for field in _QWEN_FIELDS
        if field != "document_fingerprint"
    }
    qwen_metadata["document_fingerprint"] = clean_source["fingerprint"]
    outline_response = {
        "document_fingerprint": clean_source["fingerprint"],
        "outline": row["outline"],
    }
    expected = merge_brief(
        source=clean_source,
        qwen_metadata=qwen_metadata,
        outline_response=outline_response,
        force_empty_outline=force_empty_outline,
        qwen_model=row["brief_metadata_model"],
        outline_model=row["outline_model"],
    )
    if row != expected:
        raise LowerVarianceBriefError("assembled brief does not match canonical merge")
    return expected


def _response_index(
    values: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise LowerVarianceBriefError(f"{label}[{position}] must be an object")
        fingerprint = _fingerprint(
            value.get("document_fingerprint"),
            f"{label}[{position}].document_fingerprint",
        )
        if fingerprint in indexed:
            raise LowerVarianceBriefError(f"{label} source fingerprints must be unique")
        indexed[fingerprint] = value
    return indexed


def assemble_briefs(
    *,
    sources: Iterable[Mapping[str, Any]],
    qwen_metadata_rows: Iterable[Mapping[str, Any]],
    outline_rows: Iterable[Mapping[str, Any]],
    seed: str = EMPTY_OUTLINE_SEED,
    qwen_model: str = QWEN_MODEL,
    outline_model: str = OUTLINE_MODEL,
) -> tuple[dict[str, Any], ...]:
    """Validate complete provider coverage and assemble deterministic briefs."""

    source_by_id, _ = _source_index(sources)
    empty_ids = deterministic_empty_outline_ids(source_by_id.values(), seed=seed)
    qwen_by_id = _response_index(qwen_metadata_rows, label="Qwen metadata")
    outline_by_id = _response_index(outline_rows, label="outline responses")
    expected_ids = set(source_by_id)
    if set(qwen_by_id) != expected_ids:
        raise LowerVarianceBriefError("Qwen metadata/source fingerprint set mismatch")
    if set(outline_by_id) != expected_ids:
        raise LowerVarianceBriefError("outline/source fingerprint set mismatch")

    assembled = [
        merge_brief(
            source=source_by_id[fingerprint],
            qwen_metadata=qwen_by_id[fingerprint],
            outline_response=outline_by_id[fingerprint],
            force_empty_outline=fingerprint in empty_ids,
            qwen_model=qwen_model,
            outline_model=outline_model,
        )
        for fingerprint in sorted(expected_ids)
    ]
    observed_empty = {row["fingerprint"] for row in assembled if not row["outline"]}
    if observed_empty != set(empty_ids):
        raise LowerVarianceBriefError("assembled empty-outline assignment mismatch")
    return tuple(assembled)


__all__ = [
    "EMPTY_OUTLINE_SEED",
    "LowerVarianceBriefError",
    "MAX_TARGET_LENGTH_TOKENS",
    "OUTLINE_MODEL",
    "QWEN_MODEL",
    "TARGET_LENGTH_UNIT",
    "assemble_briefs",
    "deterministic_empty_outline_ids",
    "merge_brief",
    "outline_response_schema",
    "qwen_metadata_response_schema",
    "validate_outline",
    "validate_assembled_brief",
    "validate_qwen_metadata",
]
