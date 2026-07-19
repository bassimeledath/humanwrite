"""Frozen M3 prompt rendering and mechanical-smoke corpus assembly."""

from __future__ import annotations

import re
from typing import Any, Callable

from data.rewrite_tasks import (
    PROTOCOL as REWRITE_PROTOCOL,
    RewriteTaskError,
    render_rewrite_prompt,
    rewrite_source_records,
    validate_rewrite_task,
)


TRAINING_TASK_PROTOCOL = "humanwrite.m3.training_tasks.v1"


class M3TrainingTaskError(ValueError):
    pass


def render_generation_prompt(source: dict[str, Any]) -> str:
    required = (
        "user_prompt",
        "use_case",
        "style_kind",
        "style",
        "detail_mode",
        "target_length",
        "target_length_unit",
        "em_dashes_allowed",
        "outline",
    )
    missing = [field for field in required if field not in source]
    if missing:
        raise M3TrainingTaskError(f"generation brief missing fields: {missing}")
    if source["target_length_unit"] != "tokens":
        raise M3TrainingTaskError("generation target length must use tokens")
    if type(source["target_length"]) is not int or source["target_length"] <= 0:
        raise M3TrainingTaskError("generation target length must be a positive integer")
    if not isinstance(source["outline"], list):
        raise M3TrainingTaskError("generation outline must be a list")
    outline = "\n".join(f"- {item}" for item in source["outline"]) or "(none provided)"
    return (
        "MODE: GENERATE\n"
        f"WRITING REQUEST: {source['user_prompt']}\n"
        f"USE CASE: {source['use_case']}\n"
        f"STYLE CATEGORY: {source['style_kind']}\n"
        f"STYLE: {source['style']}\n"
        f"DETAIL MODE: {source['detail_mode']}\n"
        f"TARGET LENGTH: approximately {source['target_length']} tokens\n"
        f"EM DASHES ALLOWED: {'yes' if source['em_dashes_allowed'] else 'no'}\n"
        "GROUNDING OUTLINE:\n"
        f"{outline}\n\n"
        "RETURN: only the requested text."
    )


def _training_row(
    *, source: dict[str, Any], task_mode: str, origin: str, prompt: str
) -> dict[str, Any]:
    fingerprint = str(source.get("fingerprint") or "")
    completion = str(source.get("completion") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or not completion or not prompt.strip():
        raise M3TrainingTaskError("training task identity, prompt, and completion are required")
    if "�" in completion or "�" in prompt:
        raise M3TrainingTaskError("replacement characters are forbidden in M3 training tasks")
    return {
        "artifact_schema": TRAINING_TASK_PROTOCOL,
        "task_mode": task_mode,
        "origin": origin,
        "fingerprint": fingerprint,
        "source_fingerprint": str(source.get("source_fingerprint") or fingerprint),
        "prompt": prompt.strip(),
        "completion": completion,
        "target_length": int(source["target_length"]),
        "target_length_unit": "tokens",
    }


def assemble_mechanical_smoke_corpus(
    sources: list[dict[str, Any]],
    rewrite_tasks: list[dict[str, Any]],
    *,
    token_counter: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Build the 96-rewrite/32-generate wiring smoke without scientific claims."""
    if len(sources) != 128:
        raise M3TrainingTaskError("mechanical smoke requires exactly 128 frozen sources")
    expected_rewrites = rewrite_source_records(sources)
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    expected_ids = {str(row["fingerprint"]) for row in expected_rewrites}
    rewrite_by_id: dict[str, dict[str, Any]] = {}
    for row in rewrite_tasks:
        fingerprint = str(row.get("fingerprint") or "")
        if fingerprint in rewrite_by_id or fingerprint not in expected_ids:
            raise M3TrainingTaskError("mechanical smoke rewrite identity mismatch")
        if row.get("artifact_schema") != REWRITE_PROTOCOL:
            raise M3TrainingTaskError("mechanical smoke rewrite protocol mismatch")
        validate_rewrite_task(
            row,
            source=source_by_id[fingerprint],
            token_counter=token_counter,
        )
        rewrite_by_id[fingerprint] = row
    if set(rewrite_by_id) != expected_ids:
        raise M3TrainingTaskError("mechanical smoke requires all 96 validated rewrites")
    result: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        fingerprint = str(source["fingerprint"])
        if index % 4 != 3:
            rewrite = rewrite_by_id[fingerprint]
            result.append(
                _training_row(
                    source=source,
                    task_mode="rewrite",
                    origin="multi_provider_ai_smoke",
                    prompt=render_rewrite_prompt(rewrite),
                )
            )
        else:
            result.append(
                _training_row(
                    source=source,
                    task_mode="generate",
                    origin="structured_generation_smoke",
                    prompt=render_generation_prompt(source),
                )
            )
    if len(result) != 128 or len({row["fingerprint"] for row in result}) != 128:
        raise M3TrainingTaskError("mechanical smoke assembly cardinality mismatch")
    return result


__all__ = [
    "M3TrainingTaskError",
    "TRAINING_TASK_PROTOCOL",
    "assemble_mechanical_smoke_corpus",
    "render_generation_prompt",
]
