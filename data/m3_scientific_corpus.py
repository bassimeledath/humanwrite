"""Frozen construction helpers for the M3 4K/16K/46K scientific corpus."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from data.m3_training_tasks import (
    M3TrainingTaskError,
    TRAINING_TASK_PROTOCOL,
    render_generation_prompt,
)
from data.rewrite_tasks import (
    GENERATOR_MODELS,
    TEMPLATE_IDS,
    VERIFIER_BY_GENERATOR,
    RewriteTaskError,
    contains_unexpected_non_latin,
    frozen_task_strata,
    protected_literals,
)


SCIENTIFIC_REWRITE_PROTOCOL = "humanwrite.m3.scientific_rewrite.v1"
BASE_MODEL = "Qwen/Qwen3-14B"
BASE_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
API_REWRITE_ORIGINS = {"multi_provider_ai", "controlled_light_edit"}
CONSTRUCTED_REWRITE_ORIGINS = API_REWRITE_ORIGINS | {"baseline_model_draft"}


class M3ScientificCorpusError(ValueError):
    pass


def _fingerprint_int(fingerprint: str) -> int:
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise M3ScientificCorpusError("fingerprint must be lowercase SHA-256")
    return int(hashlib.sha256(f"m3-scientific:{fingerprint}".encode()).hexdigest(), 16)


def scientific_assignment(
    source: dict[str, Any], origin: str, *, slot: int | None = None
) -> dict[str, str]:
    fingerprint = str(source.get("fingerprint") or "")
    value = _fingerprint_int(fingerprint)
    if origin in API_REWRITE_ORIGINS:
        assignment_index = value if slot is None else slot
        if type(assignment_index) is not int or assignment_index < 0:
            raise M3ScientificCorpusError("API assignment slot must be a nonnegative integer")
        generator = GENERATOR_MODELS[assignment_index % len(GENERATOR_MODELS)]
        verifier = VERIFIER_BY_GENERATOR[generator]
        template = TEMPLATE_IDS[
            (assignment_index // len(GENERATOR_MODELS)) % len(TEMPLATE_IDS)
        ]
    elif origin == "baseline_model_draft":
        generator = f"{BASE_MODEL}@{BASE_REVISION}"
        verifier = "qwen/qwen3-32b"
        template = "source_brief_reconstruction"
    elif origin == "already_human_noop":
        generator, verifier, template = "identity", "deterministic", "minimal_restraint"
    elif origin == "generate":
        generator, verifier, template = "none", "deterministic", "structured_generation"
    else:
        raise M3ScientificCorpusError(f"unsupported M3 origin: {origin}")
    return {
        "generator_model": generator,
        "verifier_model": verifier,
        "template_id": template,
    }


def scientific_manifest(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    origins = frozen_task_strata(sources)
    rows: list[dict[str, Any]] = []
    origin_slots: dict[str, int] = {}
    for source, origin in zip(sources, origins):
        fingerprint = str(source.get("fingerprint") or "")
        # Opposite parity for the second odd-sized API stratum keeps the full
        # 4K API pool exactly balanced while each stratum differs by at most one.
        offset = 1 if origin == "controlled_light_edit" else 0
        counter = origin_slots.get(origin, 0)
        slot = counter + offset
        assignment = scientific_assignment(source, origin, slot=slot)
        origin_slots[origin] = counter + 1
        rows.append(
            {
                "artifact_schema": "humanwrite.m3.scientific_manifest.v1",
                "fingerprint": fingerprint,
                "source_fingerprint": str(source.get("source_fingerprint") or fingerprint),
                "origin": origin,
                "task_mode": "generate" if origin == "generate" else "rewrite",
                **assignment,
            }
        )
    if len(rows) != len(sources) or len({row["fingerprint"] for row in rows}) != len(rows):
        raise M3ScientificCorpusError("scientific manifest identity mismatch")
    return rows


def render_noop_prompt(source: dict[str, Any]) -> str:
    return (
        "MODE: REWRITE\n"
        "INSTRUCTION: Review the passage conservatively. Preserve wording that already reads "
        "naturally; make only corrections that are genuinely necessary. Do not paraphrase for "
        "its own sake.\n"
        f"REQUESTED STYLE: {source.get('style') or 'preserve the existing human style'}\n"
        "PRESERVE EXACTLY: names, numbers, dates, quotations, URLs, and factual claims.\n\n"
        f"SOURCE TEXT:\n{str(source.get('completion') or '').strip()}\n\n"
        "RETURN: only the revised passage."
    )


def render_scientific_rewrite_prompt(row: dict[str, Any], source: dict[str, Any]) -> str:
    instruction = str(row.get("rewrite_instruction") or "").strip()
    input_text = str(row.get("input_text") or "").strip()
    return (
        "MODE: REWRITE\n"
        f"INSTRUCTION: {instruction}\n"
        f"REQUESTED STYLE: {source.get('style') or 'natural prose appropriate to the request'}\n"
        f"TARGET LENGTH: approximately {int(source['target_length'])} tokens\n"
        "PRESERVE EXACTLY: names, numbers, dates, quotations, URLs, and factual claims.\n\n"
        f"SOURCE TEXT:\n{input_text}\n\n"
        "RETURN: only the revised passage."
    )


def validate_scientific_rewrite(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    origin: str,
    token_counter: Callable[[str], int],
    semantic_similarity_min: float = 0.90,
    expected_assignment: dict[str, str] | None = None,
) -> None:
    expected_keys = {
        "artifact_schema",
        "origin",
        "fingerprint",
        "source_fingerprint",
        "input_text",
        "rewrite_instruction",
        "completion",
        "target_length",
        "target_length_unit",
        "input_length",
        "input_length_unit",
        "protected_literals",
        "semantic_similarity",
        "generator_model",
        "verifier_model",
        "template_id",
        "generation_attempt",
        "verification",
    }
    if set(row) != expected_keys:
        raise M3ScientificCorpusError("scientific rewrite exact schema mismatch")
    fingerprint = str(source.get("fingerprint") or "")
    if row.get("artifact_schema") != SCIENTIFIC_REWRITE_PROTOCOL:
        raise M3ScientificCorpusError("scientific rewrite protocol mismatch")
    if origin not in CONSTRUCTED_REWRITE_ORIGINS or row.get("origin") != origin:
        raise M3ScientificCorpusError("scientific rewrite origin mismatch")
    if row.get("fingerprint") != fingerprint:
        raise M3ScientificCorpusError("scientific rewrite fingerprint mismatch")
    if row.get("source_fingerprint") != str(source.get("source_fingerprint") or fingerprint):
        raise M3ScientificCorpusError("scientific rewrite source fingerprint mismatch")
    assignment = expected_assignment or scientific_assignment(source, origin)
    if set(assignment) != {"generator_model", "verifier_model", "template_id"}:
        raise M3ScientificCorpusError("scientific rewrite expected assignment is invalid")
    if any(row.get(field) != assignment[field] for field in assignment):
        raise M3ScientificCorpusError("scientific rewrite provider assignment drift")
    target = str(source.get("completion") or "").strip()
    input_text = str(row.get("input_text") or "").strip()
    instruction = str(row.get("rewrite_instruction") or "").strip()
    if not target or not input_text or not instruction or input_text == target:
        raise M3ScientificCorpusError("scientific non-noop rewrite text is invalid")
    if "�" in target or "�" in input_text:
        raise M3ScientificCorpusError("replacement character is forbidden")
    if contains_unexpected_non_latin(target) or contains_unexpected_non_latin(input_text):
        raise M3ScientificCorpusError("unexpected non-Latin alphabetic character")
    literals = list(protected_literals(target))
    if row.get("protected_literals") != literals or any(value not in input_text for value in literals):
        raise M3ScientificCorpusError("protected literal mismatch")
    target_tokens = token_counter(target)
    input_tokens = token_counter(input_text)
    if target_tokens < 32 or input_tokens < 32 or not 0.70 <= input_tokens / target_tokens <= 1.35:
        raise M3ScientificCorpusError("scientific rewrite token-length gate failed")
    if row.get("target_length") != target_tokens or row.get("input_length") != input_tokens:
        raise M3ScientificCorpusError("scientific rewrite stored token lengths drifted")
    if row.get("target_length_unit") != "tokens" or row.get("input_length_unit") != "tokens":
        raise M3ScientificCorpusError("scientific rewrite lengths must use tokens")
    verification = row.get("verification")
    required_true = (
        "same_language",
        "all_target_facts_supported_by_source",
        "no_source_fact_outside_target",
        "names_preserved",
        "numbers_dates_quotes_preserved",
    )
    if not isinstance(verification, dict) or any(verification.get(key) is not True for key in required_true):
        raise M3ScientificCorpusError("scientific rewrite verification failed")
    if verification.get("missing_facts") or verification.get("unsupported_source_claims"):
        raise M3ScientificCorpusError("scientific rewrite verifier reported factual mismatch")
    similarity = float(verification.get("semantic_similarity", -1))
    if not 0.0 <= similarity <= 1.0 or similarity < semantic_similarity_min:
        raise M3ScientificCorpusError("scientific rewrite semantic gate failed")
    if float(row.get("semantic_similarity", -1)) != similarity:
        raise M3ScientificCorpusError("scientific rewrite semantic score drifted")
    if type(row.get("generation_attempt")) is not int or row["generation_attempt"] < 1:
        raise M3ScientificCorpusError("scientific rewrite attempt provenance is invalid")


def _training_row(
    source: dict[str, Any], *, task_mode: str, origin: str, prompt: str
) -> dict[str, Any]:
    fingerprint = str(source.get("fingerprint") or "")
    completion = str(source.get("completion") or "").strip()
    if not fingerprint or not completion or not prompt.strip():
        raise M3TrainingTaskError("scientific training row is incomplete")
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


def assemble_scientific_training_corpus(
    sources: list[dict[str, Any]],
    rewrite_rows: list[dict[str, Any]],
    *,
    token_counter: Callable[[str], int],
) -> list[dict[str, Any]]:
    manifest = scientific_manifest(sources)
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    expected_constructed = {
        row["fingerprint"]: row for row in manifest if row["origin"] in CONSTRUCTED_REWRITE_ORIGINS
    }
    rewrites: dict[str, dict[str, Any]] = {}
    for row in rewrite_rows:
        fingerprint = str(row.get("fingerprint") or "")
        assignment = expected_constructed.get(fingerprint)
        if assignment is None or fingerprint in rewrites:
            raise M3ScientificCorpusError("scientific rewrite identity mismatch")
        validate_scientific_rewrite(
            row,
            source=source_by_id[fingerprint],
            origin=str(assignment["origin"]),
            token_counter=token_counter,
            expected_assignment={
                field: str(assignment[field])
                for field in ("generator_model", "verifier_model", "template_id")
            },
        )
        rewrites[fingerprint] = row
    if set(rewrites) != set(expected_constructed):
        raise M3ScientificCorpusError("scientific corpus is missing constructed rewrites")
    result: list[dict[str, Any]] = []
    for assignment in manifest:
        source = source_by_id[assignment["fingerprint"]]
        origin = str(assignment["origin"])
        if origin in CONSTRUCTED_REWRITE_ORIGINS:
            prompt = render_scientific_rewrite_prompt(rewrites[assignment["fingerprint"]], source)
            task_mode = "rewrite"
        elif origin == "already_human_noop":
            prompt, task_mode = render_noop_prompt(source), "rewrite"
        else:
            prompt, task_mode = render_generation_prompt(source), "generate"
        result.append(_training_row(source, task_mode=task_mode, origin=origin, prompt=prompt))
    if len(result) != len(sources) or len({row["fingerprint"] for row in result}) != len(result):
        raise M3ScientificCorpusError("scientific training corpus cardinality mismatch")
    return result


__all__ = [
    "API_REWRITE_ORIGINS",
    "BASE_MODEL",
    "BASE_REVISION",
    "CONSTRUCTED_REWRITE_ORIGINS",
    "M3ScientificCorpusError",
    "SCIENTIFIC_REWRITE_PROTOCOL",
    "assemble_scientific_training_corpus",
    "render_noop_prompt",
    "render_scientific_rewrite_prompt",
    "scientific_assignment",
    "scientific_manifest",
    "validate_scientific_rewrite",
]
