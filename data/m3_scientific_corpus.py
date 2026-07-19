"""Frozen construction helpers for the M3 4K/16K/46K scientific corpus."""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
import math
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


SCIENTIFIC_REWRITE_PROTOCOL = "humanwrite.m3.scientific_rewrite.v2"
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
    target_length = row.get("target_length", source.get("target_length"))
    if type(target_length) is not int or target_length < 1:
        raise M3ScientificCorpusError("scientific rewrite prompt requires token target length")
    return (
        "MODE: REWRITE\n"
        f"INSTRUCTION: {instruction}\n"
        f"REQUESTED STYLE: {source.get('style') or 'natural prose appropriate to the request'}\n"
        f"TARGET LENGTH: approximately {target_length} tokens\n"
        "PRESERVE EXACTLY: names, numbers, dates, quotations, URLs, and factual claims.\n\n"
        f"SOURCE TEXT:\n{input_text}\n\n"
        "RETURN: only the revised passage."
    )


def scientific_generator_prompt(
    source: dict[str, Any],
    assignment: dict[str, str],
    origin: str,
    *,
    attempt: int = 1,
    previous_error: str = "",
    explicit_literal_inventory: bool = False,
    literal_placeholders: bool = False,
    target_token_count: int | None = None,
) -> str:
    if origin not in API_REWRITE_ORIGINS:
        raise M3ScientificCorpusError("API generator prompt requires an API rewrite origin")
    target = str(source.get("completion") or "").strip()
    if not target or type(attempt) is not int or attempt < 1:
        raise M3ScientificCorpusError("scientific generator prompt inputs are invalid")
    if origin == "controlled_light_edit":
        mode = (
            "Make a restrained, realistic pre-edit draft. Preserve the paragraph order and most "
            "wording, but introduce a small number of discourse-level weaknesses such as an "
            "overexplicit transition, a needlessly repeated subject, or an awkward sentence join. "
            "Do not introduce spelling errors, factual errors, a phrase blacklist, or parody."
        )
        instruction = (
            "Lightly edit this passage for naturalness and flow while preserving its existing "
            "voice and every factual detail."
        )
    else:
        mode = (
            "Create a realistic AI-written alternate draft with conventionally polished, somewhat "
            "formulaic exposition. Substantially recast sentence structure, paragraph flow, transitions, "
            "and unprotected wording while keeping it usable rather than a parody. The result must not "
            "be a whitespace-, formatting-, or punctuation-only variant and must not preserve nearly all "
            "of the target's phrasing."
        )
        instruction = (
            "Rewrite this draft so it reads naturally and distinctly human while preserving every "
            "fact, name, number, date, quotation, URL, and intent."
        )
    recovery = ""
    if attempt > 1:
        error = re.sub(r"\s+", " ", previous_error or "validation failure")[:220]
        recovery = (
            f"\nRecovery attempt {attempt}; the prior candidate failed: {error}. Correct that exact "
            "failure without relaxing factual or literal preservation.\n"
        )
    literal_inventory = ""
    if explicit_literal_inventory:
        literals = list(protected_literals(target))
        literal_inventory = (
            "\nThe following protected literals must each appear byte-for-byte in source_text. "
            "Do not reformat, translate, normalize, or omit them:\n"
            + "\n".join(f"- {json.dumps(value, ensure_ascii=False)}" for value in literals)
            + "\n"
        )
    prompt_target = target
    placeholder_instruction = ""
    if literal_placeholders:
        prompt_target, mapping = mask_protected_literals(target)
        placeholder_instruction = (
            "\nProtected spans in HUMAN TARGET have been replaced by placeholder tokens. "
            "Every placeholder must appear exactly once and byte-for-byte in source_text; do not "
            "rewrite, remove, duplicate, or explain a placeholder.\n"
            f"Required placeholders: {', '.join(mapping)}\n"
        )
    token_instruction = ""
    if target_token_count is not None:
        if type(target_token_count) is not int or target_token_count < 1:
            raise M3ScientificCorpusError("target token count is invalid")
        token_instruction = (
            f"\nThe human target contains {target_token_count} tokenizer tokens. Keep source_text "
            f"between {math.ceil(0.70 * target_token_count)} and "
            f"{math.floor(1.35 * target_token_count)} tokenizer tokens.\n"
        )
    return (
        f"{mode} Preserve every fact, name, number, date, quotation, URL, email address, language, "
        "scope, attribution, and intent. Do not summarize, add facts, omit details, or mention this "
        "task. Keep the source between roughly 85% and 115% of the human target token length. "
        f"Template: {assignment['template_id']}. Repeat document_fingerprint exactly. Return "
        "source_text plus the supplied natural rewrite_instruction."
        f"{literal_inventory}"
        f"{placeholder_instruction}"
        f"{token_instruction}"
        f"{recovery}\n\n"
        f"document_fingerprint: {source['fingerprint']}\n"
        f"rewrite_instruction: {instruction}\n\nHUMAN TARGET:\n{prompt_target}"
    )


def mask_protected_literals(target: str) -> tuple[str, dict[str, str]]:
    values = sorted(set(protected_literals(target)), key=lambda value: (-len(value), value))
    def label(index: int) -> str:
        letters = ""
        value = index
        while value:
            value, remainder = divmod(value - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return letters

    mapping = {
        f"[[PROTECTED_LITERAL_{label(index)}]]": value
        for index, value in enumerate(values, 1)
    }
    masked = target
    for placeholder, value in mapping.items():
        masked = masked.replace(value, placeholder)
    return masked, mapping


def restore_protected_literals(text: str, target: str) -> str:
    _, mapping = mask_protected_literals(target)
    restored = text
    for placeholder, value in mapping.items():
        restored = restored.replace(placeholder, value)
    return restored


def assemble_scientific_rewrite(
    *,
    source: dict[str, Any],
    origin: str,
    generated: dict[str, Any],
    verified: dict[str, Any],
    assignment: dict[str, str],
    token_counter: Callable[[str], int],
    semantic_similarity_min: float = 0.90,
) -> dict[str, Any]:
    fingerprint = str(source.get("fingerprint") or "")
    if origin not in CONSTRUCTED_REWRITE_ORIGINS:
        raise M3ScientificCorpusError("scientific assembly origin is invalid")
    if generated.get("document_fingerprint") != fingerprint:
        raise M3ScientificCorpusError("scientific generator fingerprint mismatch")
    if verified.get("document_fingerprint") != fingerprint:
        raise M3ScientificCorpusError("scientific verifier fingerprint mismatch")
    target = str(source.get("completion") or "").strip()
    input_text = str(generated.get("source_text") or "").strip()
    verification = {
        field: verified[field]
        for field in (
            "same_language",
            "all_target_facts_supported_by_source",
            "no_source_fact_outside_target",
            "names_preserved",
            "numbers_dates_quotes_preserved",
            "semantic_similarity",
            "missing_facts",
            "unsupported_source_claims",
        )
    }
    row = {
        "artifact_schema": SCIENTIFIC_REWRITE_PROTOCOL,
        "origin": origin,
        "fingerprint": fingerprint,
        "source_fingerprint": str(source.get("source_fingerprint") or fingerprint),
        "input_text": input_text,
        "rewrite_instruction": str(generated.get("rewrite_instruction") or "").strip(),
        "completion": target,
        "target_length": token_counter(target),
        "target_length_unit": "tokens",
        "input_length": token_counter(input_text),
        "input_length_unit": "tokens",
        "protected_literals": list(protected_literals(target)),
        "semantic_similarity": float(verified.get("semantic_similarity", -1)),
        **assignment,
        "generation_attempt": int(generated.get("generation_attempt", 1)),
        "verification": verification,
    }
    validate_scientific_rewrite(
        row,
        source=source,
        origin=origin,
        token_counter=token_counter,
        semantic_similarity_min=semantic_similarity_min,
        expected_assignment=assignment,
    )
    return row


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
    normalize = lambda text: re.sub(r"\s+", " ", text).strip().casefold()
    normalized_target, normalized_input = normalize(target), normalize(input_text)
    if normalized_input == normalized_target:
        raise M3ScientificCorpusError("scientific rewrite differs only in whitespace or casing")
    surface_similarity = SequenceMatcher(
        None, normalized_input, normalized_target, autojunk=False
    ).ratio()
    if origin == "multi_provider_ai" and surface_similarity >= 0.95:
        raise M3ScientificCorpusError("multi-provider rewrite is too surface-similar to target")
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
    "assemble_scientific_rewrite",
    "assemble_scientific_training_corpus",
    "render_noop_prompt",
    "render_scientific_rewrite_prompt",
    "scientific_assignment",
    "scientific_generator_prompt",
    "scientific_manifest",
    "validate_scientific_rewrite",
]
