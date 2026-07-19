"""Frozen M3 rewrite-task construction and validation helpers."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from typing import Any, Callable


PROTOCOL = "humanwrite.m3.rewrite_tasks.v1"
TASK_STRATA = (
    "multi_provider_ai",
    "baseline_model_draft",
    "controlled_light_edit",
    "already_human_noop",
    "generate",
)
FROZEN_STAGE_COUNTS = {
    128: (58, 26, 6, 6, 32),
    4096: (1843, 819, 205, 205, 1024),
    16384: (7373, 3277, 819, 819, 4096),
    46080: (20736, 9216, 2304, 2304, 11520),
}
GENERATOR_MODELS = ("google/gemini-3.1-flash-lite", "anthropic/claude-haiku-4.5")
VERIFIER_BY_GENERATOR = {
    "google/gemini-3.1-flash-lite": "qwen/qwen3-32b",
    "anthropic/claude-haiku-4.5": "qwen/qwen3-32b",
}
TEMPLATE_IDS = (
    "generic_polished",
    "overstructured",
    "verbose_transitions",
    "corporate_expository",
)
URL_RE = re.compile(r"https?://[^\s<>\]\[(){}]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$£€]\s*)?\d(?:[\d,.:/-]*\d)?(?:\s*%|\s*(?:am|pm))?(?!\w)",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(r"[“\"]([^“”\"\n]{2,240})[”\"]")


class RewriteTaskError(ValueError):
    """Raised when a synthesized rewrite pair violates the frozen contract."""


def _fingerprint_int(fingerprint: str) -> int:
    if not re.fullmatch(r"[0-9a-f]{64}", str(fingerprint or "")):
        raise RewriteTaskError("document fingerprint must be lowercase SHA-256")
    return int(hashlib.sha256(fingerprint.encode("ascii")).hexdigest(), 16)


def deterministic_assignment(fingerprint: str) -> dict[str, str]:
    """Balance generator families and prompt templates without outcome access."""
    value = _fingerprint_int(fingerprint)
    generator = GENERATOR_MODELS[value % len(GENERATOR_MODELS)]
    return {
        "generator_model": generator,
        "verifier_model": VERIFIER_BY_GENERATOR[generator],
        "template_id": TEMPLATE_IDS[(value // len(GENERATOR_MODELS)) % len(TEMPLATE_IDS)],
    }


def contains_unexpected_non_latin(text: str) -> bool:
    return any(
        char.isalpha() and "LATIN" not in unicodedata.name(char, "")
        for char in text
    )


def protected_literals(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for expression in (URL_RE, EMAIL_RE, NUMBER_RE):
        values.extend(match.group(0).strip() for match in expression.finditer(text))
    values.extend(match.group(1).strip() for match in QUOTE_RE.finditer(text))
    return tuple(dict.fromkeys(value for value in values if value))


def rewrite_source_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign exactly three of every four frozen-order records to rewriting."""
    if len(records) % 4:
        raise RewriteTaskError("task-mixture source cardinality must be divisible by four")
    fingerprints = [str(record.get("fingerprint") or "") for record in records]
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in fingerprints):
        raise RewriteTaskError("task-mixture sources require lowercase SHA-256 fingerprints")
    if len(fingerprints) != len(set(fingerprints)):
        raise RewriteTaskError("task-mixture sources contain duplicate fingerprints")
    return [record for index, record in enumerate(records) if index % 4 != 3]


def frozen_task_strata(records: list[dict[str, Any]]) -> list[str]:
    """Return outcome-independent, exact, prefix-stable M3 task assignments."""
    cardinality = len(records)
    if cardinality not in FROZEN_STAGE_COUNTS:
        raise RewriteTaskError(
            f"task-mixture cardinality must be one of {sorted(FROZEN_STAGE_COUNTS)}"
        )
    fingerprints = [str(record.get("fingerprint") or "") for record in records]
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in fingerprints):
        raise RewriteTaskError("task-mixture sources require lowercase SHA-256 fingerprints")
    if len(fingerprints) != len(set(fingerprints)):
        raise RewriteTaskError("task-mixture sources contain duplicate fingerprints")
    assignments: list[str] = []
    prior_size = 0
    prior_counts = (0,) * len(TASK_STRATA)
    for stage_size, cumulative_counts in FROZEN_STAGE_COUNTS.items():
        if stage_size > cardinality:
            break
        segment_counts = tuple(
            current - prior for current, prior in zip(cumulative_counts, prior_counts)
        )
        labels = [
            stratum
            for stratum, count in zip(TASK_STRATA, segment_counts)
            for _ in range(count)
        ]
        if len(labels) != stage_size - prior_size or any(count < 0 for count in segment_counts):
            raise RewriteTaskError("invalid frozen task-mixture stage counts")
        seed_material = f"humanwrite.m3.task-strata.v1:{prior_size}:{stage_size}"
        random.Random(int(hashlib.sha256(seed_material.encode()).hexdigest(), 16)).shuffle(labels)
        assignments.extend(labels)
        prior_size, prior_counts = stage_size, cumulative_counts
    if len(assignments) != cardinality:
        raise RewriteTaskError("frozen task-mixture assignment did not cover the corpus")
    return assignments


def generator_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["document_fingerprint", "source_text", "rewrite_instruction"],
        "properties": {
            "document_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "source_text": {"type": "string", "minLength": 80},
            "rewrite_instruction": {"type": "string", "minLength": 8, "maxLength": 300},
        },
    }


def verifier_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "document_fingerprint",
            "same_language",
            "all_target_facts_supported_by_source",
            "no_source_fact_outside_target",
            "names_preserved",
            "numbers_dates_quotes_preserved",
            "semantic_similarity",
            "missing_facts",
            "unsupported_source_claims",
        ],
        "properties": {
            "document_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "same_language": {"type": "boolean"},
            "all_target_facts_supported_by_source": {"type": "boolean"},
            "no_source_fact_outside_target": {"type": "boolean"},
            "names_preserved": {"type": "boolean"},
            "numbers_dates_quotes_preserved": {"type": "boolean"},
            # Some otherwise compatible structured-output providers reject
            # numeric range keywords. The same [0, 1] bound is enforced after
            # decoding by assemble_rewrite_task, so portability does not relax
            # acceptance.
            "semantic_similarity": {"type": "number"},
            "missing_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "unsupported_source_claims": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 12,
            },
        },
    }


def generator_prompt(
    source: dict[str, Any],
    assignment: dict[str, str],
    *,
    attempt: int = 1,
    previous_error: str = "",
) -> str:
    if type(attempt) is not int or attempt < 1:
        raise RewriteTaskError("generation attempt must be a positive integer")
    target = str(source["completion"])
    template = assignment["template_id"]
    template_guidance = {
        "generic_polished": "Use smooth, generic, conventionally polished assistant prose.",
        "overstructured": "Use conspicuous structure and explicit signposting where natural.",
        "verbose_transitions": "Use wordier transitions and explanatory connective language.",
        "corporate_expository": "Use safe, professional, corporate-expository phrasing.",
    }[template]
    recovery = ""
    if attempt > 1:
        error = re.sub(r"\s+", " ", str(previous_error or "unspecified validation failure"))[:220]
        recovery = (
            f"\nThis is recovery attempt {attempt}. The previous candidate failed validation: {error}. "
            "Correct that failure. In particular, produce a genuinely different draft: reorganize "
            "at least one sentence or clause and replace enough unprotected wording that the source "
            "cannot be byte-identical to the target. Keep every protected literal and fact exact. "
            "Keep the source between roughly 85% and 115% of the target length.\n"
        )
    return (
        "Create a realistic AI-written version of the human passage below for a rewriting "
        "training task. Preserve every fact, name, number, date, quotation, URL, and intent. "
        "Do not summarize, add facts, remove details, change language, or mention this task. "
        "Keep length close to the original. The result should remain usable prose, not a parody. "
        "Even if the target is already polished, do not copy it verbatim: produce an alternate "
        "draft with meaning-preserving changes to sentence structure, transitions, and unprotected "
        "wording. Preserve exact quotations and other literal values verbatim. "
        f"Stylistic variant: {template}. {template_guidance} "
        "Also provide a short, natural user instruction asking a writing assistant to improve "
        "the source while preserving its meaning. Repeat document_fingerprint exactly."
        f"{recovery}\n\n"
        f"document_fingerprint: {source['fingerprint']}\n\nHUMAN TARGET:\n{target}"
    )


def verifier_prompt(
    source: dict[str, Any], generated: dict[str, Any]
) -> str:
    return (
        "Audit whether an AI-styled source and a human target express the same complete factual "
        "content. Be strict: names, numbers, dates, quotations, URLs, causality, attribution, "
        "negation, and scope must agree. The source may differ stylistically but may neither omit "
        "a target fact nor add an unsupported claim. semantic_similarity is 0..1 and must reflect "
        "content rather than fluency. Return only the schema JSON and repeat the fingerprint.\n\n"
        f"document_fingerprint: {source['fingerprint']}\n\n"
        f"AI-STYLED SOURCE:\n{generated['source_text']}\n\n"
        f"HUMAN TARGET:\n{source['completion']}"
    )


def assemble_rewrite_task(
    *,
    source: dict[str, Any],
    generated: dict[str, Any],
    verified: dict[str, Any],
    assignment: dict[str, str],
    token_counter: Callable[[str], int],
    semantic_similarity_min: float = 0.90,
) -> dict[str, Any]:
    fingerprint = str(source.get("fingerprint") or "")
    expected_assignment = deterministic_assignment(fingerprint)
    if assignment != expected_assignment:
        raise RewriteTaskError("provider assignment drift")
    if generated.get("document_fingerprint") != fingerprint:
        raise RewriteTaskError("generator fingerprint mismatch")
    if verified.get("document_fingerprint") != fingerprint:
        raise RewriteTaskError("verifier fingerprint mismatch")
    target = str(source.get("completion") or "").strip()
    ai_source = str(generated.get("source_text") or "").strip()
    instruction = str(generated.get("rewrite_instruction") or "").strip()
    if not target or not ai_source or not instruction:
        raise RewriteTaskError("rewrite text and instruction must be nonempty")
    if target == ai_source:
        raise RewriteTaskError("non-noop rewrite source equals target")
    if "�" in target or "�" in ai_source:
        raise RewriteTaskError("replacement character is forbidden")
    if contains_unexpected_non_latin(target) or contains_unexpected_non_latin(ai_source):
        raise RewriteTaskError("unexpected non-Latin alphabetic character")
    missing_literals = [value for value in protected_literals(target) if value not in ai_source]
    if missing_literals:
        raise RewriteTaskError(f"protected literals missing from source: {missing_literals[:3]}")
    target_tokens, source_tokens = token_counter(target), token_counter(ai_source)
    if target_tokens < 32 or source_tokens < 32:
        raise RewriteTaskError("rewrite pair is too short")
    ratio = source_tokens / target_tokens
    if not 0.70 <= ratio <= 1.35:
        raise RewriteTaskError("source/target token-length ratio is outside the frozen range")
    required_true = (
        "same_language",
        "all_target_facts_supported_by_source",
        "no_source_fact_outside_target",
        "names_preserved",
        "numbers_dates_quotes_preserved",
    )
    if any(verified.get(field) is not True for field in required_true):
        raise RewriteTaskError("provider content-preservation verification failed")
    if verified.get("missing_facts") or verified.get("unsupported_source_claims"):
        raise RewriteTaskError("provider reported factual mismatch")
    similarity = float(verified.get("semantic_similarity", -1))
    if not 0.0 <= similarity <= 1.0:
        raise RewriteTaskError("semantic similarity must be within [0, 1]")
    if similarity < semantic_similarity_min:
        raise RewriteTaskError("semantic similarity is below the frozen minimum")
    return {
        "artifact_schema": PROTOCOL,
        "task_mode": "rewrite",
        "fingerprint": fingerprint,
        "source_fingerprint": str(source.get("source_fingerprint") or fingerprint),
        "input_text": ai_source,
        "rewrite_instruction": instruction,
        "completion": target,
        "target_length": target_tokens,
        "target_length_unit": "tokens",
        "input_length": source_tokens,
        "input_length_unit": "tokens",
        "protected_literals": list(protected_literals(target)),
        "semantic_similarity": similarity,
        "generator_model": assignment["generator_model"],
        "verifier_model": assignment["verifier_model"],
        "template_id": assignment["template_id"],
        "generation_attempt": int(generated.get("generation_attempt", 1)),
        "verification": {
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
        },
    }


def validate_rewrite_task(
    record: dict[str, Any],
    *,
    source: dict[str, Any],
    token_counter: Callable[[str], int],
    semantic_similarity_min: float = 0.90,
) -> None:
    fingerprint = str(source.get("fingerprint") or "")
    assignment = deterministic_assignment(fingerprint)
    expected_keys = {
        "artifact_schema",
        "task_mode",
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
    if set(record) != expected_keys:
        raise RewriteTaskError("rewrite task exact schema mismatch")
    if record.get("artifact_schema") != PROTOCOL or record.get("task_mode") != "rewrite":
        raise RewriteTaskError("rewrite task protocol mismatch")
    if record.get("fingerprint") != fingerprint:
        raise RewriteTaskError("rewrite task fingerprint mismatch")
    if record.get("completion") != str(source.get("completion") or "").strip():
        raise RewriteTaskError("rewrite task human target mismatch")
    if any(record.get(key) != value for key, value in assignment.items()):
        raise RewriteTaskError("rewrite task provider assignment mismatch")
    verification = record.get("verification")
    if not isinstance(verification, dict):
        raise RewriteTaskError("rewrite task verification is missing")
    reconstructed_generated = {
        "document_fingerprint": fingerprint,
        "source_text": record.get("input_text"),
        "rewrite_instruction": record.get("rewrite_instruction"),
        "generation_attempt": record.get("generation_attempt"),
    }
    reconstructed_verified = {"document_fingerprint": fingerprint, **verification}
    rebuilt = assemble_rewrite_task(
        source=source,
        generated=reconstructed_generated,
        verified=reconstructed_verified,
        assignment=assignment,
        token_counter=token_counter,
        semantic_similarity_min=semantic_similarity_min,
    )
    if rebuilt != record:
        raise RewriteTaskError("rewrite task failed deterministic reconstruction")


def render_rewrite_prompt(record: dict[str, Any]) -> str:
    if record.get("artifact_schema") != PROTOCOL or record.get("task_mode") != "rewrite":
        raise RewriteTaskError("cannot render an invalid rewrite task")
    return (
        "MODE: REWRITE\n"
        "SOURCE TEXT:\n"
        f"{record['input_text']}\n\n"
        "USER INSTRUCTION:\n"
        f"{record['rewrite_instruction']}\n\n"
        "PRESERVATION REQUIREMENTS:\n"
        "Preserve all facts, names, numbers, dates, quotations, URLs, intent, and language.\n\n"
        f"TARGET LENGTH: approximately {record['target_length']} tokens\n"
        "RETURN: only the rewritten text."
    )
