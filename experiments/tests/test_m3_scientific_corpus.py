from __future__ import annotations

from collections import Counter
import hashlib

import pytest

from data.m3_scientific_corpus import (
    M3ScientificCorpusError,
    SCIENTIFIC_REWRITE_PROTOCOL,
    assemble_scientific_training_corpus,
    scientific_assignment,
    scientific_manifest,
    validate_scientific_rewrite,
)
from data.rewrite_tasks import protected_literals


def source(index: int) -> dict:
    completion = (
        f"Document {index} explains a stable factual process in clear prose. "
        "It preserves the exact value 2026 while developing enough detail for a valid test. "
        "Readers receive practical context, a second complete sentence, and a concise conclusion."
    )
    return {
        "fingerprint": hashlib.sha256(f"fingerprint-{index}".encode()).hexdigest(),
        "source_fingerprint": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
        "completion": completion,
        "target_length": len(completion.split()),
        "target_length_unit": "tokens",
        "user_prompt": f"Explain document {index}",
        "use_case": "informational writing",
        "style_kind": "expository",
        "style": "clear natural prose",
        "detail_mode": "grounded",
        "em_dashes_allowed": False,
        "outline": [] if index % 4 == 0 else ["Explain the process", "Conclude"],
    }


def token_counter(text: str) -> int:
    return len(text.split())


def accepted_rewrite(
    item: dict, origin: str, assignment: dict[str, str] | None = None
) -> dict:
    assignment = assignment or scientific_assignment(item, origin)
    target = item["completion"]
    input_text = target.replace("explains", "provides an explanation of", 1)
    verification = {
        "same_language": True,
        "all_target_facts_supported_by_source": True,
        "no_source_fact_outside_target": True,
        "names_preserved": True,
        "numbers_dates_quotes_preserved": True,
        "semantic_similarity": 0.97,
        "missing_facts": [],
        "unsupported_source_claims": [],
    }
    return {
        "artifact_schema": SCIENTIFIC_REWRITE_PROTOCOL,
        "origin": origin,
        "fingerprint": item["fingerprint"],
        "source_fingerprint": item["source_fingerprint"],
        "input_text": input_text,
        "rewrite_instruction": "Improve the prose while preserving every fact.",
        "completion": target,
        "target_length": token_counter(target),
        "target_length_unit": "tokens",
        "input_length": token_counter(input_text),
        "input_length_unit": "tokens",
        "protected_literals": list(protected_literals(target)),
        "semantic_similarity": 0.97,
        **assignment,
        "generation_attempt": 1,
        "verification": verification,
    }


def test_4k_manifest_has_exact_frozen_mixture_and_balanced_api_generators() -> None:
    sources = [source(index) for index in range(4096)]
    manifest = scientific_manifest(sources)
    assert Counter(row["origin"] for row in manifest) == {
        "multi_provider_ai": 1843,
        "baseline_model_draft": 819,
        "controlled_light_edit": 205,
        "already_human_noop": 205,
        "generate": 1024,
    }
    api = [row for row in manifest if row["origin"] in {"multi_provider_ai", "controlled_light_edit"}]
    counts = Counter(row["generator_model"] for row in api)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_manifest_is_prefix_stable() -> None:
    sources = [source(index) for index in range(16384)]
    assert scientific_manifest(sources)[:4096] == scientific_manifest(sources[:4096])


def test_scientific_rewrite_rejects_identity_and_provider_drift() -> None:
    item = source(3)
    row = accepted_rewrite(item, "multi_provider_ai")
    validate_scientific_rewrite(
        row, source=item, origin="multi_provider_ai", token_counter=token_counter
    )
    identical = row | {
        "input_text": item["completion"],
        "input_length": token_counter(item["completion"]),
    }
    with pytest.raises(M3ScientificCorpusError, match="non-noop"):
        validate_scientific_rewrite(
            identical, source=item, origin="multi_provider_ai", token_counter=token_counter
        )
    drifted = row | {"generator_model": "unexpected/model"}
    with pytest.raises(M3ScientificCorpusError, match="assignment drift"):
        validate_scientific_rewrite(
            drifted, source=item, origin="multi_provider_ai", token_counter=token_counter
        )


def test_assembly_requires_every_constructed_rewrite_and_builds_noops() -> None:
    sources = [source(index) for index in range(4096)]
    manifest = scientific_manifest(sources)
    rewrites = [
        accepted_rewrite(
            item,
            assignment["origin"],
            {
                field: assignment[field]
                for field in ("generator_model", "verifier_model", "template_id")
            },
        )
        for item, assignment in zip(sources, manifest)
        if assignment["origin"] in {
            "multi_provider_ai",
            "baseline_model_draft",
            "controlled_light_edit",
        }
    ]
    corpus = assemble_scientific_training_corpus(
        sources, rewrites, token_counter=token_counter
    )
    assert len(corpus) == 4096
    counts = Counter(row["origin"] for row in corpus)
    assert counts["already_human_noop"] == 205
    noops = [row for row in corpus if row["origin"] == "already_human_noop"]
    assert all("Do not paraphrase for its own sake" in row["prompt"] for row in noops)
    with pytest.raises(M3ScientificCorpusError, match="missing constructed"):
        assemble_scientific_training_corpus(
            sources, rewrites[:-1], token_counter=token_counter
        )
