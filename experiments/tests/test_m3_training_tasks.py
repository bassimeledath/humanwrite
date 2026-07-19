from __future__ import annotations

from data.m3_training_tasks import (
    TRAINING_TASK_PROTOCOL,
    assemble_mechanical_smoke_corpus,
    render_generation_prompt,
)
from data.rewrite_tasks import assemble_rewrite_task, deterministic_assignment


def source(index: int) -> dict:
    completion = (
        f"Acme unit {index} reported revenue of $12.4 million in 2025. "
        "The result remained steady across the quarter and the company retained its plan."
    )
    return {
        "fingerprint": format(index, "064x"),
        "source_fingerprint": format(index + 1000, "064x"),
        "completion": completion,
        "user_prompt": f"Write a concise report about Acme unit {index}.",
        "use_case": "business report",
        "style_kind": "informative",
        "style": "Clear and direct.",
        "detail_mode": "grounded",
        "target_length": 48,
        "target_length_unit": "tokens",
        "em_dashes_allowed": False,
        "outline": [f"Unit {index}", "$12.4 million", "2025"],
    }


def tokens(text: str) -> int:
    return len(text.split()) * 2


def rewrite(row: dict) -> dict:
    fingerprint = row["fingerprint"]
    generated = {
        "document_fingerprint": fingerprint,
        "source_text": row["completion"].replace("reported revenue", "recorded revenue"),
        "rewrite_instruction": "Rewrite this naturally while preserving every fact.",
        "generation_attempt": 1,
    }
    verified = {
        "document_fingerprint": fingerprint,
        "same_language": True,
        "all_target_facts_supported_by_source": True,
        "no_source_fact_outside_target": True,
        "names_preserved": True,
        "numbers_dates_quotes_preserved": True,
        "semantic_similarity": 0.98,
        "missing_facts": [],
        "unsupported_source_claims": [],
    }
    return assemble_rewrite_task(
        source=row,
        generated=generated,
        verified=verified,
        assignment=deterministic_assignment(fingerprint),
        token_counter=tokens,
    )


def test_generation_prompt_preserves_token_semantics() -> None:
    prompt = render_generation_prompt(source(1))
    assert "approximately 48 tokens" in prompt
    assert "EM DASHES ALLOWED: no" in prompt
    assert "$12.4 million" in prompt


def test_mechanical_smoke_assembles_exact_rewrite_generation_mix() -> None:
    sources = [source(index) for index in range(128)]
    rewrites = [rewrite(row) for index, row in enumerate(sources) if index % 4 != 3]
    tasks = assemble_mechanical_smoke_corpus(sources, rewrites, token_counter=tokens)
    assert len(tasks) == 128
    assert all(row["artifact_schema"] == TRAINING_TASK_PROTOCOL for row in tasks)
    assert sum(row["task_mode"] == "rewrite" for row in tasks) == 96
    assert sum(row["task_mode"] == "generate" for row in tasks) == 32
    assert len({row["fingerprint"] for row in tasks}) == 128
