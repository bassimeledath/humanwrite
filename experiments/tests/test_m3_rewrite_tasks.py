from __future__ import annotations

import copy

import pytest

from data.rewrite_tasks import (
    PROTOCOL,
    RewriteTaskError,
    assemble_rewrite_task,
    deterministic_assignment,
    protected_literals,
    render_rewrite_prompt,
    rewrite_source_records,
    validate_rewrite_task,
)


FINGERPRINT = "a" * 64


def source() -> dict:
    return {
        "fingerprint": FINGERPRINT,
        "source_fingerprint": "b" * 64,
        "completion": (
            'Acme reported revenue of $12.4 million on March 4, 2025. '
            'Chief executive Ana Diaz said “Demand remained steady.” '
            'Details are available at https://example.com/report.'
        ),
    }


def generated() -> dict:
    return {
        "document_fingerprint": FINGERPRINT,
        "source_text": (
            'On March 4, 2025, Acme reported revenue totaling $12.4 million. '
            'According to chief executive Ana Diaz, “Demand remained steady.” '
            'Additional details can be found at https://example.com/report.'
        ),
        "rewrite_instruction": "Rewrite this in a natural, concise news style without changing facts.",
        "generation_attempt": 1,
    }


def verified() -> dict:
    return {
        "document_fingerprint": FINGERPRINT,
        "same_language": True,
        "all_target_facts_supported_by_source": True,
        "no_source_fact_outside_target": True,
        "names_preserved": True,
        "numbers_dates_quotes_preserved": True,
        "semantic_similarity": 0.97,
        "missing_facts": [],
        "unsupported_source_claims": [],
    }


def tokens(text: str) -> int:
    return len(text.split()) * 2


def test_assignment_is_stable_and_cross_provider() -> None:
    first = deterministic_assignment(FINGERPRINT)
    assert first == deterministic_assignment(FINGERPRINT)
    assert first["generator_model"] != first["verifier_model"]


def test_protected_literals_cover_factual_surface() -> None:
    values = protected_literals(source()["completion"])
    assert "$12.4" in values
    assert "2025" in values
    assert "Demand remained steady." in values
    assert "https://example.com/report." in values


def test_assemble_and_render_valid_pair() -> None:
    row = assemble_rewrite_task(
        source=source(),
        generated=generated(),
        verified=verified(),
        assignment=deterministic_assignment(FINGERPRINT),
        token_counter=tokens,
    )
    assert row["artifact_schema"] == PROTOCOL
    assert row["task_mode"] == "rewrite"
    assert row["completion"] == source()["completion"]
    validate_rewrite_task(row, source=source(), token_counter=tokens)
    prompt = render_rewrite_prompt(row)
    assert row["input_text"] in prompt
    assert row["completion"] not in prompt


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda g, v: g.update(source_text=source()["completion"]), "equals target"),
        (lambda g, v: g.update(source_text=g["source_text"].replace("$12.4", "$12.5")), "protected literals"),
        (lambda g, v: g.update(source_text=g["source_text"] + " 中文"), "non-Latin"),
        (lambda g, v: v.update(unsupported_source_claims=["invented"]), "factual mismatch"),
        (lambda g, v: v.update(semantic_similarity=0.50), "semantic similarity"),
    ],
)
def test_rejects_invalid_pairs(mutation, message: str) -> None:
    g, v = copy.deepcopy(generated()), copy.deepcopy(verified())
    mutation(g, v)
    with pytest.raises(RewriteTaskError, match=message):
        assemble_rewrite_task(
            source=source(),
            generated=g,
            verified=v,
            assignment=deterministic_assignment(FINGERPRINT),
            token_counter=tokens,
        )


def test_rewrite_source_assignment_is_exact_prefix_stable() -> None:
    records = [
        {"fingerprint": format(index, "064x"), "completion": f"record {index}"}
        for index in range(16)
    ]
    first = rewrite_source_records(records[:8])
    full = rewrite_source_records(records)
    assert len(first) == 6
    assert len(full) == 12
    assert [row["fingerprint"] for row in full[:6]] == [
        row["fingerprint"] for row in first
    ]
