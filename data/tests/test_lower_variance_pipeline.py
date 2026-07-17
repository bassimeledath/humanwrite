from __future__ import annotations

import hashlib

import pytest

from data.lower_variance_pipeline import (
    CLEANING_MODE,
    CLEANING_MODEL,
    DISTRIBUTION_REFERENCE_COUNT,
    EVAL_COUNT,
    FLOOR_A_COUNT,
    FLOOR_B_COUNT,
    PROMPT_SOURCE_COUNT,
    TRAIN_COUNT,
    LowerVarianceQualificationError,
    qualify_and_partition,
    qualify_clean_pool,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair(index: int, *, prefix: str) -> tuple[dict, dict]:
    source_text = (
        f"Navigation {prefix} {index}\n"
        f"Article heading {prefix} {index}\n\n"
        f"The first substantive line for document {index} has enough words.\n"
        f"The second substantive line for document {index} remains unchanged.\n"
        "Subscribe"
    )
    completion = (
        f"Article heading {prefix} {index}\n\n"
        f"The first substantive line for document {index} has enough words.\n"
        f"The second substantive line for document {index} remains unchanged."
    )
    domain = f"{prefix}-{index}.example"
    source = {
        "completion": source_text,
        "domain": domain,
        "fineweb_id": f"{prefix}-{index}",
        "fingerprint": _sha(source_text),
        "source_config": "CC-MAIN-2024-10",
        "source_revision": "a" * 40,
        "split": "dev" if prefix == "eval" else "train",
        "url": f"https://{domain}/article/{index}",
        "word_count": len(source_text.split()),
    }
    cleaned = {
        **source,
        "completion": completion,
        "fingerprint": _sha(completion),
        "word_count": len(completion.split()),
        "source_fingerprint": source["fingerprint"],
        "source_word_count": source["word_count"],
        "cleaning_model": CLEANING_MODEL,
        "cleaning_mode": CLEANING_MODE,
    }
    return source, cleaned


def _pool(count: int, *, prefix: str) -> tuple[list[dict], list[dict]]:
    pairs = [_pair(index, prefix=prefix) for index in range(count)]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


@pytest.fixture(scope="module")
def complete_pools():
    eval_sources, eval_cleaned = _pool(EVAL_COUNT, prefix="eval")
    train_sources, train_cleaned = _pool(TRAIN_COUNT, prefix="train")
    return eval_sources, eval_cleaned, train_sources, train_cleaned


def test_qualification_and_partition_is_deterministic_and_disjoint(complete_pools):
    eval_sources, eval_cleaned, train_sources, train_cleaned = complete_pools
    first = qualify_and_partition(
        eval_cleaned_rows=reversed(eval_cleaned),
        eval_source_rows=eval_sources,
        train_cleaned_rows=reversed(train_cleaned),
        train_source_rows=train_sources,
    )
    second = qualify_and_partition(
        eval_cleaned_rows=eval_cleaned,
        eval_source_rows=reversed(eval_sources),
        train_cleaned_rows=train_cleaned,
        train_source_rows=reversed(train_sources),
    )

    assert first == second
    assert len(first.prompt_sources) == PROMPT_SOURCE_COUNT
    assert len(first.distribution_references) == DISTRIBUTION_REFERENCE_COUNT
    assert len(first.floor_a) == FLOOR_A_COUNT
    assert len(first.floor_b) == FLOOR_B_COUNT
    assert len(first.training) == TRAIN_COUNT
    evaluation = first.evaluation()
    assert len(evaluation) == EVAL_COUNT
    assert len({row["fingerprint"] for row in evaluation}) == EVAL_COUNT
    assert len({row["domain"] for row in evaluation}) == EVAL_COUNT
    assert {row["fingerprint"] for row in evaluation}.isdisjoint(
        row["fingerprint"] for row in first.training
    )
    panel_sets = [
        {row["fingerprint"] for row in panel}
        for panel in (
            first.prompt_sources,
            first.distribution_references,
            first.floor_a,
            first.floor_b,
        )
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(panel_sets)
        for right in panel_sets[index + 1 :]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(cleaning_model="another/model"), "frozen Qwen"),
        (lambda row: row.update(cleaning_mode="rewritten.v1"), "cleaning mode"),
        (lambda row: row.update(source_word_count=1), "source word_count"),
        (lambda row: row.update(source_revision="b" * 40), "source source_revision"),
        (lambda row: row.update(domain="other.example"), "source domain"),
    ],
)
def test_qwen_cleaning_provenance_is_required(mutation, message):
    source, cleaned = _pair(0, prefix="eval")
    mutation(cleaned)
    with pytest.raises(LowerVarianceQualificationError, match=message):
        qualify_clean_pool([cleaned], [source], expected_count=1, pool_name="test")


def test_cleaned_text_must_be_an_exact_ordered_line_subset():
    source, cleaned = _pair(0, prefix="eval")
    cleaned["completion"] = cleaned["completion"].replace("unchanged", "rewritten")
    cleaned["fingerprint"] = _sha(cleaned["completion"])
    with pytest.raises(LowerVarianceQualificationError, match="exact ordered source-line subset"):
        qualify_clean_pool([cleaned], [source], expected_count=1, pool_name="test")

    source, cleaned = _pair(0, prefix="eval")
    cleaned["completion"] = "\n".join(reversed(cleaned["completion"].splitlines()))
    cleaned["fingerprint"] = _sha(cleaned["completion"])
    with pytest.raises(LowerVarianceQualificationError, match="exact ordered source-line subset"):
        qualify_clean_pool([cleaned], [source], expected_count=1, pool_name="test")


def test_fingerprints_bind_source_and_cleaned_bytes():
    source, cleaned = _pair(0, prefix="eval")
    source["completion"] += " changed"
    with pytest.raises(LowerVarianceQualificationError, match="bind source completion"):
        qualify_clean_pool([cleaned], [source], expected_count=1, pool_name="test")

    source, cleaned = _pair(0, prefix="eval")
    cleaned["completion"] += " changed"
    with pytest.raises(LowerVarianceQualificationError, match="bind cleaned completion"):
        qualify_clean_pool([cleaned], [source], expected_count=1, pool_name="test")


def test_source_word_count_must_bind_source_bytes():
    source, cleaned = _pair(0, prefix="eval")
    source["word_count"] += 1
    cleaned["source_word_count"] += 1
    with pytest.raises(LowerVarianceQualificationError, match="source word_count is invalid"):
        qualify_clean_pool([cleaned], [source], expected_count=1, pool_name="test")


def test_duplicate_domain_or_fingerprint_fails_closed():
    source_a, cleaned_a = _pair(0, prefix="eval")
    source_b, cleaned_b = _pair(1, prefix="eval")
    source_b["domain"] = source_a["domain"]
    source_b["url"] = source_a["url"]
    cleaned_b["domain"] = source_a["domain"]
    cleaned_b["url"] = source_a["url"]
    with pytest.raises(LowerVarianceQualificationError, match="domains are not unique"):
        qualify_clean_pool(
            [cleaned_a, cleaned_b],
            [source_a, source_b],
            expected_count=2,
            pool_name="test",
        )

    source_b, cleaned_b = _pair(1, prefix="eval")
    cleaned_b["fingerprint"] = cleaned_a["fingerprint"]
    with pytest.raises(LowerVarianceQualificationError, match="cleaned fingerprints"):
        qualify_clean_pool(
            [cleaned_a, cleaned_b],
            [source_a, source_b],
            expected_count=2,
            pool_name="test",
        )


@pytest.mark.parametrize("identity", ["source", "cleaned", "domain"])
def test_historical_disjointness_checks_all_identities(identity):
    source, cleaned = _pair(0, prefix="eval")
    kwargs = {}
    if identity == "source":
        kwargs["historical_fingerprints"] = [source["fingerprint"]]
    elif identity == "cleaned":
        kwargs["historical_fingerprints"] = [cleaned["fingerprint"]]
    else:
        kwargs["historical_domains"] = [cleaned["domain"].upper()]
    with pytest.raises(LowerVarianceQualificationError, match="historical"):
        qualify_clean_pool(
            [cleaned], [source], expected_count=1, pool_name="test", **kwargs
        )


def test_eval_and_training_must_be_separate(complete_pools):
    eval_sources, eval_cleaned, train_sources, train_cleaned = complete_pools
    overlapping_source = dict(train_sources[0], domain=eval_sources[0]["domain"], url=eval_sources[0]["url"])
    overlapping_cleaned = dict(train_cleaned[0], domain=eval_sources[0]["domain"], url=eval_sources[0]["url"])
    with pytest.raises(LowerVarianceQualificationError, match="overlap by domain"):
        qualify_and_partition(
            eval_cleaned_rows=eval_cleaned,
            eval_source_rows=eval_sources,
            train_cleaned_rows=[overlapping_cleaned, *train_cleaned[1:]],
            train_source_rows=[overlapping_source, *train_sources[1:]],
        )


def test_eval_and_training_split_labels_are_frozen(complete_pools):
    eval_sources, eval_cleaned, train_sources, train_cleaned = complete_pools
    bad_source = dict(eval_sources[0], split="train")
    bad_cleaned = dict(eval_cleaned[0], split="train")
    with pytest.raises(LowerVarianceQualificationError, match="source split dev"):
        qualify_and_partition(
            eval_cleaned_rows=[bad_cleaned, *eval_cleaned[1:]],
            eval_source_rows=[bad_source, *eval_sources[1:]],
            train_cleaned_rows=train_cleaned,
            train_source_rows=train_sources,
        )


def test_exact_target_cardinality_is_required():
    sources, cleaned = _pool(2, prefix="eval")
    with pytest.raises(LowerVarianceQualificationError, match="requires exactly 3"):
        qualify_clean_pool(cleaned, sources, expected_count=3, pool_name="test")
