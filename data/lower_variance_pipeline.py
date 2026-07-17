"""Qualification and deterministic partitioning for the lower-variance corpus.

The Qwen cleaning artifact overwrites ``completion`` with the selected lines, so
qualification deliberately requires the corresponding raw source records.  This
keeps the line-subset and provenance checks independent of the model response.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


CLEANING_MODEL = "qwen/qwen3-32b"
CLEANING_MODE = "ordered_original_line_subset.v1"
DEFAULT_PARTITION_SEED = "dftr-m2-lower-variance-partition-v1"

EVAL_COUNT = 640
TRAIN_COUNT = 1024
PROMPT_SOURCE_COUNT = 128
DISTRIBUTION_REFERENCE_COUNT = 256
FLOOR_A_COUNT = 128
FLOOR_B_COUNT = 128

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_PROVENANCE_FIELDS = (
    "domain",
    "fineweb_id",
    "source_config",
    "source_revision",
    "split",
    "url",
)


class LowerVarianceQualificationError(ValueError):
    """Raised when a pool cannot be safely used by the lower-variance study."""


@dataclass(frozen=True)
class LowerVariancePools:
    """The four disjoint evaluation panels plus the separate training pool."""

    prompt_sources: tuple[dict[str, Any], ...]
    distribution_references: tuple[dict[str, Any], ...]
    floor_a: tuple[dict[str, Any], ...]
    floor_b: tuple[dict[str, Any], ...]
    training: tuple[dict[str, Any], ...]

    def evaluation(self) -> tuple[dict[str, Any], ...]:
        return (
            self.prompt_sources
            + self.distribution_references
            + self.floor_a
            + self.floor_b
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise LowerVarianceQualificationError(f"{label} has invalid {field}")
    return value


def _required_count(row: Mapping[str, Any], field: str, *, label: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LowerVarianceQualificationError(f"{label} has invalid {field}")
    return value


def _required_sha256(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = _required_text(row, field, label=label)
    if _SHA256_RE.fullmatch(value) is None:
        raise LowerVarianceQualificationError(
            f"{label} has invalid lowercase SHA-256 field {field}"
        )
    return value


def _is_exact_ordered_line_subset(cleaned: str, source: str) -> bool:
    if not cleaned or cleaned != cleaned.strip():
        return False
    source_lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines = cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    source_index = 0
    for cleaned_line in cleaned_lines:
        while source_index < len(source_lines) and source_lines[source_index] != cleaned_line:
            source_index += 1
        if source_index == len(source_lines):
            return False
        source_index += 1
    return True


def _source_index(
    source_rows: Iterable[Mapping[str, Any]], *, pool_name: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(source_rows):
        label = f"{pool_name} source row {position}"
        fingerprint = _required_sha256(row, "fingerprint", label=label)
        completion = _required_text(row, "completion", label=label)
        if _sha256_text(completion) != fingerprint:
            raise LowerVarianceQualificationError(
                f"{label} fingerprint does not bind source completion"
            )
        word_count = _required_count(row, "word_count", label=label)
        if word_count != len(_WORD_RE.findall(completion)):
            raise LowerVarianceQualificationError(
                f"{label} source word_count is invalid"
            )
        if fingerprint in indexed:
            raise LowerVarianceQualificationError(
                f"{pool_name} source fingerprints are not unique"
            )
        indexed[fingerprint] = row
    return indexed


def _normalized_exclusions(values: Iterable[str], *, domains: bool) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise LowerVarianceQualificationError("historical exclusions contain an invalid value")
        candidate = value.casefold() if domains else value
        if not domains and _SHA256_RE.fullmatch(candidate) is None:
            raise LowerVarianceQualificationError(
                "historical fingerprints must be lowercase SHA-256 values"
            )
        normalized.add(candidate)
    return normalized


def qualify_clean_pool(
    cleaned_rows: Iterable[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
    pool_name: str,
    historical_fingerprints: Iterable[str] = (),
    historical_domains: Iterable[str] = (),
    expected_split: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate one cleaned pool and return stable, fingerprint-sorted copies.

    Historical fingerprint checks cover both raw source and cleaned-text
    identities.  Domain comparisons are case-insensitive.
    """

    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise LowerVarianceQualificationError("expected_count must be a positive integer")
    if not isinstance(pool_name, str) or not pool_name:
        raise LowerVarianceQualificationError("pool_name must be non-empty")

    source_by_fingerprint = _source_index(source_rows, pool_name=pool_name)
    excluded_fingerprints = _normalized_exclusions(
        historical_fingerprints, domains=False
    )
    excluded_domains = _normalized_exclusions(historical_domains, domains=True)
    qualified: list[dict[str, Any]] = []
    clean_fingerprints: set[str] = set()
    source_fingerprints: set[str] = set()
    fingerprint_owners: dict[str, str] = {}
    domains: set[str] = set()

    for position, raw_cleaned in enumerate(cleaned_rows):
        row = dict(raw_cleaned)
        label = f"{pool_name} cleaned row {position}"
        source_fingerprint = _required_sha256(
            row, "source_fingerprint", label=label
        )
        clean_fingerprint = _required_sha256(row, "fingerprint", label=label)
        if source_fingerprint in source_fingerprints:
            raise LowerVarianceQualificationError(
                f"{pool_name} source fingerprints are not unique"
            )
        if clean_fingerprint in clean_fingerprints:
            raise LowerVarianceQualificationError(
                f"{pool_name} cleaned fingerprints are not unique"
            )
        for fingerprint in (source_fingerprint, clean_fingerprint):
            owner = fingerprint_owners.get(fingerprint)
            if owner is not None and owner != source_fingerprint:
                raise LowerVarianceQualificationError(
                    f"{pool_name} fingerprints are not unique across records"
                )
        source = source_by_fingerprint.get(source_fingerprint)
        if source is None:
            raise LowerVarianceQualificationError(
                f"{label} does not bind to a supplied source record"
            )

        source_completion = _required_text(source, "completion", label=f"{label} source")
        clean_completion = _required_text(row, "completion", label=label)
        if _sha256_text(clean_completion) != clean_fingerprint:
            raise LowerVarianceQualificationError(
                f"{label} fingerprint does not bind cleaned completion"
            )
        if not _is_exact_ordered_line_subset(clean_completion, source_completion):
            raise LowerVarianceQualificationError(
                f"{label} completion is not an exact ordered source-line subset"
            )
        if row.get("cleaning_model") != CLEANING_MODEL:
            raise LowerVarianceQualificationError(
                f"{label} was not cleaned by the frozen Qwen model"
            )
        if row.get("cleaning_mode") != CLEANING_MODE:
            raise LowerVarianceQualificationError(
                f"{label} has invalid cleaning mode"
            )

        for field in _PROVENANCE_FIELDS:
            source_value = _required_text(source, field, label=f"{label} source")
            cleaned_value = _required_text(row, field, label=label)
            if cleaned_value != source_value:
                raise LowerVarianceQualificationError(
                    f"{label} does not preserve source {field}"
                )
        if expected_split is not None and row["split"] != expected_split:
            raise LowerVarianceQualificationError(
                f"{label} must have source split {expected_split}"
            )
        source_word_count = _required_count(source, "word_count", label=f"{label} source")
        if _required_count(row, "source_word_count", label=label) != source_word_count:
            raise LowerVarianceQualificationError(
                f"{label} does not preserve source word_count"
            )
        clean_word_count = _required_count(row, "word_count", label=label)
        if clean_word_count != len(_WORD_RE.findall(clean_completion)):
            raise LowerVarianceQualificationError(f"{label} cleaned word_count is invalid")

        domain = _required_text(row, "domain", label=label).casefold()
        hostname = (urlparse(_required_text(row, "url", label=label)).hostname or "").casefold()
        if not hostname or hostname != domain:
            raise LowerVarianceQualificationError(
                f"{label} domain does not match URL hostname"
            )
        if domain in domains:
            raise LowerVarianceQualificationError(f"{pool_name} domains are not unique")
        if domain in excluded_domains:
            raise LowerVarianceQualificationError(
                f"{pool_name} overlaps a historical domain"
            )
        if {source_fingerprint, clean_fingerprint} & excluded_fingerprints:
            raise LowerVarianceQualificationError(
                f"{pool_name} overlaps a historical fingerprint"
            )

        source_fingerprints.add(source_fingerprint)
        clean_fingerprints.add(clean_fingerprint)
        fingerprint_owners[source_fingerprint] = source_fingerprint
        fingerprint_owners[clean_fingerprint] = source_fingerprint
        domains.add(domain)
        qualified.append(row)

    if len(qualified) != expected_count:
        raise LowerVarianceQualificationError(
            f"{pool_name} requires exactly {expected_count} cleaned records; "
            f"found {len(qualified)}"
        )
    qualified.sort(key=lambda row: (row["fingerprint"], row["source_fingerprint"]))
    return tuple(qualified)


def partition_eval_pool(
    qualified_eval_rows: Sequence[Mapping[str, Any]],
    *,
    seed: str = DEFAULT_PARTITION_SEED,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    """Hash-rank 640 qualified eval documents into the frozen panel sizes."""

    if len(qualified_eval_rows) != EVAL_COUNT:
        raise LowerVarianceQualificationError(
            f"evaluation pool requires exactly {EVAL_COUNT} records"
        )
    if not isinstance(seed, str) or not seed:
        raise LowerVarianceQualificationError("partition seed must be non-empty")
    rows = [dict(row) for row in qualified_eval_rows]
    identities = []
    for position, row in enumerate(rows):
        label = f"qualified evaluation row {position}"
        clean_fingerprint = _required_sha256(row, "fingerprint", label=label)
        source_fingerprint = _required_sha256(row, "source_fingerprint", label=label)
        identities.append((clean_fingerprint, source_fingerprint))
    if len(set(identities)) != EVAL_COUNT:
        raise LowerVarianceQualificationError("qualified evaluation identities are not unique")
    ranked = sorted(
        rows,
        key=lambda row: (
            _sha256_text(f"{seed}:{row['source_fingerprint']}:{row['fingerprint']}"),
            row["source_fingerprint"],
            row["fingerprint"],
        ),
    )
    prompt_end = PROMPT_SOURCE_COUNT
    distribution_end = prompt_end + DISTRIBUTION_REFERENCE_COUNT
    floor_a_end = distribution_end + FLOOR_A_COUNT
    return (
        tuple(ranked[:prompt_end]),
        tuple(ranked[prompt_end:distribution_end]),
        tuple(ranked[distribution_end:floor_a_end]),
        tuple(ranked[floor_a_end:]),
    )


def qualify_and_partition(
    *,
    eval_cleaned_rows: Iterable[Mapping[str, Any]],
    eval_source_rows: Iterable[Mapping[str, Any]],
    train_cleaned_rows: Iterable[Mapping[str, Any]],
    train_source_rows: Iterable[Mapping[str, Any]],
    historical_fingerprints: Iterable[str] = (),
    historical_domains: Iterable[str] = (),
    seed: str = DEFAULT_PARTITION_SEED,
) -> LowerVariancePools:
    """Qualify both clean pools, enforce separation, and partition evaluation."""

    # Materialize once because callers commonly pass generators and both pools
    # must see exactly the same frozen historical exclusions.
    history_fingerprints = tuple(historical_fingerprints)
    history_domains = tuple(historical_domains)
    evaluation = qualify_clean_pool(
        eval_cleaned_rows,
        eval_source_rows,
        expected_count=EVAL_COUNT,
        pool_name="evaluation",
        historical_fingerprints=history_fingerprints,
        historical_domains=history_domains,
        expected_split="dev",
    )
    training = qualify_clean_pool(
        train_cleaned_rows,
        train_source_rows,
        expected_count=TRAIN_COUNT,
        pool_name="training",
        historical_fingerprints=history_fingerprints,
        historical_domains=history_domains,
        expected_split="train",
    )

    evaluation_domains = {row["domain"].casefold() for row in evaluation}
    training_domains = {row["domain"].casefold() for row in training}
    if evaluation_domains & training_domains:
        raise LowerVarianceQualificationError(
            "evaluation and training pools overlap by domain"
        )
    evaluation_fingerprints = {
        value
        for row in evaluation
        for value in (row["fingerprint"], row["source_fingerprint"])
    }
    training_fingerprints = {
        value
        for row in training
        for value in (row["fingerprint"], row["source_fingerprint"])
    }
    if evaluation_fingerprints & training_fingerprints:
        raise LowerVarianceQualificationError(
            "evaluation and training pools overlap by fingerprint"
        )

    prompt_sources, distribution_references, floor_a, floor_b = partition_eval_pool(
        evaluation, seed=seed
    )
    return LowerVariancePools(
        prompt_sources=prompt_sources,
        distribution_references=distribution_references,
        floor_a=floor_a,
        floor_b=floor_b,
        training=training,
    )
