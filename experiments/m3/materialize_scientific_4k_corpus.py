"""Assemble and SHA-freeze the exact M3 4K scientific training corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from data.m3_scientific_corpus import (
    BASE_MODEL,
    BASE_REVISION,
    assemble_scientific_training_corpus,
    scientific_manifest,
)
from experiments.m1.contracts import write_json, write_jsonl
from experiments.m3.rewrite_4k_train import build_config, prepare_batch


SOURCE_SHA256 = "723ebf559a4139c49454f5898a0e51120cdf424bd3cd12e39466c6758d25217b"
OUTPUT = Path("/tmp/scientific-training-corpus-4096-v1.jsonl")
MANIFEST_OUTPUT = Path("/tmp/scientific-training-corpus-4096-v1.manifest.json")


class M3MaterializationError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: str, field: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise M3MaterializationError(f"{field} must be lowercase SHA-256")
    return value


def load_bound_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    _require_sha(expected_sha256, str(path))
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise M3MaterializationError(f"artifact hash mismatch: {path}")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise M3MaterializationError(f"artifact is empty: {path}")
    return values


def validate_source_bank(sources: list[dict[str, Any]]) -> None:
    if len(sources) != 4096:
        raise M3MaterializationError("4K source bank must contain exactly 4096 rows")
    if len({str(row.get("fingerprint") or "") for row in sources}) != 4096:
        raise M3MaterializationError("4K source fingerprints are not unique")
    if len({str(row.get("domain") or "").casefold() for row in sources}) != 4096:
        raise M3MaterializationError("4K source domains are not unique")
    for row in sources:
        if (
            row.get("cleaning_model") != "qwen/qwen3-32b"
            or row.get("target_length_unit") != "tokens"
            or type(row.get("target_length")) is not int
            or not 32 <= int(row["target_length"]) <= 383
            or "�" in str(row.get("completion") or "")
        ):
            raise M3MaterializationError("4K source cleaning or target-length contract drifted")


def corpus_audit(rows: list[dict[str, Any]], sources: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    assignments = scientific_manifest(sources)
    origins = Counter(str(row["origin"]) for row in assignments)
    expected_origins = {
        "multi_provider_ai": 1843,
        "baseline_model_draft": 819,
        "controlled_light_edit": 205,
        "already_human_noop": 205,
        "generate": 1024,
    }
    if origins != expected_origins:
        raise M3MaterializationError(f"4K origin counts drifted: {dict(origins)}")
    task_modes = Counter(str(row["task_mode"]) for row in rows)
    if task_modes != {"rewrite": 3072, "generate": 1024}:
        raise M3MaterializationError("4K rewrite/generate counts drifted")
    generated_sources = [source for source, assignment in zip(sources, assignments) if assignment["origin"] == "generate"]
    if sum(not source.get("outline") for source in generated_sources) != 256:
        raise M3MaterializationError("4K generation empty-outline count drifted")
    api_assignments = [
        assignment
        for assignment in assignments
        if assignment["origin"] in {"multi_provider_ai", "controlled_light_edit"}
    ]
    provider_counts = Counter(str(row["generator_model"]) for row in api_assignments)
    if sorted(provider_counts.values()) != [1024, 1024]:
        raise M3MaterializationError("4K API generator balance drifted")
    config = build_config("SFT14", "a" * 64)
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    sequence_lengths: list[int] = []
    for row in rows:
        prepared = prepare_batch(tokenizer, [row], config)
        completion_length = int(prepared["completion_tokens"][0]) - 1
        sequence_length = int(prepared["attention_mask"].sum().item())
        prompt_length = sequence_length - completion_length - 1
        prompt_lengths.append(prompt_length)
        completion_lengths.append(completion_length)
        sequence_lengths.append(sequence_length)
    return {
        "records": len(rows),
        "origin_counts": dict(sorted(origins.items())),
        "task_mode_counts": dict(sorted(task_modes.items())),
        "api_generator_counts": dict(sorted(provider_counts.items())),
        "empty_generation_outlines": 256,
        "prompt_tokens": {"min": min(prompt_lengths), "max": max(prompt_lengths)},
        "completion_tokens": {"min": min(completion_lengths), "max": max(completion_lengths)},
        "sequence_tokens": {"min": min(sequence_lengths), "max": max(sequence_lengths)},
    }


def materialize(
    *,
    source_path: Path,
    source_sha256: str,
    api_path: Path,
    api_sha256: str,
    baseline_path: Path,
    baseline_sha256: str,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    sources = load_bound_jsonl(source_path, source_sha256)
    api_rows = load_bound_jsonl(api_path, api_sha256)
    baseline_rows = load_bound_jsonl(baseline_path, baseline_sha256)
    validate_source_bank(sources)
    if len(api_rows) != 2048 or len(baseline_rows) != 819:
        raise M3MaterializationError("constructed rewrite artifact cardinality mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = assemble_scientific_training_corpus(
        sources,
        api_rows + baseline_rows,
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
    audit = corpus_audit(rows, sources, tokenizer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or manifest_path.exists():
        raise M3MaterializationError("refusing to overwrite frozen 4K corpus artifacts")
    write_jsonl(output_path, rows)
    output_sha256 = file_sha256(output_path)
    manifest = {
        "artifact_schema": "humanwrite.m3.scientific_training_corpus_manifest.v1",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "api_rewrites_path": str(api_path),
        "api_rewrites_sha256": api_sha256,
        "baseline_rewrites_path": str(baseline_path),
        "baseline_rewrites_sha256": baseline_sha256,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "model": BASE_MODEL,
        "model_revision": BASE_REVISION,
        "audit": audit,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--api-sha256", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_OUTPUT)
    args = parser.parse_args()
    manifest = materialize(
        source_path=args.source,
        source_sha256=args.source_sha256,
        api_path=args.api,
        api_sha256=args.api_sha256,
        baseline_path=args.baseline,
        baseline_sha256=args.baseline_sha256,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
