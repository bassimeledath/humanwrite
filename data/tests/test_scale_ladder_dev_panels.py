from __future__ import annotations

import hashlib
import json
import re

import pytest

from data.materialize_scale_ladder_dev_panels import (
    PARTITION_SEED,
    materialize,
)
from data.lower_variance_pipeline import (
    CLEANING_MODE,
    CLEANING_MODEL,
    EVAL_COUNT,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _pair(index: int, *, prefix: str, split: str) -> tuple[dict, dict]:
    source_text = (
        f"Header {prefix} {index}\n"
        f"Deck {prefix} {index}\n\n"
        f"The first substantive line for {prefix} document {index} carries enough words.\n"
        f"The second substantive line for {prefix} document {index} remains unchanged.\n"
        "Subscribe"
    )
    completion = (
        f"Header {prefix} {index}\n"
        f"Deck {prefix} {index}\n\n"
        f"The first substantive line for {prefix} document {index} carries enough words.\n"
        f"The second substantive line for {prefix} document {index} remains unchanged."
    )
    domain = f"{prefix}-{index}.example"
    source = {
        "completion": source_text,
        "domain": domain,
        "fineweb_id": f"{prefix}-{index}",
        "fingerprint": _sha(source_text),
        "source_config": "CC-MAIN-2024-10",
        "source_revision": "a" * 40,
        "split": split,
        "url": f"https://{domain}/article/{index}",
        "word_count": len(_WORD_RE.findall(source_text)),
    }
    cleaned = {
        **source,
        "completion": completion,
        "fingerprint": _sha(completion),
        "word_count": len(_WORD_RE.findall(completion)),
        "source_fingerprint": source["fingerprint"],
        "source_word_count": source["word_count"],
        "cleaning_model": CLEANING_MODEL,
        "cleaning_mode": CLEANING_MODE,
    }
    return source, cleaned


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_materialize_scale_dev_panels_writes_frozen_bundle(tmp_path):
    eval_pairs = [_pair(index, prefix="scale-dev", split="dev") for index in range(EVAL_COUNT)]
    train_pairs = [_pair(index, prefix="scale-train", split="train") for index in range(32)]
    raw_dev = [pair[0] for pair in eval_pairs]
    clean_dev = [pair[1] for pair in eval_pairs]
    raw_train = [pair[0] for pair in train_pairs]

    raw_dev_path = tmp_path / "raw-dev.jsonl"
    clean_dev_path = tmp_path / "clean-dev.jsonl"
    raw_train_path = tmp_path / "raw-train.jsonl"
    source_manifest_path = tmp_path / "source-manifest.json"
    output_dir = tmp_path / "scale-dev-panels"

    _write_jsonl(raw_dev_path, raw_dev)
    _write_jsonl(clean_dev_path, clean_dev)
    _write_jsonl(raw_train_path, raw_train)
    source_manifest_path.write_text(
        json.dumps(
            {
                "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
                "policy": {"candidate_outputs_opened": False},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = materialize(
        raw_dev_path=raw_dev_path,
        clean_dev_path=clean_dev_path,
        raw_train_path=raw_train_path,
        source_manifest_path=source_manifest_path,
        output_dir=output_dir,
    )

    assert bundle["artifact_schema"] == "dftr.scale_ladder.dev_panel_bundle.v1"
    assert bundle["partition_seed"] == PARTITION_SEED
    assert bundle["artifacts"]["prompt_sources"]["rows"] == 128
    assert bundle["artifacts"]["distribution_references"]["rows"] == 256
    assert bundle["artifacts"]["human_floor_a"]["rows"] == 128
    assert bundle["artifacts"]["human_floor_b"]["rows"] == 128
    assert (output_dir / "prompt_sources.jsonl").is_file()
    assert (output_dir / "panel_bundle.json").is_file()


def test_materialize_scale_dev_panels_fails_if_train_domain_overlaps(tmp_path):
    eval_pairs = [_pair(index, prefix="scale-dev", split="dev") for index in range(EVAL_COUNT)]
    raw_dev = [pair[0] for pair in eval_pairs]
    clean_dev = [pair[1] for pair in eval_pairs]
    overlapping_train = dict(raw_dev[0])
    overlapping_train["split"] = "train"

    raw_dev_path = tmp_path / "raw-dev.jsonl"
    clean_dev_path = tmp_path / "clean-dev.jsonl"
    raw_train_path = tmp_path / "raw-train.jsonl"
    source_manifest_path = tmp_path / "source-manifest.json"
    output_dir = tmp_path / "scale-dev-panels"

    _write_jsonl(raw_dev_path, raw_dev)
    _write_jsonl(clean_dev_path, clean_dev)
    _write_jsonl(raw_train_path, [overlapping_train])
    source_manifest_path.write_text(
        json.dumps(
            {
                "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
                "policy": {"candidate_outputs_opened": False},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="historical domain"):
        materialize(
            raw_dev_path=raw_dev_path,
            clean_dev_path=clean_dev_path,
            raw_train_path=raw_train_path,
            source_manifest_path=source_manifest_path,
            output_dir=output_dir,
        )
