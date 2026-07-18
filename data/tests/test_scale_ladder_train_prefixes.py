from __future__ import annotations

import hashlib
import json
import re

import pytest

from data.lower_variance_pipeline import CLEANING_MODE, CLEANING_MODEL
from data.materialize_scale_ladder_train_prefixes import materialize


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _pair(index: int, *, split: str = "train") -> tuple[dict, dict]:
    source_text = (
        f"Header train {index}\n"
        f"Deck train {index}\n\n"
        f"The first substantive line for train document {index} carries enough words.\n"
        f"The second substantive line for train document {index} remains unchanged.\n"
        "Subscribe"
    )
    completion = (
        f"Header train {index}\n"
        f"Deck train {index}\n\n"
        f"The first substantive line for train document {index} carries enough words.\n"
        f"The second substantive line for train document {index} remains unchanged."
    )
    domain = f"scale-train-{index}.example"
    source = {
        "completion": source_text,
        "domain": domain,
        "fineweb_id": f"scale-train-{index}",
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


def _write_jsonl(path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_materialize_scale_train_prefixes_writes_frozen_bundle(tmp_path):
    pairs = [_pair(index) for index in range(8)]
    raw_train = [pair[0] for pair in pairs]
    clean_train = [pair[1] for pair in pairs]

    raw_train_path = tmp_path / "raw-train.jsonl"
    clean_train_path = tmp_path / "clean-train.jsonl"
    source_manifest_path = tmp_path / "source-manifest.json"
    output_dir = tmp_path / "scale-train-prefixes"

    raw_train_sha256 = _write_jsonl(raw_train_path, raw_train)
    _write_jsonl(clean_train_path, clean_train)
    source_manifest_path.write_text(
        json.dumps(
            {
                "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
                "policy": {"candidate_outputs_opened": False},
                "train": {
                    "sha256": raw_train_sha256,
                    "domains": [row["domain"] for row in raw_train],
                    "fingerprints": [row["fingerprint"] for row in raw_train],
                },
                "dev": {"domains": [], "fingerprints": []},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = materialize(
        raw_train_path=raw_train_path,
        clean_train_path=clean_train_path,
        source_manifest_path=source_manifest_path,
        output_dir=output_dir,
        expected_clean_records=8,
        prefix_counts=(4, 8),
    )

    assert bundle["artifact_schema"] == "dftr.scale_ladder.train_prefix_bundle.v1"
    assert bundle["artifacts"]["clean_train_4"]["rows"] == 4
    assert bundle["artifacts"]["clean_train_8"]["rows"] == 8
    assert (output_dir / "clean-train-4.jsonl").is_file()
    assert (output_dir / "clean-train-8.jsonl").is_file()
    assert (output_dir / "train_prefix_bundle.json").is_file()


def test_materialize_scale_train_prefixes_rejects_dev_overlap(tmp_path):
    pairs = [_pair(index) for index in range(4)]
    raw_train = [pair[0] for pair in pairs]
    clean_train = [pair[1] for pair in pairs]

    raw_train_path = tmp_path / "raw-train.jsonl"
    clean_train_path = tmp_path / "clean-train.jsonl"
    source_manifest_path = tmp_path / "source-manifest.json"
    output_dir = tmp_path / "scale-train-prefixes"

    raw_train_sha256 = _write_jsonl(raw_train_path, raw_train)
    _write_jsonl(clean_train_path, clean_train)
    source_manifest_path.write_text(
        json.dumps(
            {
                "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
                "policy": {"candidate_outputs_opened": False},
                "train": {
                    "sha256": raw_train_sha256,
                    "domains": [row["domain"] for row in raw_train],
                    "fingerprints": [row["fingerprint"] for row in raw_train],
                },
                "dev": {
                    "domains": [raw_train[0]["domain"]],
                    "fingerprints": [raw_train[0]["fingerprint"]],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="historical domain"):
        materialize(
            raw_train_path=raw_train_path,
            clean_train_path=clean_train_path,
            source_manifest_path=source_manifest_path,
            output_dir=output_dir,
            expected_clean_records=4,
            prefix_counts=(2, 4),
        )
