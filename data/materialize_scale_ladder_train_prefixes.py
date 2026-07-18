"""Qualify the 16K clean train pool and freeze nested 4K/16K prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from data.lower_variance_pipeline import CLEANING_MODE, CLEANING_MODEL, qualify_clean_pool


EXPECTED_CLEAN_RECORDS = 16_384
PREFIX_COUNTS = (4_096, 16_384)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize(
    *,
    raw_train_path: Path,
    clean_train_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    expected_clean_records: int = EXPECTED_CLEAN_RECORDS,
    prefix_counts: Sequence[int] = PREFIX_COUNTS,
) -> dict[str, Any]:
    if (
        isinstance(expected_clean_records, bool)
        or not isinstance(expected_clean_records, int)
        or expected_clean_records <= 0
    ):
        raise ValueError("expected_clean_records must be a positive integer")
    if (
        not isinstance(prefix_counts, Sequence)
        or not prefix_counts
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in prefix_counts
        )
    ):
        raise ValueError("prefix_counts must be a non-empty sequence of positive integers")
    normalized_prefix_counts = tuple(prefix_counts)
    if tuple(sorted(normalized_prefix_counts)) != normalized_prefix_counts:
        raise ValueError("prefix_counts must be strictly increasing")
    if normalized_prefix_counts[-1] != expected_clean_records:
        raise ValueError("largest prefix must equal expected_clean_records")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("artifact_schema") != "dftr.realdata_pilot_source.manifest.v1":
        raise ValueError("scale-ladder source manifest schema is invalid")
    candidate_outputs_opened = (
        (source_manifest.get("policy") or {}).get("candidate_outputs_opened")
        if "policy" in source_manifest
        else source_manifest.get("candidate_outputs_opened")
    )
    if candidate_outputs_opened is not False:
        raise ValueError("scale-ladder source manifest opened candidate outputs")
    train_manifest = source_manifest.get("train") or {}
    dev_manifest = source_manifest.get("dev") or {}
    manifest_train_sha256 = str(train_manifest.get("sha256") or "")
    if len(manifest_train_sha256) != 64 or _sha(raw_train_path) != manifest_train_sha256:
        raise ValueError("raw train artifact does not match source manifest")

    qualified = qualify_clean_pool(
        _rows(clean_train_path),
        _rows(raw_train_path),
        expected_count=expected_clean_records,
        pool_name="scale-train",
        historical_fingerprints=tuple(str(item) for item in dev_manifest.get("fingerprints") or ()),
        historical_domains=tuple(str(item) for item in dev_manifest.get("domains") or ()),
        expected_split="train",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    source_files = {
        "raw_train": _sha(raw_train_path),
        "clean_train": _sha(clean_train_path),
        "source_manifest": _sha(source_manifest_path),
    }
    qualification_contract = {
        "schema": "dftr.scale_ladder.train_prefix_contract.v1",
        "status": "qualified",
        "candidate_outputs_opened": False,
        "counts": {
            "clean_train": expected_clean_records,
            **{f"clean_train_{count}": count for count in normalized_prefix_counts},
        },
        "word_bounds": {"min_word_count": 80, "max_word_count": 220},
        "prefix_rule": "deterministic fingerprint-sorted prefix of qualified clean train pool",
        "source_files": source_files,
    }
    _write_json(output_dir / "qualification_contract.json", qualification_contract)
    qualification_contract_sha256 = _sha(output_dir / "qualification_contract.json")

    artifacts = {}
    for count in normalized_prefix_counts:
        role = f"clean_train_{count}"
        data_path = output_dir / f"clean-train-{count}.jsonl"
        manifest_path = output_dir / f"clean-train-{count}.manifest.json"
        rows = qualified[:count]
        _write_jsonl(data_path, rows)
        manifest = {
            "artifact_schema": "dftr.scale_ladder.qualified_train_prefix.v1",
            "status": "qualified",
            "frozen": True,
            "role": role,
            "row_count": count,
            "qualification_contract_sha256": qualification_contract_sha256,
            "records": [
                {
                    "document_id": str(row["fingerprint"]),
                    "source_document_id": str(row["source_fingerprint"]),
                    "content_sha256": str(row["fingerprint"]),
                    "domain": str(row["domain"]).casefold(),
                }
                for row in rows
            ],
        }
        _write_json(manifest_path, manifest)
        artifacts[role] = {
            "rows": count,
            "data_path": str(data_path),
            "data_sha256": _sha(data_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha(manifest_path),
        }

    bundle = {
        "artifact_schema": "dftr.scale_ladder.train_prefix_bundle.v1",
        "status": "frozen",
        "candidate_outputs_opened": False,
        "cleaning_model": CLEANING_MODEL,
        "cleaning_mode": CLEANING_MODE,
        "qualification_contract_sha256": qualification_contract_sha256,
        "source_files": source_files,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "train_prefix_bundle.json", bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--clean-train", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        raw_train_path=args.raw_train,
        clean_train_path=args.clean_train,
        source_manifest_path=args.source_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
