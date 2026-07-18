"""Qualify and freeze the clean lower-variance measurement-v3 panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data.lower_variance_pipeline import (
    CLEANING_MODE,
    CLEANING_MODEL,
    DEFAULT_PARTITION_SEED,
    LowerVariancePools,
    qualify_and_partition,
)


PANEL_SCHEMA = "dftr.measurement.qualified_panel.v3"
ROLE_NAMES = (
    "prompt_sources",
    "distribution_references",
    "human_floor_a",
    "human_floor_b",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _manifest_record(role: str, row: dict[str, Any]) -> dict[str, str]:
    common = {
        "source_document_id": str(row["source_fingerprint"]),
        "content_sha256": str(row["fingerprint"]),
        "domain": str(row["domain"]).casefold(),
    }
    if role == "prompt_sources":
        return {"prompt_id": f"prompt-{row['fingerprint']}", **common}
    return {"document_id": str(row["fingerprint"]), **common}


def materialize(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    paths = {
        "raw_eval": input_dir / "raw-eval-pool.jsonl",
        "raw_train": input_dir / "raw-train-pool.jsonl",
        "clean_eval": input_dir / "clean-eval-640.jsonl",
        "clean_train": input_dir / "clean-train-1024.jsonl",
        "source_manifest": input_dir / "raw-source-manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing lower-variance inputs: {missing}")
    source_manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    if (
        not isinstance(source_manifest, dict)
        or (source_manifest.get("policy") or {}).get("candidate_outputs_opened") is not False
        or (source_manifest.get("policy") or {}).get("hidden_test_materialized") is not False
    ):
        raise ValueError("source manifest does not preserve the prospective boundary")

    pools: LowerVariancePools = qualify_and_partition(
        eval_cleaned_rows=_load_jsonl(paths["clean_eval"]),
        eval_source_rows=_load_jsonl(paths["raw_eval"]),
        train_cleaned_rows=_load_jsonl(paths["clean_train"]),
        train_source_rows=_load_jsonl(paths["raw_train"]),
        seed=DEFAULT_PARTITION_SEED,
    )
    source_files = {name: _file_sha256(path) for name, path in paths.items()}
    source_bundle_sha256 = _canonical_hash(source_files)
    qualification_contract = {
        "schema": "dftr.lower_variance.qualification_contract.v1",
        "cleaning_model": CLEANING_MODEL,
        "cleaning_mode": CLEANING_MODE,
        "partition_seed": DEFAULT_PARTITION_SEED,
        "required_counts": {
            "training": 1024,
            "prompt_sources": 128,
            "distribution_references": 256,
            "human_floor_a": 128,
            "human_floor_b": 128,
        },
        "constraints": [
            "exact_ordered_original_line_subset",
            "unique_domains_within_and_across_train_eval",
            "unique_source_and_clean_fingerprints",
            "train_split_train_and_eval_split_dev",
            "source_manifest_candidate_outputs_unopened",
        ],
        "source_bundle_sha256": source_bundle_sha256,
    }
    qualification_contract_sha256 = _canonical_hash(qualification_contract)
    role_rows = {
        "prompt_sources": pools.prompt_sources,
        "distribution_references": pools.distribution_references,
        "human_floor_a": pools.floor_a,
        "human_floor_b": pools.floor_b,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, Any] = {}
    for role in ROLE_NAMES:
        rows = role_rows[role]
        data_path = output_dir / f"{role}.jsonl"
        manifest_path = output_dir / f"{role}.manifest.json"
        _write_jsonl(data_path, rows)
        manifest = {
            "artifact_schema": PANEL_SCHEMA,
            "status": "qualified",
            "frozen": True,
            "role": role,
            "partition_seed": DEFAULT_PARTITION_SEED,
            "qualification_contract_sha256": qualification_contract_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "records": [_manifest_record(role, row) for row in rows],
        }
        _write_json(manifest_path, manifest)
        artifacts[role] = {
            "rows": len(rows),
            "data_path": str(data_path),
            "data_sha256": _file_sha256(data_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _file_sha256(manifest_path),
        }
    _write_json(output_dir / "qualification_contract.json", qualification_contract)
    bundle = {
        "artifact_schema": "dftr.measurement.panel_bundle.v3",
        "status": "frozen",
        "partition_seed": DEFAULT_PARTITION_SEED,
        "source_files": source_files,
        "source_bundle_sha256": source_bundle_sha256,
        "qualification_contract_sha256": qualification_contract_sha256,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "panel_bundle.json", bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.input_dir.resolve(), args.output_dir.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
