"""Qualify the 640-document scale-dev pool and freeze deterministic panel roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data.lower_variance_pipeline import (
    CLEANING_MODE,
    CLEANING_MODEL,
    partition_eval_pool,
    qualify_clean_pool,
)


PARTITION_SEED = "dftr-m2-scale-ladder-dev-partition-v1"
ROLES = ("prompt_sources", "distribution_references", "human_floor_a", "human_floor_b")


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


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize(
    *,
    raw_dev_path: Path,
    clean_dev_path: Path,
    raw_train_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    partition_seed: str = PARTITION_SEED,
) -> dict[str, Any]:
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

    raw_train_rows = _rows(raw_train_path)
    historical_fingerprints = tuple(str(row["fingerprint"]) for row in raw_train_rows)
    historical_domains = tuple(str(row["domain"]) for row in raw_train_rows)
    evaluation = qualify_clean_pool(
        _rows(clean_dev_path),
        _rows(raw_dev_path),
        expected_count=640,
        pool_name="scale-dev",
        historical_fingerprints=historical_fingerprints,
        historical_domains=historical_domains,
        expected_split="dev",
    )
    role_values = dict(
        zip(ROLES, partition_eval_pool(evaluation, seed=partition_seed), strict=True)
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    source_files = {
        "raw_dev": _sha(raw_dev_path),
        "clean_dev": _sha(clean_dev_path),
        "raw_train": _sha(raw_train_path),
        "source_manifest": _sha(source_manifest_path),
    }
    qualification_contract = {
        "schema": "dftr.scale_ladder.dev_panel_contract.v1",
        "status": "qualified",
        "candidate_outputs_opened": False,
        "partition_seed": partition_seed,
        "counts": {
            "prompt_sources": 128,
            "distribution_references": 256,
            "human_floor_a": 128,
            "human_floor_b": 128,
        },
        "word_bounds": {"min_word_count": 80, "max_word_count": 220},
        "source_files": source_files,
    }
    _write_json(output_dir / "qualification_contract.json", qualification_contract)
    qualification_contract_sha256 = _sha(output_dir / "qualification_contract.json")

    artifacts = {}
    for role, rows in role_values.items():
        data_path = output_dir / f"{role}.jsonl"
        manifest_path = output_dir / f"{role}.manifest.json"
        _write_jsonl(data_path, rows)
        records = []
        for row in rows:
            common = {
                "source_document_id": str(row["source_fingerprint"]),
                "content_sha256": str(row["fingerprint"]),
                "domain": str(row["domain"]).casefold(),
            }
            identity = (
                {"prompt_id": f"prompt-{row['fingerprint']}"}
                if role == "prompt_sources"
                else {"document_id": str(row["fingerprint"])}
            )
            records.append({**identity, **common})
        manifest = {
            "artifact_schema": "dftr.scale_ladder.qualified_panel.v1",
            "status": "qualified",
            "frozen": True,
            "role": role,
            "partition_seed": partition_seed,
            "qualification_contract_sha256": qualification_contract_sha256,
            "records": records,
        }
        _write_json(manifest_path, manifest)
        artifacts[role] = {
            "rows": len(rows),
            "data_path": str(data_path),
            "data_sha256": _sha(data_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha(manifest_path),
        }

    bundle = {
        "artifact_schema": "dftr.scale_ladder.dev_panel_bundle.v1",
        "status": "frozen",
        "candidate_outputs_opened": False,
        "cleaning_model": CLEANING_MODEL,
        "cleaning_mode": CLEANING_MODE,
        "partition_seed": partition_seed,
        "qualification_contract_sha256": qualification_contract_sha256,
        "source_files": source_files,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "panel_bundle.json", bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dev", type=Path, required=True)
    parser.add_argument("--clean-dev", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition-seed", type=str, default=PARTITION_SEED)
    args = parser.parse_args()
    result = materialize(
        raw_dev_path=args.raw_dev,
        clean_dev_path=args.clean_dev,
        raw_train_path=args.raw_train,
        source_manifest_path=args.source_manifest,
        output_dir=args.output_dir,
        partition_seed=args.partition_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
