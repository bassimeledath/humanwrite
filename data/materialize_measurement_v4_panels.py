"""Qualify a fresh confirmation pool and freeze measurement-v4 panel roles."""

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


PARTITION_SEED = "dftr-measurement-v4-confirmation-partition-v1"
ROLES = ("prompt_sources", "distribution_references", "human_floor_a", "human_floor_b")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize(
    raw_eval_path: Path,
    clean_eval_path: Path,
    source_manifest_path: Path,
    historical_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    historical = json.loads(historical_manifest_path.read_text(encoding="utf-8"))
    candidate_outputs_opened = (
        (source_manifest.get("policy") or {}).get("candidate_outputs_opened")
        if "policy" in source_manifest
        else source_manifest.get("candidate_outputs_opened")
    )
    if candidate_outputs_opened is not False:
        raise ValueError("fresh source manifest opened candidate outputs")
    historical_fingerprints = tuple(str(item) for item in historical.get("fingerprints") or [])
    historical_domains = tuple(str(item) for item in historical.get("domains") or [])
    evaluation = qualify_clean_pool(
        _rows(clean_eval_path),
        _rows(raw_eval_path),
        expected_count=640,
        pool_name="confirmation-evaluation",
        historical_fingerprints=historical_fingerprints,
        historical_domains=historical_domains,
        expected_split="dev",
    )
    role_values = dict(
        zip(ROLES, partition_eval_pool(evaluation, seed=PARTITION_SEED), strict=True)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    source_files = {
        "raw_eval": _sha(raw_eval_path),
        "clean_eval": _sha(clean_eval_path),
        "source_manifest": _sha(source_manifest_path),
        "historical_manifest": _sha(historical_manifest_path),
    }
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
            "artifact_schema": "dftr.measurement.qualified_panel.v3",
            "status": "qualified",
            "frozen": True,
            "role": role,
            "partition_seed": PARTITION_SEED,
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
        "artifact_schema": "dftr.measurement.panel_bundle.v4",
        "status": "frozen",
        "candidate_outputs_opened": False,
        "cleaning_model": CLEANING_MODEL,
        "cleaning_mode": CLEANING_MODE,
        "partition_seed": PARTITION_SEED,
        "source_files": source_files,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "panel_bundle.json", bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-eval", type=Path, required=True)
    parser.add_argument("--clean-eval", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        args.raw_eval,
        args.clean_eval,
        args.source_manifest,
        args.historical_manifest,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
