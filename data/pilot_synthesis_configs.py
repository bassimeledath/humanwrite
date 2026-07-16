from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PilotSynthesisConfigError(ValueError):
    pass


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_configs(config_path: str | Path) -> dict[str, Any]:
    config_file = _resolve(config_path)
    operator = json.loads(config_file.read_text(encoding="utf-8"))
    manifest_path = _resolve(operator["source_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema") != "dftr.realdata_pilot_source.manifest.v1":
        raise PilotSynthesisConfigError("unexpected source manifest schema")
    output_dir = _resolve(operator["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    results = {}
    for split in ("train", "dev"):
        source = manifest.get(split) or {}
        count = int((manifest.get("counts") or {}).get(f"{split}_count", 0))
        if count <= 0 or not source.get("uri") or not source.get("sha256"):
            raise PilotSynthesisConfigError(f"source manifest lacks complete {split} metadata")
        expected_empty = count // 4
        synthesis = {
            "run": {
                "comparison_id": f"M1-realdata-pilot-briefs-{split}-v1",
                "arm": "SFT",
                "budget_class": "screen",
                "task_kind": "brief_synthesis",
            },
            "compute": {"gpus": 1, "timeout_min": int(operator["timeout_min"])},
            "data": {
                "input_uri": source["uri"],
                "input_sha256": source["sha256"],
                "max_records": count,
                "output_uri": str(operator[f"{split}_output_uri"]),
                "source_split_hash": source["split_hash"],
                "expected_empty_outline_count": expected_empty,
            },
            "api": {
                "model": str(operator["model"]),
                "max_cost_usd": float(operator[f"{split}_max_cost_usd"]),
            },
            "provenance": {
                "source_manifest_sha256": source_manifest_sha,
                "source_revision": str((manifest.get("source") or {}).get("revision") or ""),
                "purpose": "M1 real-data recovery pilot brief synthesis; not promotion evidence",
            },
        }
        if not 0 < synthesis["compute"]["timeout_min"] <= 120:
            raise PilotSynthesisConfigError("pilot synthesis timeout must fit screen budget")
        if not 0 < synthesis["api"]["max_cost_usd"] <= 100:
            raise PilotSynthesisConfigError("pilot synthesis cost cap must be between 0 and 100")
        output_path = output_dir / f"m1_realdata_pilot_briefs_{split}_v1.yaml"
        output_path.write_text(
            yaml.safe_dump(synthesis, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        results[split] = {
            "canonical_config_hash": _canonical_hash(synthesis),
            "config_path": str(output_path.relative_to(ROOT))
            if output_path.is_relative_to(ROOT)
            else str(output_path),
            "input_sha256": source["sha256"],
            "max_cost_usd": synthesis["api"]["max_cost_usd"],
            "record_count": count,
        }
    return {
        "artifact_schema": "dftr.realdata_pilot_synthesis_config_index.v1",
        "source_manifest_sha256": source_manifest_sha,
        "splits": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hash-bound pilot brief-synthesis configs")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(build_configs(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
