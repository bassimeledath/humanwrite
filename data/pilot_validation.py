from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from data.pipeline import split_hash
from infra.backend.brief_contract import exact_empty_outline_ids, record_id, validate_brief


ROOT = Path(__file__).resolve().parents[1]


class PilotValidationError(ValueError):
    pass


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotValidationError(f"expected JSON object: {resolved}")
    return value


def _load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if resolved.suffix.casefold() in {".yaml", ".yml"}:
        value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    else:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotValidationError(f"expected mapping: {resolved}")
    return value


def _load_jsonl(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    resolved = _resolve(path)
    rows = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PilotValidationError(f"non-object JSONL row at {resolved}:{line_number}")
        rows.append(value)
    return resolved, rows


def _unique_index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = record_id(row)
        if not value or value in result:
            raise PilotValidationError(f"{label} requires unique non-empty record IDs")
        result[value] = row
    return result


def validate_source_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_json(config["source_manifest_path"])
    if manifest.get("artifact_schema") != "dftr.realdata_pilot_source.manifest.v1":
        raise PilotValidationError("unexpected source manifest schema")
    source_config = (
        _load_mapping(config["source_config_path"])
        if config.get("source_config_path")
        else {}
    )
    expected_source = config.get("expected_source") or source_config.get("source") or {}
    for field in ("dataset_id", "dataset_config", "revision", "split", "files"):
        if manifest.get("source", {}).get(field) != expected_source.get(field):
            raise PilotValidationError(f"source manifest {field} mismatch")
    results: dict[str, Any] = {}
    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    all_domains: set[str] = set()
    all_fingerprints: set[str] = set()
    for split in ("train", "dev"):
        path, rows = _load_jsonl(config[f"{split}_source_path"])
        index = _unique_index(rows, f"{split} source")
        indexes[split] = index
        expected = manifest.get(split) or {}
        fingerprints = [row["fingerprint"] for row in rows]
        domains = [str(row.get("domain") or "").casefold() for row in rows]
        if any(row.get("split") != split for row in rows):
            raise PilotValidationError(f"{split} source contains wrong split labels")
        if any(_sha256(str(row.get("completion") or "").encode("utf-8")) != row["fingerprint"] for row in rows):
            raise PilotValidationError(f"{split} source completion fingerprint mismatch")
        if _file_sha256(path) != expected.get("sha256"):
            raise PilotValidationError(f"{split} source file SHA mismatch")
        if split_hash(rows) != expected.get("split_hash"):
            raise PilotValidationError(f"{split} source split hash mismatch")
        if fingerprints != expected.get("fingerprints") or domains != expected.get("domains"):
            raise PilotValidationError(f"{split} source manifest rows mismatch")
        if len(set(domains)) != len(domains):
            raise PilotValidationError(f"{split} source domains are not unique")
        if all_domains.intersection(domains) or all_fingerprints.intersection(fingerprints):
            raise PilotValidationError("train/dev source overlap")
        all_domains.update(domains)
        all_fingerprints.update(fingerprints)
        results[split] = {
            "count": len(rows),
            "file_sha256": expected["sha256"],
            "split_hash": expected["split_hash"],
        }
    counts = manifest.get("counts") or {}
    if results["train"]["count"] != int(counts.get("train_count", -1)):
        raise PilotValidationError("source train count mismatch")
    if results["dev"]["count"] != int(counts.get("dev_count", -1)):
        raise PilotValidationError("source dev count mismatch")
    if len(all_domains) != int(counts.get("unique_domain_count", -1)):
        raise PilotValidationError("source unique-domain count mismatch")
    source_exclusions = source_config.get("exclusions") or {}
    excluded_fingerprints = set(
        config["excluded_fingerprints"]
        if "excluded_fingerprints" in config
        else source_exclusions.get("fingerprints") or []
    )
    excluded_domains = {
        str(value).casefold()
        for value in (
            config["excluded_domains"]
            if "excluded_domains" in config
            else source_exclusions.get("domains") or []
        )
    }
    if all_fingerprints.intersection(excluded_fingerprints):
        raise PilotValidationError("source overlaps excluded fingerprints")
    if all_domains.intersection(excluded_domains):
        raise PilotValidationError("source overlaps excluded domains")
    return {"manifest": manifest, "indexes": indexes, "summary": results}


def validate_brief_artifact(
    *,
    split: str,
    source_index: dict[str, dict[str, Any]],
    path: str | Path,
) -> dict[str, Any]:
    resolved, rows = _load_jsonl(path)
    brief_index = _unique_index(rows, f"{split} briefs")
    if set(brief_index) != set(source_index):
        raise PilotValidationError(f"{split} brief/source ID set mismatch")
    empty_ids = exact_empty_outline_ids(source_index.values())
    observed_empty: set[str] = set()
    preserved_fields = (
        "completion", "domain", "fineweb_id", "fingerprint", "source_config",
        "source_revision", "split", "url", "word_count",
    )
    for source_id, source in source_index.items():
        brief = brief_index[source_id]
        if any(brief.get(field) != source.get(field) for field in preserved_fields):
            raise PilotValidationError(f"{split} brief changed source field for {source_id}")
        if brief.get("generation_mode") != "generate":
            raise PilotValidationError(f"{split} brief generation_mode mismatch")
        is_empty = source_id in empty_ids
        validate_brief(
            brief,
            source_text=str(source["completion"]),
            force_empty_outline=is_empty,
        )
        if not brief.get("outline"):
            observed_empty.add(source_id)
    if observed_empty != empty_ids:
        raise PilotValidationError(f"{split} exact empty-outline ID set mismatch")
    return {
        "briefs_sha256": _file_sha256(resolved),
        "count": len(rows),
        "empty_outline_count": len(observed_empty),
        "empty_outline_ratio": len(observed_empty) / len(rows),
        "source_split_hash": split_hash(source_index.values()),
    }


def validate_pilot(config_path: str | Path) -> dict[str, Any]:
    config = _load_json(config_path)
    source = validate_source_artifacts(config)
    summary = {
        "artifact_schema": "dftr.realdata_pilot_validation.v1",
        "source_manifest_sha256": _file_sha256(_resolve(config["source_manifest_path"])),
        "source": source["summary"],
        "briefs": {},
    }
    for split in ("train", "dev"):
        summary["briefs"][split] = validate_brief_artifact(
            split=split,
            source_index=source["indexes"][split],
            path=config[f"{split}_briefs_path"],
        )
    output_path = _resolve(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real-data pilot source and brief artifacts")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(validate_pilot(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
