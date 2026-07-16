from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from data.pipeline import normalize_whitespace


ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


class Tier1BankError(ValueError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Tier1BankError("Tier-1 bank config must be a JSON object")
    return value


def _excluded_fingerprints(config: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in config.get("exclude_manifests") or []:
        manifest = json.loads(_resolve(raw_path).read_text(encoding="utf-8"))
        fingerprints = manifest.get("fingerprints")
        if not isinstance(fingerprints, list):
            raise Tier1BankError(f"excluded manifest lacks fingerprints: {raw_path}")
        excluded.update(str(value) for value in fingerprints)
    return excluded


def _non_latin_letter_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for character in letters if "LATIN" in unicodedata.name(character, ""))
    return (len(letters) - latin) / len(letters)


def _source_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("fineweb_id") or row.get("url") or "").strip()


def _candidate(
    row: dict[str, Any],
    *,
    source_config: str,
    source_revision: str,
    minimum_words: int,
    maximum_words: int,
    maximum_non_latin_ratio: float,
    excluded: set[str],
) -> dict[str, Any] | None:
    source_id = _source_id(row)
    url = str(row.get("url") or "").strip()
    raw_text = row.get("text")
    if not source_id or not url or not isinstance(raw_text, str):
        return None
    text = normalize_whitespace(raw_text)
    word_count = len(WORD_RE.findall(text))
    if word_count < minimum_words or word_count > maximum_words:
        return None
    if _non_latin_letter_ratio(text) > maximum_non_latin_ratio:
        return None
    fingerprint = _sha256_bytes(text.encode("utf-8"))
    if fingerprint in excluded:
        return None
    domain = (urlparse(url).hostname or "").casefold()
    if not domain:
        return None
    return {
        "completion": text,
        "domain": domain,
        "fineweb_id": source_id,
        "fingerprint": fingerprint,
        "source_config": source_config,
        "source_revision": source_revision,
        "split": "tier1_visible_human",
        "url": url,
        "word_count": word_count,
    }


def select_bank(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = config.get("source") or {}
    selection = config.get("selection") or {}
    bank_size = int(selection.get("bank_size", 0))
    pool_size = int(selection.get("eligible_pool_size", 0))
    scan_limit = int(selection.get("stream_scan_limit", 0))
    if bank_size < 4:
        raise Tier1BankError("Tier-1 bank_size must be at least four")
    if pool_size < bank_size or scan_limit < pool_size:
        raise Tier1BankError("selection requires scan_limit >= pool_size >= bank_size")
    excluded = _excluded_fingerprints(config)
    candidates: dict[str, dict[str, Any]] = {}
    scanned = 0
    for row in rows:
        if scanned >= scan_limit or len(candidates) >= pool_size:
            break
        scanned += 1
        candidate = _candidate(
            row,
            source_config=str(source.get("dataset_config") or ""),
            source_revision=str(source.get("revision") or ""),
            minimum_words=int(selection.get("min_word_count", 0)),
            maximum_words=int(selection.get("max_word_count", 0)),
            maximum_non_latin_ratio=float(selection.get("max_non_latin_letter_ratio", 0.0)),
            excluded=excluded,
        )
        if candidate is not None:
            candidates.setdefault(candidate["fingerprint"], candidate)
    if len(candidates) < bank_size:
        raise Tier1BankError(
            f"only {len(candidates)} eligible unique records found after scanning {scanned}"
        )
    seed_label = str(selection.get("seed_label") or "")
    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            _sha256_bytes(f"{seed_label}:{row['fingerprint']}".encode("utf-8")),
            row["fingerprint"],
        ),
    )
    selected: list[dict[str, Any]] = []
    used_domains: set[str] = set()
    if bool(selection.get("require_distinct_domains", True)):
        for row in ranked:
            if row["domain"] in used_domains:
                continue
            selected.append(row)
            used_domains.add(row["domain"])
            if len(selected) == bank_size:
                break
    if len(selected) < bank_size:
        already_selected = {row["fingerprint"] for row in selected}
        selected.extend(
            row for row in ranked if row["fingerprint"] not in already_selected
        )
        selected = selected[:bank_size]
    if len({row["fingerprint"] for row in selected}) != bank_size:
        raise Tier1BankError("selected Tier-1 records are not fingerprint-unique")
    if bool(selection.get("require_distinct_domains", True)) and len(used_domains) < bank_size:
        raise Tier1BankError(
            f"distinct-domain requirement failed: only {len(used_domains)} domains for {bank_size} records"
        )
    return selected, {
        "bank_size": len(selected),
        "eligible_unique_count": len(candidates),
        "excluded_fingerprint_count": len(excluded),
        "scanned_count": scanned,
        "unique_domain_count": len({row["domain"] for row in selected}),
    }


def _stream_source(config: dict[str, Any]):
    source = config.get("source") or {}
    required = ("dataset_id", "dataset_config", "revision", "split")
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise Tier1BankError("source is missing fields: " + ", ".join(missing))
    from datasets import load_dataset

    return load_dataset(
        str(source["dataset_id"]),
        name=str(source["dataset_config"]),
        split=str(source["split"]),
        revision=str(source["revision"]),
        streaming=True,
    )


def materialize(config_path: str | Path, rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    resolved_config = _resolve(config_path)
    config = _load_config(resolved_config)
    selected, counts = select_bank(_stream_source(config) if rows is None else rows, config)
    output = config.get("output") or {}
    bank_path = _resolve(str(output.get("bank_path") or ""))
    manifest_path = _resolve(str(output.get("manifest_path") or ""))
    if bank_path == ROOT or manifest_path == ROOT:
        raise Tier1BankError("output bank_path and manifest_path are required")
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(_canonical_json(row) + "\n" for row in selected)
    bank_path.write_text(payload, encoding="utf-8")
    manifest = {
        "artifact_schema": "dftr.tier1_human_bank.manifest.v1",
        "bank_path": _display_path(bank_path),
        "bank_sha256": _sha256_bytes(payload.encode("utf-8")),
        "config_path": _display_path(resolved_config),
        "config_sha256": _sha256_bytes(resolved_config.read_bytes()),
        "counts": counts,
        "domains": [row["domain"] for row in selected],
        "fingerprints": [row["fingerprint"] for row in selected],
        "policy": config.get("policy") or {},
        "selection": config.get("selection") or {},
        "source": config.get("source") or {},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the frozen visible Tier-1 human bank")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(materialize(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
