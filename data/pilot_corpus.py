from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from data.pipeline import normalize_whitespace, split_hash


ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


class PilotCorpusError(ValueError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotCorpusError("pilot corpus config must be a JSON object")
    return value


def _load_exclusions(config: dict[str, Any]) -> tuple[set[str], set[str]]:
    fingerprints: set[str] = set()
    domains: set[str] = set()
    for raw_path in config.get("exclude_manifests") or []:
        manifest = json.loads(_resolve(raw_path).read_text(encoding="utf-8"))
        raw_fingerprints = manifest.get("fingerprints")
        if not isinstance(raw_fingerprints, list):
            raise PilotCorpusError(f"excluded manifest lacks fingerprints: {raw_path}")
        fingerprints.update(str(value) for value in raw_fingerprints)
        domains.update(str(value).casefold() for value in manifest.get("domains") or [])
        for record in manifest.get("records") or []:
            if isinstance(record, dict) and record.get("domain"):
                domains.add(str(record["domain"]).casefold())
    return fingerprints, domains


def _non_latin_letter_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for character in letters if "LATIN" in unicodedata.name(character, ""))
    return (len(letters) - latin) / len(letters)


def _candidate(
    row: dict[str, Any],
    *,
    source_config: str,
    source_revision: str,
    minimum_words: int,
    maximum_words: int,
    maximum_non_latin_ratio: float,
    excluded_fingerprints: set[str],
    excluded_domains: set[str],
) -> dict[str, Any] | None:
    source_id = str(row.get("id") or row.get("fineweb_id") or row.get("url") or "").strip()
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
    fingerprint = _sha256(text.encode("utf-8"))
    domain = (urlparse(url).hostname or "").casefold()
    if not domain or fingerprint in excluded_fingerprints or domain in excluded_domains:
        return None
    return {
        "completion": text,
        "domain": domain,
        "fineweb_id": source_id,
        "fingerprint": fingerprint,
        "source_config": source_config,
        "source_revision": source_revision,
        "url": url,
        "word_count": word_count,
    }


def select_corpus(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    source = config.get("source") or {}
    selection = config.get("selection") or {}
    corpus_size = int(selection.get("corpus_size", 0))
    dev_count = int(selection.get("dev_count", 0))
    pool_size = int(selection.get("eligible_pool_size", 0))
    scan_limit = int(selection.get("stream_scan_limit", 0))
    if corpus_size < 8 or dev_count < 2 or dev_count >= corpus_size:
        raise PilotCorpusError("selection requires corpus_size >= 8 and 2 <= dev_count < corpus_size")
    if pool_size < corpus_size or scan_limit < pool_size:
        raise PilotCorpusError("selection requires scan_limit >= pool_size >= corpus_size")
    excluded_fingerprints, excluded_domains = _load_exclusions(config)
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
            excluded_fingerprints=excluded_fingerprints,
            excluded_domains=excluded_domains,
        )
        if candidate is not None:
            candidates.setdefault(candidate["fingerprint"], candidate)
    selection_seed = str(selection.get("selection_seed") or "")
    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            _sha256(f"{selection_seed}:{row['fingerprint']}".encode("utf-8")),
            row["fingerprint"],
        ),
    )
    selected: list[dict[str, Any]] = []
    used_domains: set[str] = set()
    for row in ranked:
        if row["domain"] in used_domains:
            continue
        selected.append(row)
        used_domains.add(row["domain"])
        if len(selected) == corpus_size:
            break
    if len(selected) != corpus_size:
        raise PilotCorpusError(
            f"only {len(selected)} distinct-domain records selected after scanning {scanned}"
        )
    split_seed = str(selection.get("split_seed") or "")
    split_ranked = sorted(
        selected,
        key=lambda row: (
            _sha256(f"{split_seed}:{row['fingerprint']}".encode("utf-8")),
            row["fingerprint"],
        ),
    )
    dev_fingerprints = {row["fingerprint"] for row in split_ranked[:dev_count]}
    splits = {"train": [], "dev": []}
    for row in selected:
        emitted = dict(row)
        emitted["split"] = "dev" if row["fingerprint"] in dev_fingerprints else "train"
        splits[emitted["split"]].append(emitted)
    for split in splits:
        splits[split].sort(key=lambda row: row["fingerprint"])
    return splits, {
        "corpus_size": corpus_size,
        "dev_count": len(splits["dev"]),
        "eligible_unique_count": len(candidates),
        "excluded_domain_count": len(excluded_domains),
        "excluded_fingerprint_count": len(excluded_fingerprints),
        "scanned_count": scanned,
        "train_count": len(splits["train"]),
        "unique_domain_count": len(used_domains),
    }


def _stream_source(config: dict[str, Any]):
    source = config.get("source") or {}
    required = ("dataset_id", "dataset_config", "revision", "split")
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise PilotCorpusError("source is missing fields: " + ", ".join(missing))
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
    splits, counts = select_corpus(_stream_source(config) if rows is None else rows, config)
    output = config.get("output") or {}
    train_path = _resolve(str(output.get("train_source_path") or ""))
    dev_path = _resolve(str(output.get("dev_source_path") or ""))
    manifest_path = _resolve(str(output.get("manifest_path") or ""))
    if ROOT in {train_path, dev_path, manifest_path}:
        raise PilotCorpusError("all output paths are required")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train": "".join(_canonical_json(row) + "\n" for row in splits["train"]),
        "dev": "".join(_canonical_json(row) + "\n" for row in splits["dev"]),
    }
    train_path.write_text(payloads["train"], encoding="utf-8")
    dev_path.write_text(payloads["dev"], encoding="utf-8")
    manifest = {
        "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
        "config_path": _display(resolved_config),
        "config_sha256": _sha256(resolved_config.read_bytes()),
        "counts": counts,
        "dev": {
            "domains": [row["domain"] for row in splits["dev"]],
            "fingerprints": [row["fingerprint"] for row in splits["dev"]],
            "path": _display(dev_path),
            "sha256": _sha256(payloads["dev"].encode("utf-8")),
            "split_hash": split_hash(splits["dev"]),
        },
        "policy": config.get("policy") or {},
        "selection": config.get("selection") or {},
        "source": config.get("source") or {},
        "train": {
            "domains": [row["domain"] for row in splits["train"]],
            "fingerprints": [row["fingerprint"] for row in splits["train"]],
            "path": _display(train_path),
            "sha256": _sha256(payloads["train"].encode("utf-8")),
            "split_hash": split_hash(splits["train"]),
        },
    }
    manifest["fingerprints"] = manifest["train"]["fingerprints"] + manifest["dev"]["fingerprints"]
    manifest["domains"] = manifest["train"]["domains"] + manifest["dev"]["domains"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a fixed real-FineWeb M1 pilot corpus")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(materialize(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
