"""Fixed-code FineWeb source selection for the privileged Modal boundary."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable
from urllib.parse import urlparse


WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
NOISE_PREFIXES = (
    "home", "about", "contact", "subscribe", "sign in", "sign up", "cookie",
    "privacy", "terms", "skip to content", "menu", "related articles",
)


class SourceMaterializationError(ValueError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        lowered = line.casefold()
        if any(lowered.startswith(prefix) for prefix in NOISE_PREFIXES):
            continue
        if line.isupper() and len(line.split()) <= 6:
            continue
        lines.append(line)
    collapsed = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return re.sub(r"[ \t]+", " ", collapsed).strip()


def _non_latin_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for character in letters if "LATIN" in unicodedata.name(character, ""))
    return (len(letters) - latin) / len(letters)


def _split_hash(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(str(row["fingerprint"]) for row in rows)).encode("utf-8")
    return _sha256(payload)


def _candidate(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    source = config["source"]
    selection = config["selection"]
    exclusions = config.get("exclusions") or {}
    source_id = str(row.get("id") or row.get("fineweb_id") or row.get("url") or "").strip()
    url = str(row.get("url") or "").strip()
    raw_text = row.get("text")
    if not source_id or not url or not isinstance(raw_text, str):
        return None
    text = _normalize(raw_text)
    word_count = len(WORD_RE.findall(text))
    if not int(selection["min_word_count"]) <= word_count <= int(selection["max_word_count"]):
        return None
    if _non_latin_ratio(text) > float(selection["max_non_latin_letter_ratio"]):
        return None
    fingerprint = _sha256(text.encode("utf-8"))
    domain = (urlparse(url).hostname or "").casefold()
    if not domain:
        return None
    if fingerprint in {str(value) for value in exclusions.get("fingerprints") or []}:
        return None
    if domain in {str(value).casefold() for value in exclusions.get("domains") or []}:
        return None
    return {
        "completion": text,
        "domain": domain,
        "fineweb_id": source_id,
        "fingerprint": fingerprint,
        "source_config": str(source["dataset_config"]),
        "source_revision": str(source["revision"]),
        "url": url,
        "word_count": word_count,
    }


def materialize_rows(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    selection = config["selection"]
    corpus_size = int(selection["corpus_size"])
    dev_count = int(selection["dev_count"])
    pool_size = int(selection["eligible_pool_size"])
    scan_limit = int(selection["stream_scan_limit"])
    if corpus_size < 8 or dev_count < 2 or dev_count >= corpus_size:
        raise SourceMaterializationError("invalid corpus/dev cardinality")
    if scan_limit < pool_size or pool_size < corpus_size:
        raise SourceMaterializationError("requires scan_limit >= pool_size >= corpus_size")
    candidates: dict[str, dict[str, Any]] = {}
    scanned = 0
    for row in rows:
        if scanned >= scan_limit or len(candidates) >= pool_size:
            break
        scanned += 1
        candidate = _candidate(row, config)
        if candidate is not None:
            candidates.setdefault(candidate["fingerprint"], candidate)
    selection_seed = str(selection["selection_seed"])
    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            _sha256(f"{selection_seed}:{row['fingerprint']}".encode("utf-8")),
            row["fingerprint"],
        ),
    )
    selected: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    max_records_per_domain = int(selection.get("max_records_per_domain", 1))
    if max_records_per_domain < 1:
        raise SourceMaterializationError("max_records_per_domain must be positive")
    for row in ranked:
        if domain_counts.get(row["domain"], 0) >= max_records_per_domain:
            continue
        selected.append(row)
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
        if len(selected) == corpus_size:
            break
    if len(selected) != corpus_size:
        raise SourceMaterializationError(
            f"only {len(selected)} distinct-domain records selected after {scanned} rows"
        )
    split_seed = str(selection["split_seed"])
    split_ranked = sorted(
        selected,
        key=lambda row: (
            _sha256(f"{split_seed}:{row['fingerprint']}".encode("utf-8")),
            row["fingerprint"],
        ),
    )
    dev_fingerprints = {row["fingerprint"] for row in split_ranked[:dev_count]}
    split_rows = {"train": [], "dev": []}
    for row in selected:
        emitted = dict(row)
        emitted["split"] = "dev" if row["fingerprint"] in dev_fingerprints else "train"
        split_rows[emitted["split"]].append(emitted)
    for split in split_rows:
        split_rows[split].sort(key=lambda row: row["fingerprint"])
    payloads = {
        split: "".join(_canonical_json(row) + "\n" for row in split_rows[split])
        for split in ("train", "dev")
    }
    manifest = {
        "artifact_schema": "dftr.realdata_pilot_source.manifest.v1",
        "counts": {
            "corpus_size": corpus_size,
            "dev_count": len(split_rows["dev"]),
            "eligible_unique_count": len(candidates),
            "scanned_count": scanned,
            "train_count": len(split_rows["train"]),
            "unique_domain_count": len(domain_counts),
        },
        "policy": config.get("policy") or {},
        "selection": selection,
        "source": config["source"],
    }
    for split in ("train", "dev"):
        manifest[split] = {
            "domains": [row["domain"] for row in split_rows[split]],
            "fingerprints": [row["fingerprint"] for row in split_rows[split]],
            "sha256": _sha256(payloads[split].encode("utf-8")),
            "split_hash": _split_hash(split_rows[split]),
        }
    manifest["fingerprints"] = manifest["train"]["fingerprints"] + manifest["dev"]["fingerprints"]
    manifest["domains"] = manifest["train"]["domains"] + manifest["dev"]["domains"]
    return payloads, manifest
