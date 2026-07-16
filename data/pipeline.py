from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_INPUT = ROOT / "fixtures" / "fineweb_fixture.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "m0"
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
NOISE_PREFIXES = (
    "home",
    "about",
    "contact",
    "subscribe",
    "sign in",
    "sign up",
    "cookie",
    "privacy",
    "terms",
    "skip to content",
    "menu",
    "related articles",
)
EMPTY_OUTLINE_RATIO = (1, 4)


@dataclass(frozen=True)
class RawDocument:
    fineweb_id: str
    url: str
    domain: str
    title: str
    text: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_raw_documents(path: Path) -> list[RawDocument]:
    return [
        RawDocument(
            fineweb_id=str(row["fineweb_id"]),
            url=str(row["url"]),
            domain=str(row["domain"]),
            title=str(row["title"]),
            text=str(row["text"]),
        )
        for row in _read_jsonl(path)
    ]


def normalize_whitespace(text: str) -> str:
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
    collapsed = re.sub(r"[ \t]+", " ", collapsed).strip()
    return collapsed


def clean_document(document: RawDocument) -> dict:
    cleaned = normalize_whitespace(document.text)
    fingerprint = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return {
        "fineweb_id": document.fineweb_id,
        "url": document.url,
        "domain": document.domain,
        "title": document.title,
        "original_text": document.text,
        "cleaned_text": cleaned,
        "fingerprint": fingerprint,
    }


def dedupe_records(records: Iterable[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for record in sorted(records, key=lambda item: (item["fingerprint"], item["fineweb_id"])):
        unique.setdefault(record["fingerprint"], record)
    return list(unique.values())


def _ranked_fingerprints(records: list[dict], label: str) -> list[str]:
    ranked = []
    for fingerprint in sorted(record["fingerprint"] for record in records):
        digest = hashlib.sha256(f"{label}:{fingerprint}".encode("utf-8")).hexdigest()
        ranked.append((digest, fingerprint))
    ranked.sort()
    return [fingerprint for _, fingerprint in ranked]


def assign_splits(records: list[dict]) -> dict[str, str]:
    if len(records) < 2:
        raise ValueError("at least two cleaned records are required")
    ranked = _ranked_fingerprints(records, "split")
    dev_count = max(1, len(records) // 4)
    dev_fingerprints = set(ranked[:dev_count])
    result = {}
    for record in records:
        result[record["fingerprint"]] = "dev" if record["fingerprint"] in dev_fingerprints else "train"
    return result


def _empty_outline_fingerprints(records: list[dict]) -> set[str]:
    ranked = _ranked_fingerprints(records, "empty-outline")
    numerator, denominator = EMPTY_OUTLINE_RATIO
    target = int(math.floor(len(records) * numerator / denominator))
    return set(ranked[:target])


def split_hash(records: Iterable[dict]) -> str:
    values = sorted(str(record["fingerprint"]) for record in records)
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in PARAGRAPH_RE.split(text) if part.strip()]


def _trim_fact(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def infer_use_case(record: dict) -> str:
    title = record["title"].casefold()
    domain = record["domain"].casefold()
    if "news" in domain or title.startswith("market") or "briefing" in title:
        return "news"
    if "guide" in title or "how to" in title:
        return "guide"
    if "review" in title:
        return "review"
    if "essay" in title or "opinion" in title:
        return "essay"
    if "faq" in title:
        return "faq"
    return "company_blog"


def infer_style(record: dict) -> tuple[str, str]:
    text = record["cleaned_text"]
    length = len(WORD_RE.findall(text))
    if "?" in text:
        return "instructive", "direct, pragmatic"
    if length > 220:
        return "professional", "measured, explanatory"
    if any(token in text.casefold() for token in ("reported", "according to", "officials")):
        return "reported", "neutral, sourced"
    return "professional", "clear, restrained"


def infer_detail_mode(record: dict) -> str:
    digest = hashlib.sha256(record["fingerprint"].encode("utf-8")).digest()[0]
    return "strict" if digest % 2 == 0 else "creative"


def build_outline(record: dict, empty_outline: bool) -> list[dict]:
    if empty_outline:
        return []
    paragraphs = _paragraphs(record["cleaned_text"])
    if not paragraphs:
        return []
    sections = []
    for index, paragraph in enumerate(paragraphs[:3], start=1):
        sentences = _sentences(paragraph)
        facts = [_trim_fact(sentence) for sentence in sentences[:2]]
        if not facts:
            continue
        section_title = record["title"] if index == 1 else f"{record['title']} ({index})"
        sections.append(
            {
                "section": section_title,
                "supported_facts": facts,
                "quotations": [facts[0]],
            }
        )
    return sections


def build_user_prompt(record: dict, use_case: str) -> str:
    if use_case == "news":
        return f"Write a concise reported update about {record['title']}."
    if use_case == "guide":
        return f"Explain {record['title']} in a practical step-by-step article."
    if use_case == "faq":
        return f"Answer common questions about {record['title']}."
    if use_case == "essay":
        return f"Write a short essay about {record['title']}."
    if use_case == "review":
        return f"Write a balanced review covering {record['title']}."
    return f"Draft a polished web article about {record['title']}."


def build_brief_record(record: dict, split_name: str, empty_outline: bool) -> dict:
    use_case = infer_use_case(record)
    style_kind, style = infer_style(record)
    completion = record["cleaned_text"]
    return {
        "fineweb_id": record["fineweb_id"],
        "domain": record["domain"],
        "fingerprint": record["fingerprint"],
        "split": split_name,
        "generation_mode": "generate",
        "use_case": use_case,
        "style_kind": style_kind,
        "style": style,
        "detail_mode": infer_detail_mode(record),
        "target_length": len(WORD_RE.findall(completion)),
        "em_dashes_allowed": "—" in completion,
        "user_prompt": build_user_prompt(record, use_case),
        "outline": build_outline(record, empty_outline),
        "completion": completion,
    }


def build_dataset(path: Path = DEFAULT_INPUT) -> dict[str, object]:
    cleaned = dedupe_records(clean_document(document) for document in load_raw_documents(path))
    split_map = assign_splits(cleaned)
    empty_outline = _empty_outline_fingerprints(cleaned)
    try:
        fixture_ref = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        fixture_ref = str(path)
    briefs = [
        build_brief_record(record, split_map[record["fingerprint"]], record["fingerprint"] in empty_outline)
        for record in sorted(cleaned, key=lambda item: item["fingerprint"])
    ]
    by_split = {
        "train": [record for record in briefs if record["split"] == "train"],
        "dev": [record for record in briefs if record["split"] == "dev"],
    }
    manifests = {}
    for split_name, rows in by_split.items():
        manifests[split_name] = {
            "split": split_name,
            "count": len(rows),
            "hash": split_hash(rows),
            "fingerprints": [row["fingerprint"] for row in rows],
            "records": [
                {
                    "fineweb_id": row["fineweb_id"],
                    "domain": row["domain"],
                    "fingerprint": row["fingerprint"],
                }
                for row in rows
            ],
        }
    return {
        "source": {
            "fixture": fixture_ref,
            "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "record_count": len(cleaned),
            "deduped_count": len(cleaned),
        },
        "cleaned": cleaned,
        "briefs": briefs,
        "train": by_split["train"],
        "dev": by_split["dev"],
        "manifests": manifests,
        "split_hashes": {
            "train": manifests["train"]["hash"],
            "dev": manifests["dev"]["hash"],
        },
        "hidden_test_boundary": {
            "owned_by": "sealed_evaluator",
            "materialized_locally": False,
            "policy": "Do not emit hidden test completions into the agent-readable tree.",
            "local_fields": ["comparison_id", "split_seed", "train_hash", "dev_hash", "excluded_fields"],
            "excluded_fields": ["completion", "cleaned_text", "original_text", "outline"],
            "split_seed": "dftr-m0-hidden-test-boundary-v1",
            "train_hash": manifests["train"]["hash"],
            "dev_hash": manifests["dev"]["hash"],
        },
        "summary": {
            "empty_outline_count": sum(1 for row in briefs if not row["outline"]),
            "empty_outline_ratio": sum(1 for row in briefs if not row["outline"]) / max(1, len(briefs)),
            "train_count": len(by_split["train"]),
            "dev_count": len(by_split["dev"]),
        },
    }


def write_dataset(dataset: dict[str, object], output_dir: Path = DEFAULT_OUTPUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = dataset["cleaned"]
    briefs = dataset["briefs"]
    train = dataset["train"]
    dev = dataset["dev"]
    manifests = dataset["manifests"]
    files = {
        "cleaned_records.jsonl": cleaned,
        "brief_records_all.jsonl": briefs,
        "train_briefs.jsonl": train,
        "dev_briefs.jsonl": dev,
    }
    for name, rows in files.items():
        payload = "\n".join(_canonical_json(row) for row in rows) + "\n"
        (output_dir / name).write_text(payload, encoding="utf-8")
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifests["train"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dev_manifest.json").write_text(
        json.dumps(manifests["dev"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("split_hashes", "hidden_test_boundary", "summary", "source"):
        (output_dir / f"{name}.json").write_text(
            json.dumps(dataset[name], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the offline M0 data artifacts.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = build_dataset(Path(args.input).resolve())
    write_dataset(dataset, Path(args.output_dir).resolve())
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "train_hash": dataset["split_hashes"]["train"],
                "dev_hash": dataset["split_hashes"]["dev"],
                "empty_outline_count": dataset["summary"]["empty_outline_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
