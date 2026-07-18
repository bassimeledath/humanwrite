"""Freeze exact token lengths for the 128 prospective evaluation prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from data.lower_variance_briefs import (
    deterministic_empty_outline_ids,
    validate_assembled_brief,
)


MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


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


def normalize(
    source_path: Path,
    brief_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("evaluation normalization never overwrites artifacts")
    sources, briefs = _rows(source_path), _rows(brief_path)
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    brief_by_id = {str(row["fingerprint"]): row for row in briefs}
    if (
        len(sources) != 128
        or len(briefs) != 128
        or len(source_by_id) != 128
        or set(source_by_id) != set(brief_by_id)
    ):
        raise ValueError("evaluation normalization requires 128 identity-aligned rows")
    empty_ids = deterministic_empty_outline_ids(sources)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True
    )
    old_lengths, new_lengths = [], []
    normalized: list[dict[str, Any]] = []
    for source in sources:
        fingerprint = str(source["fingerprint"])
        row = dict(brief_by_id[fingerprint])
        old_lengths.append(int(row["target_length"]))
        exact_length = len(
            tokenizer.encode(str(row["completion"]), add_special_tokens=False)
        )
        if not 1 <= exact_length <= 4096:
            raise ValueError("exact evaluation token length is outside the contract")
        row["target_length"] = exact_length
        validate_assembled_brief(
            row,
            source=source,
            force_empty_outline=fingerprint in empty_ids,
        )
        new_lengths.append(exact_length)
        normalized.append(row)
    normalized.sort(key=lambda row: str(row["fingerprint"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in normalized
        ),
        encoding="utf-8",
    )
    ordered = sorted(new_lengths)
    manifest = {
        "artifact_schema": "dftr.eval_prompt_exact_token_length_normalization.v1",
        "status": "completed",
        "candidate_outputs_opened": False,
        "rows": len(normalized),
        "source_sha256": _sha(source_path),
        "input_briefs_sha256": _sha(brief_path),
        "output_briefs_sha256": _sha(output_path),
        "changed_rows": sum(a != b for a, b in zip(old_lengths, new_lengths)),
        "tokenizer": MODEL,
        "tokenizer_revision": REVISION,
        "new_min": ordered[0],
        "new_median": ordered[len(ordered) // 2],
        "new_max": ordered[-1],
        "empty_outlines": len(empty_ids),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            normalize(
                args.sources.resolve(),
                args.briefs.resolve(),
                args.output.resolve(),
                args.manifest.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
