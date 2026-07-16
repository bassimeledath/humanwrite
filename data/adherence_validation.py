"""Validate that adherence recovery changed only defective user prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from infra.backend.brief_contract import record_id, validate_repaired_user_prompt


class AdherenceValidationError(ValueError):
    pass


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_prompt_repair(
    original_path: str | Path,
    repaired_path: str | Path,
    *,
    expected_count: int,
) -> dict[str, Any]:
    original_path, repaired_path = Path(original_path), Path(repaired_path)
    original, repaired = _load(original_path), _load(repaired_path)
    if len(original) != expected_count or len(repaired) != expected_count:
        raise AdherenceValidationError("prompt repair record count mismatch")
    original_index = {record_id(row): row for row in original}
    repaired_index = {record_id(row): row for row in repaired}
    if len(original_index) != expected_count or set(original_index) != set(repaired_index):
        raise AdherenceValidationError("prompt repair record identities changed")
    prompts: list[str] = []
    for source_id, before in original_index.items():
        after = repaired_index[source_id]
        if set(before) != set(after):
            raise AdherenceValidationError("prompt repair field set changed")
        for field, value in before.items():
            if field != "user_prompt" and after[field] != value:
                raise AdherenceValidationError(f"prompt repair changed frozen field: {field}")
        prompt = validate_repaired_user_prompt(
            after["user_prompt"], source_text=str(before["completion"])
        )
        if prompt == str(before.get("user_prompt") or ""):
            raise AdherenceValidationError("prompt repair left a defective prompt unchanged")
        prompts.append(prompt)
    return {
        "artifact_schema": "dftr.adherence_prompt_repair_validation.v1",
        "count": expected_count,
        "original_sha256": _sha256(original_path),
        "repaired_sha256": _sha256(repaired_path),
        "unique_prompt_count": len(set(prompts)),
        "min_prompt_characters": min(map(len, prompts)),
        "max_prompt_characters": max(map(len, prompts)),
        "only_user_prompt_changed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--repaired", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(validate_prompt_repair(
        args.original, args.repaired, expected_count=args.expected_count
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
