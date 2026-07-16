from __future__ import annotations

import json

import pytest

from data.adherence_validation import AdherenceValidationError, validate_prompt_repair


def _write(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _rows():
    return [{
        "fingerprint": "a" * 64,
        "completion": "Orlando Health MyChart provides secure patient access.",
        "user_prompt": "Convert the supplied human web document into a DFT training brief.",
        "outline": [],
    }]


def test_validation_accepts_only_prompt_change(tmp_path):
    original, repaired = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    before = _rows()
    after = [dict(before[0], user_prompt="Write an overview of Orlando Health MyChart patient access.")]
    _write(original, before)
    _write(repaired, after)
    result = validate_prompt_repair(original, repaired, expected_count=1)
    assert result["only_user_prompt_changed"] is True
    assert result["unique_prompt_count"] == 1


def test_validation_rejects_any_other_field_change(tmp_path):
    original, repaired = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    before = _rows()
    after = [dict(before[0], completion="changed", user_prompt="Write about Orlando Health.")]
    _write(original, before)
    _write(repaired, after)
    with pytest.raises(AdherenceValidationError, match="frozen field"):
        validate_prompt_repair(original, repaired, expected_count=1)
