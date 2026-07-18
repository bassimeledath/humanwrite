from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "score_v3_candidates.py"
SPEC = importlib.util.spec_from_file_location("score_v3_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_text_safety_metrics_are_document_level_and_length_normalized() -> None:
    assert module._unexpected_non_latin("Plain English prose.") == 0.0
    assert module._unexpected_non_latin("English with 한국어 corruption") == 1.0
    assert module._repeated_bigram_rate("one two one two") == 1 / 3
    assert module._repeated_bigram_rate("one two three four") == 0.0


def test_hard_validity_rejects_empty_replacement_and_duplicate_panels() -> None:
    valid = module._hard_validity([f"Distinct Latin text {index}" for index in range(10)])
    assert valid["pass"] is True
    assert module._hard_validity(["valid", "�"])["pass"] is False
    assert module._hard_validity(["", "valid"])["pass"] is False
    assert module._hard_validity(["same"] * 10)["pass"] is False
