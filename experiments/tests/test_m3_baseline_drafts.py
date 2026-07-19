from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from experiments.m3.baseline_drafts import M3BaselineDraftError, baseline_prompt, validate_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/m3/m3_baseline_drafts_14b_4096_v1.yaml"


def test_frozen_baseline_draft_config_validates() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert validate_config(config) == config


def test_baseline_draft_contract_hash_rejects_tampering() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tampered = deepcopy(config)
    tampered["generation"]["temperature"] = 1.0
    with pytest.raises(M3BaselineDraftError, match="generation contract"):
        validate_config(tampered)


def test_baseline_prompt_never_contains_hidden_rewrite_language() -> None:
    row = {
        "user_prompt": "Write a factual announcement.",
        "use_case": "announcement",
        "style_kind": "news",
        "style": "concise",
        "detail_mode": "grounded",
        "target_length": 96,
        "target_length_unit": "tokens",
        "em_dashes_allowed": False,
        "outline": ["State the event", "Give the date 2026"],
    }
    rendered = baseline_prompt(row)
    assert "Give the date 2026" in rendered
    assert "human target" not in rendered.casefold()
    assert "later editing pass" in rendered
