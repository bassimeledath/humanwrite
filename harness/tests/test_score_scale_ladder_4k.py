from __future__ import annotations

import json
from pathlib import Path

import pytest

from score_scale_ladder_4k import binomial_lower_tail, hard_validity, validate_contract


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_scale_ladder_contract_is_structurally_valid():
    contract = json.loads(
        (ROOT / "configs/m2/m2_scale_ladder_4k_scoring_contract_v1.json").read_text()
    )
    validate_contract(contract)
    assert contract["metric_values_observed_before_binding"] is False
    assert contract["decision"]["no_hyperparameter_selection_from_this_panel"] is True


def test_binomial_lower_tail_is_exact_and_validated():
    assert binomial_lower_tail(0, 2) == 0.25
    assert binomial_lower_tail(1, 2) == 0.75
    with pytest.raises(ValueError):
        binomial_lower_tail(3, 2)


def test_hard_validity_enforces_all_frozen_bounds():
    rule = {
        "empty_rate_max": 0.0,
        "replacement_character_rate_max": 0.0,
        "unexpected_non_latin_rate_max": 0.15,
        "unique_fraction_min": 0.80,
    }
    assert hard_validity(["alpha", "beta"], rule)["pass"] is True
    assert hard_validity(["", "beta"], rule)["pass"] is False
    assert hard_validity(["ไทย", "beta"], rule)["pass"] is False
