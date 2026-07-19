from __future__ import annotations

from backend.policy import canonical_hash, validate_launch
from experiments.m3.materialize_eval_clean_config import INPUT_URI, OUTPUT_URI, config


def test_fresh_eval_cleaner_config_is_sha_bound_and_policy_valid() -> None:
    value = config("a" * 64)
    assert value["data"]["input_uri"] == INPUT_URI
    assert value["data"]["output_uri"] == OUTPUT_URI
    assert value["data"]["target_records"] == 640
    payload = {
        "config": value,
        "config_hash": canonical_hash(value),
        "budget_class": "promo",
        "preregistration": {
            "kind": "prereg",
            "status": "open",
            "comparison": value["run"]["comparison_id"],
        },
    }
    policy = validate_launch(payload)
    assert policy.task_kind == "document_cleaning"
    assert policy.api_reserved_cost_usd == 3.0
