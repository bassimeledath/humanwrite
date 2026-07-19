from __future__ import annotations

from backend.policy import canonical_hash, validate_launch
from experiments.m3.materialize_eval_rewrite_config import config


def test_eval_rewrite_config_is_exact_and_policy_valid() -> None:
    value = config("a" * 64)
    assert value["data"]["max_records"] == 640
    assert value["data"]["target_records"] == 224
    policy = validate_launch(
        {
            "config": value,
            "config_hash": canonical_hash(value),
            "budget_class": "promo",
            "preregistration": {
                "kind": "prereg",
                "status": "open",
                "comparison": value["run"]["comparison_id"],
            },
        }
    )
    assert policy.api_reserved_cost_usd == 6.0
