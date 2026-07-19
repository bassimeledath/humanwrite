from __future__ import annotations

import pytest

from experiments.m3.materialize_baseline_verify_config import config
from infra.backend.policy import canonical_hash, validate_launch


def test_materialized_baseline_verifier_passes_gateway_policy() -> None:
    value = config("a" * 64)
    payload = {
        "config": value,
        "config_hash": canonical_hash(value),
        "budget_class": "promo",
        "preregistration": {
            "kind": "prereg",
            "status": "open",
            "comparison": "M3-rewriting-14b-4096-scientific-screen-v1",
        },
    }
    policy = validate_launch(payload)
    assert policy.task_kind == "rewrite_synthesis"
    assert policy.api_reserved_cost_usd == 6.0


def test_materializer_rejects_non_sha() -> None:
    with pytest.raises(ValueError, match="SHA"):
        config("not-a-sha")
