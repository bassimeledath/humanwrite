from __future__ import annotations

import copy

import pytest

from backend.policy import canonical_hash, validate_launch
from experiments.m3.rewrite_generate_14b import (
    M3RewriteGenerationError,
    build_config,
    prompt_seed,
    validate_config,
)


PANEL_SHA = "a" * 64


def test_base_and_trained_generation_contracts_are_exact() -> None:
    base = build_config("BASE", PANEL_SHA)
    assert validate_config(base) is base
    trained = build_config(
        "HUMANWRITE14",
        PANEL_SHA,
        training_manifest_path="/checkpoints/runs/dftr-treatment/run_manifest.json",
        training_manifest_sha256="b" * 64,
    )
    assert validate_config(trained) is trained
    assert trained["generation"] == base["generation"]


def test_generation_contract_rejects_drift() -> None:
    value = copy.deepcopy(build_config("BASE", PANEL_SHA))
    value["generation"]["temperature"] = 0.8
    with pytest.raises(M3RewriteGenerationError, match="frozen contract"):
        validate_config(value)


def test_prompt_seed_is_paired_and_stable() -> None:
    fingerprint = "c" * 64
    assert prompt_seed(4101, fingerprint) == prompt_seed(4101, fingerprint)
    assert 0 <= prompt_seed(4101, fingerprint) < 2**63
    assert prompt_seed(4102, fingerprint) != prompt_seed(4101, fingerprint)


@pytest.mark.parametrize("arm", ["BASE", "SFT14", "HUMANWRITE14"])
def test_generation_gateway_policy_accepts_only_frozen_contract(arm: str) -> None:
    kwargs = {}
    if arm != "BASE":
        kwargs = {
            "training_manifest_path": f"/checkpoints/runs/{arm}/run_manifest.json",
            "training_manifest_sha256": "d" * 64,
        }
    value = build_config(arm, PANEL_SHA, **kwargs)
    policy = validate_launch(
        {
            "config": value,
            "config_hash": canonical_hash(value),
            "budget_class": "screen",
            "human_scaleup_approved": True,
            "preregistration": {
                "kind": "prereg",
                "status": "open",
                "comparison": value["run"]["comparison_id"],
            },
        }
    )
    assert policy.gpu == "H100"
