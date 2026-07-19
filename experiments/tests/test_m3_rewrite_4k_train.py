from __future__ import annotations

import copy

import pytest

from backend.policy import canonical_hash as policy_hash, validate_launch
from experiments.m3.rewrite_4k_train import (
    M3Rewrite4KError,
    _accumulate_component,
    _component_gradient_stats,
    build_config,
    deterministic_schedule,
    validate_config,
)


CORPUS_SHA = "a" * 64


@pytest.mark.parametrize("arm", ["SFT14", "HUMANWRITE14"])
def test_frozen_4k_training_config_and_gateway_policy(arm: str) -> None:
    config = build_config(arm, CORPUS_SHA)
    assert validate_config(config) is config
    payload = {
        "run_id": "dftr-m3-4k-test",
        "config": config,
        "config_hash": policy_hash(config),
        "git_sha": "b" * 40,
        "budget_class": "promo",
        "preregistration": {
            "kind": "prereg",
            "comparison": config["run"]["comparison_id"],
            "status": "open",
        },
        "human_scaleup_approved": True,
    }
    policy = validate_launch(payload)
    assert policy.gpu == "H100"
    assert policy.timeout_seconds == 4 * 60 * 60


def test_4k_schedule_is_exact_and_deterministic() -> None:
    first = deterministic_schedule(4096, 3701)
    second = deterministic_schedule(4096, 3701)
    assert first == second
    assert len(first) == 4096
    assert sorted(first) == list(range(4096))
    assert set(first[:2048]).isdisjoint(first[2048:])


def test_4k_contract_rejects_silent_drift() -> None:
    config = copy.deepcopy(build_config("HUMANWRITE14", CORPUS_SHA))
    config["training"]["learning_rate"] = 3e-5
    with pytest.raises(M3Rewrite4KError, match="frozen contract"):
        validate_config(config)


def test_sft_disables_both_distribution_terms() -> None:
    sft = build_config("SFT14", CORPUS_SHA)
    treatment = build_config("HUMANWRITE14", CORPUS_SHA)
    assert sft["objectives"]["moment_enabled"] is False
    assert sft["objectives"]["witness_enabled"] is False
    assert treatment["objectives"]["moment_enabled"] is True
    assert treatment["objectives"]["witness_enabled"] is True


def test_component_gradient_accumulation_and_cosine_are_exact() -> None:
    torch = pytest.importorskip("torch")
    ce = [torch.zeros(2), torch.zeros(1)]
    moment = [torch.zeros(2), torch.zeros(1)]
    _accumulate_component(ce, [torch.tensor([3.0, 4.0]), torch.tensor([0.0])])
    _accumulate_component(moment, [torch.tensor([4.0, -3.0]), torch.tensor([0.0])])
    ce_norm, moment_norm, cosine = _component_gradient_stats(ce, moment)
    assert ce_norm == pytest.approx(5.0)
    assert moment_norm == pytest.approx(5.0)
    assert cosine == pytest.approx(0.0)
