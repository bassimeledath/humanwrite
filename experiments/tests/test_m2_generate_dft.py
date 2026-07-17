from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from experiments.m2.generate_dft import (
    GenerationConfigError,
    canonical_hash,
    decoding_policy_payload,
    generation_contract_payload,
    validate_generation_config,
)
from infra.backend.policy import PolicyError, validate_launch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/m2/m2_dft_4b_a0_generation_seed101_v1.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def payload(value: dict) -> dict:
    return {
        "config": value,
        "config_hash": canonical_hash(value),
        "git_sha": "a" * 40,
        "budget_class": "screen",
        "preregistration": {
            "kind": "prereg",
            "comparison": value["run"]["comparison_id"],
            "status": "open",
        },
    }


def test_frozen_generation_config_validates() -> None:
    value = config()
    assert validate_generation_config(value) is value
    assert value["workflow"]["generation_contract_sha256"] == canonical_hash(
        generation_contract_payload(value)
    )
    assert value["workflow"]["decoding_policy_sha256"] == canonical_hash(
        decoding_policy_payload(value)
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["sampling"].update({"new_tokens": 63}),
        lambda value: value["sampling"].update({"distribution": "top_p"}),
        lambda value: value["sampling"].update({"sampling_seed": 102}),
        lambda value: value["checkpoint"].update({"arm": "A64"}),
        lambda value: value["data"].update({"prompt_briefs_path": "relative.jsonl"}),
    ],
)
def test_generation_config_fails_closed(mutate) -> None:
    value = config()
    mutate(value)
    with pytest.raises(GenerationConfigError):
        validate_generation_config(value)


def test_gateway_accepts_only_modal_volume_generation(monkeypatch) -> None:
    value = config()
    policy = validate_launch(payload(value), backend="modal")
    assert policy.task_kind == "experiment"
    changed = copy.deepcopy(value)
    changed["checkpoint"]["path"] = "/tmp/checkpoint"
    changed_payload = payload(changed)
    with pytest.raises(PolicyError, match="generate_dft"):
        validate_launch(changed_payload, backend="modal")


def test_gateway_rejects_local_generation() -> None:
    with pytest.raises(PolicyError, match="generate_dft"):
        validate_launch(payload(config()), backend="local")
