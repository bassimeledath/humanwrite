from __future__ import annotations

import json

import pytest

from backend.policy import (
    PolicyError,
    accrued_gpu_spend,
    append_event,
    authorized,
    budget_snapshot,
    canonical_hash,
    has_capacity,
    run_snapshot,
    validate_launch,
)


def payload(**compute_overrides):
    config = {
        "run": {
            "comparison_id": "M2-A-vs-SFT",
            "budget_class": "smoke",
            "command": ["python", "-m", "experiments.runner"],
        },
        "model": {"base": "Qwen/Qwen3-1.7B"},
        "compute": {"gpu": "L4", "gpus": 1, "timeout_min": 10, **compute_overrides},
    }
    return {
        "run_id": "dftr-test",
        "config": config,
        "config_hash": canonical_hash(config),
        "git_sha": "a" * 40,
        "budget_class": "smoke",
        "preregistration": {
            "kind": "prereg",
            "comparison": "M2-A-vs-SFT",
            "status": "open",
        },
        "human_scaleup_approved": False,
    }


def test_valid_launch_reserves_conservative_cost():
    policy = validate_launch(payload())
    assert policy.gpu == "L4"
    assert policy.timeout_seconds == 600
    assert 0 < policy.worst_case_cost_usd < 1


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda p: p["config"]["compute"].update(gpus=2), "single-GPU"),
        (lambda p: p["config"]["compute"].update(timeout_min=21), "timeout"),
        (lambda p: p["config"]["run"].update(command=["bash", "-lc", "env"]), "allowlist"),
        (lambda p: p["preregistration"].update(status="closed"), "preregistration"),
    ],
)
def test_policy_rejects_invalid_requests(mutation, error):
    value = payload()
    mutation(value)
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match=error):
        validate_launch(value)


def test_hash_tampering_is_rejected():
    value = payload()
    value["config"]["compute"]["timeout_min"] = 11
    with pytest.raises(PolicyError, match="hash mismatch"):
        validate_launch(value)


def test_14b_requires_human_flag():
    value = payload()
    value["config"]["model"]["base"] = "Qwen/Qwen3-14B"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="human approval"):
        validate_launch(value)
    value["human_scaleup_approved"] = True
    assert validate_launch(value).comparison_id == "M2-A-vs-SFT"


def test_brief_synthesis_reserves_api_not_gpu():
    value = payload()
    value["config"]["run"]["task_kind"] = "brief_synthesis"
    value["config"]["data"] = {
        "input_uri": "modal-volume://humanwrite-checkpoints/data/clean.jsonl",
        "output_uri": "modal-volume://humanwrite-checkpoints/data/briefs.jsonl",
        "input_sha256": "a" * 64,
        "max_records": 320,
    }
    value["config"]["api"] = {"max_cost_usd": 25.0, "model": "openai/gpt-5-mini"}
    value["config_hash"] = canonical_hash(value["config"])
    policy = validate_launch(value)
    assert policy.task_kind == "brief_synthesis"
    assert policy.worst_case_cost_usd == 0
    assert policy.api_reserved_cost_usd == 25.0


def test_brief_synthesis_rejects_unsafe_volume_uri():
    value = payload()
    value["config"]["run"]["task_kind"] = "brief_synthesis"
    value["config"]["data"] = {
        "input_uri": "file:///tmp/leak.jsonl",
        "output_uri": "modal-volume://humanwrite-checkpoints/data/briefs.jsonl",
        "input_sha256": "a" * 64,
        "max_records": 320,
    }
    value["config"]["api"] = {"max_cost_usd": 25.0, "model": "openai/gpt-5-mini"}
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="checkpoint volume"):
        validate_launch(value)


def test_brief_synthesis_requires_hash_count_and_model_binding():
    value = payload()
    value["config"]["run"]["task_kind"] = "brief_synthesis"
    value["config"]["data"] = {
        "input_uri": "modal-volume://humanwrite-checkpoints/data/clean.jsonl",
        "output_uri": "modal-volume://humanwrite-checkpoints/data/briefs.jsonl",
        "input_sha256": "bad",
        "max_records": 320,
    }
    value["config"]["api"] = {"max_cost_usd": 5.0, "model": "openai/gpt-5-mini"}
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="input_sha256"):
        validate_launch(value)

    value["config"]["data"]["input_sha256"] = "a" * 64
    value["config"]["api"]["model"] = ""
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="api.model"):
        validate_launch(value)


def test_quote_free_brief_recovery_is_tightly_bounded():
    value = payload()
    value["config"]["run"]["task_kind"] = "brief_synthesis"
    value["config"]["data"] = {
        "input_uri": "modal-volume://humanwrite-checkpoints/data/clean.jsonl",
        "output_uri": "modal-volume://humanwrite-checkpoints/data/briefs.jsonl",
        "input_sha256": "a" * 64,
        "max_records": 256,
    }
    value["config"]["api"] = {
        "max_cost_usd": 0.25,
        "model": "openai/gpt-5-mini",
        "force_empty_quotations": True,
    }
    value["config"]["recovery"] = {"max_missing_records": 3}
    value["config_hash"] = canonical_hash(value["config"])
    assert validate_launch(value).api_reserved_cost_usd == 0.25
    value["config"]["recovery"]["max_missing_records"] = 17
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="1..16"):
        validate_launch(value)


def test_prompt_repair_is_bounded_and_cannot_overwrite_or_mix_recovery_modes():
    value = payload()
    value["config"]["run"]["task_kind"] = "brief_synthesis"
    value["config"]["data"] = {
        "input_uri": "modal-volume://humanwrite-checkpoints/data/original.jsonl",
        "output_uri": "modal-volume://humanwrite-checkpoints/data/repaired.jsonl",
        "input_sha256": "a" * 64,
        "max_records": 320,
    }
    value["config"]["api"] = {
        "max_cost_usd": 2.0,
        "model": "openai/gpt-5-mini",
        "prompt_repair_only": True,
    }
    value["config_hash"] = canonical_hash(value["config"])
    assert validate_launch(value).api_reserved_cost_usd == 2.0

    value["config"]["data"]["output_uri"] = value["config"]["data"]["input_uri"]
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="distinct"):
        validate_launch(value)

    value["config"]["data"]["output_uri"] = (
        "modal-volume://humanwrite-checkpoints/data/repaired.jsonl"
    )
    value["config"]["api"]["force_empty_quotations"] = True
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="cannot also"):
        validate_launch(value)


def test_source_materialization_requires_pinned_source_and_volume_outputs():
    value = payload()
    value["config"]["run"]["task_kind"] = "source_materialization"
    value["config"]["source"] = {
        "dataset_id": "HuggingFaceFW/fineweb",
        "dataset_config": "CC-MAIN-2024-10",
        "revision": "a" * 40,
        "split": "train",
        "files": ["data/CC-MAIN-2024-10/000_00000.parquet"],
    }
    value["config"]["selection"] = {"corpus_size": 320}
    value["config"]["data"] = {
        "train_output_uri": "modal-volume://humanwrite-checkpoints/data/pilot/train.jsonl",
        "dev_output_uri": "modal-volume://humanwrite-checkpoints/data/pilot/dev.jsonl",
        "manifest_output_uri": "modal-volume://humanwrite-checkpoints/data/pilot/manifest.json",
    }
    value["config_hash"] = canonical_hash(value["config"])
    policy = validate_launch(value)
    assert policy.task_kind == "source_materialization"
    assert policy.gpu == "CPU"
    assert policy.worst_case_cost_usd == 0
    assert policy.api_reserved_cost_usd == 0

    value["config"]["data"]["manifest_output_uri"] = "file:///tmp/manifest.json"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="checkpoint volume"):
        validate_launch(value)


def test_append_only_state_and_budget(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, {
        "kind": "run", "run_id": "r1", "status": "running",
        "reserved_cost_usd": 12.0, "billing_month": "2026-07", "ts": 1_784_000_000,
    })
    append_event(path, {
        "kind": "api_cost", "cost_usd": 4.5, "ts": 1_784_000_001,
    })
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert run_snapshot(events, "r1")["status"] == "running"
    budget = budget_snapshot(events, month="2026-07")
    assert budget["gpu_remaining_usd"] == 28.0
    assert budget["api_remaining_usd"] == 95.5
    assert has_capacity(events, 1.0) is True


def test_bearer_auth_uses_exact_token():
    assert authorized("Bearer correct", "correct")
    assert not authorized("Bearer wrong", "correct")
    assert not authorized(None, "correct")


def test_accrued_spend_does_not_treat_full_reservation_as_spent():
    events = [{
        "kind": "run", "run_id": "r1", "status": "running",
        "reserved_cost_usd": 100.0, "timeout_seconds": 1000,
        "started_at": 100.0, "ts": 100.0,
    }]
    assert accrued_gpu_spend(events, now=200.0) == 10.0
    accrued_gpu_spend,
