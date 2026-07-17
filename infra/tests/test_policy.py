from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from backend.policy import (
    PolicyError,
    REPLAY_GENERATION_CONTRACT_PATH,
    REPLAY_GENERATION_CONTRACT_SHA256,
    REPLAY_HISTORICAL_CONFIG_PATH,
    REPLAY_HISTORICAL_CONFIG_SHA256,
    REPLAY_CANONICAL_V2_CONFIG_HASH,
    REPLAY_ORIGINAL_MERGE_HASH_V2,
    REPLAY_SNAPSHOT_IDENTITY_PATH,
    REPLAY_SNAPSHOT_IDENTITY_SHA256,
    REPLAY_SUBMITTED_SNAPSHOT_HASH_V2,
    REPLAY_TRANSFORMERS_VERSION,
    accrued_gpu_spend,
    append_event,
    authorized,
    budget_snapshot,
    canonical_hash,
    forbidden_replay_surface_keys,
    has_capacity,
    replay_key_is_sensitive,
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


def configure_replay(value):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs/m2/m2_adapter_merge_fidelity_replay_v2.yaml"
    )
    value["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    comparison = value["config"]["run"]["comparison_id"]
    value["budget_class"] = "screen"
    value["preregistration"]["comparison"] = comparison


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


def test_replay_equivalence_requires_immutable_revision_and_public_runner():
    value = payload()
    configure_replay(value)
    exact_revision = value["config"]["model"]["revision"]
    value["config"]["model"]["revision"] = "__M1_RESOLVE_QWEN__"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="resolved immutable"):
        validate_launch(value)

    value["config"]["model"]["revision"] = exact_revision
    value["config_hash"] = canonical_hash(value["config"])
    assert validate_launch(value).task_kind == "experiment"

    value["config"]["run"]["command"] = ["python", "-m", "paid.judge"]
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="canonical prospective config"):
        validate_launch(value)


def test_replay_equivalence_policy_rejects_paid_or_hidden_surfaces():
    value = payload()
    configure_replay(value)
    value["config"]["model"]["revision"] = "a" * 40
    value["config"]["judge"] = {"model": "remote"}
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="paid or hidden"):
        validate_launch(value)


def test_replay_policy_rejects_runtime_or_canonical_binding_substitution():
    value = payload()
    configure_replay(value)
    value["config"]["model"]["revision"] = "a" * 40
    value["config"]["runtime"]["transformers_version"] = "4.57.5"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="Transformers version"):
        validate_launch(value)

    value["config"]["runtime"]["transformers_version"] = REPLAY_TRANSFORMERS_VERSION
    value["config"]["workflow"]["historical_sampling_config"] = "/tmp/substitute.yaml"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="canonical frozen contract bindings"):
        validate_launch(value)


@pytest.mark.parametrize(
    "config",
    [
        {"workflow": {"runtime": {"judge_url": "https://example.invalid"}}},
        {"runtime": [{"provider-config": {"name": "remote"}}]},
        {"nested": {"sealedEvaluator": "remote"}},
        {"nested": {"hidden_data": "/private"}},
        {"nested": {"apiKey": "secret"}},
    ],
)
def test_recursive_replay_surface_alias_scan(config):
    assert forbidden_replay_surface_keys(config)


@pytest.mark.parametrize(
    "alias",
    [
        "credential", "client_secret", "access_token", "authentication",
        "signing_key", "private_endpoint", "remote_service",
        "credentialStore", "clientSecret", "authConfig", "serviceUrl",
    ],
)
def test_recursive_replay_surface_scan_rejects_neutral_credential_aliases(alias):
    assert forbidden_replay_surface_keys({"nested": {alias: "private-value"}})


@pytest.mark.parametrize(
    "field",
    [
        "split_special_tokens", "extra_special_tokens", "all_special_tokens",
        "all_special_tokens_extended", "spaces_between_special_tokens",
        "token_healing", "image_token_id", "video_token_id",
        "vision_start_token_id",
    ],
)
def test_replay_surface_scan_allows_standard_model_token_fields(field):
    assert replay_key_is_sensitive(field) is False
    assert forbidden_replay_surface_keys({"public_metadata": {field: "value"}}) == []


def test_replay_surface_scan_rejects_oauth_id_token():
    assert replay_key_is_sensitive("id_token") is True
    assert forbidden_replay_surface_keys({"nested": {"id_token": "private"}})


def test_replay_v2_canonical_hash_is_frozen():
    assert REPLAY_CANONICAL_V2_CONFIG_HASH == (
        "ee76ca0ecda72321f07cecd1c70fba5905779321e3169579e357bafdad4cd1da"
    )


def test_14b_requires_human_flag():
    value = payload()
    value["config"]["model"]["base"] = "Qwen/Qwen3-14B"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="human approval"):
        validate_launch(value)
    value["human_scaleup_approved"] = True
    assert validate_launch(value).comparison_id == "M2-A-vs-SFT"


def test_frozen_estimator_audit_is_hash_bound_and_checkpoint_scoped():
    value = payload()
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs/m2/m2_frozen_estimator_audit_4b_v2.yaml"
    )
    value["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    value["budget_class"] = "screen"
    value["preregistration"]["comparison"] = value["config"]["run"]["comparison_id"]
    value["config_hash"] = canonical_hash(value["config"])
    policy = validate_launch(value)
    assert policy.comparison_id == "M2-frozen-estimator-audit-4b-v2"
    assert policy.worst_case_cost_usd < 5.0

    value["config"]["audit"]["replicates"] = 8
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="frozen contract"):
        validate_launch(value)


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


def test_lower_variance_briefs_require_two_frozen_providers_and_exact_corpus():
    value = payload()
    value["budget_class"] = "promo"
    value["config"] = {
        "run": {
            "comparison_id": "M2-lower-variance-train-briefs-v1",
            "budget_class": "promo",
            "task_kind": "brief_synthesis",
        },
        "compute": {"gpus": 1, "timeout_min": 480},
        "data": {
            "input_uri": "modal-volume://humanwrite-checkpoints/data/clean-train.jsonl",
            "output_uri": "modal-volume://humanwrite-checkpoints/data/train-briefs.jsonl",
            "input_sha256": "a" * 64,
            "max_records": 1024,
        },
        "api": {
            "protocol": "dftr.lower_variance_briefs.two_provider.v1",
            "metadata_model": "qwen/qwen3-32b",
            "outline_model": "openai/gpt-5-mini",
            "max_cost_usd": 20.0,
        },
    }
    value["preregistration"]["comparison"] = value["config"]["run"]["comparison_id"]
    value["config_hash"] = canonical_hash(value["config"])
    policy = validate_launch(value)
    assert policy.task_kind == "brief_synthesis"
    assert policy.worst_case_cost_usd == 0
    assert policy.api_reserved_cost_usd == 20.0

    value["config"]["api"]["metadata_model"] = "openai/gpt-5-mini"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="provider models"):
        validate_launch(value)

    value["config"]["api"]["metadata_model"] = "qwen/qwen3-32b"
    value["config"]["data"]["max_records"] = 1000
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="128 or 1024"):
        validate_launch(value)


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


def test_document_cleaning_is_qwen32b_only_and_volume_scoped():
    value = payload()
    value["budget_class"] = "promo"
    value["config"] = {
        "run": {
            "comparison_id": "M2-clean-data-v1",
            "budget_class": "promo",
            "task_kind": "document_cleaning",
        },
        "compute": {"gpus": 1, "timeout_min": 480},
        "data": {
            "input_uri": "modal-volume://humanwrite-checkpoints/data/raw.jsonl",
            "output_uri": "modal-volume://humanwrite-checkpoints/data/clean.jsonl",
            "input_sha256": "a" * 64,
            "max_records": 1024,
            "target_records": 1024,
        },
        "api": {"model": "qwen/qwen3-32b", "max_cost_usd": 40.0},
        "quality": {"min_word_count": 80, "max_word_count": 220},
    }
    value["preregistration"]["comparison"] = "M2-clean-data-v1"
    value["config_hash"] = canonical_hash(value["config"])
    policy = validate_launch(value)
    assert policy.task_kind == "document_cleaning"
    assert policy.api_reserved_cost_usd == 40.0
    assert policy.worst_case_cost_usd == 0.0

    value["config"]["api"]["model"] = "openai/gpt-5-mini"
    value["config_hash"] = canonical_hash(value["config"])
    with pytest.raises(PolicyError, match="qwen/qwen3-32b"):
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
