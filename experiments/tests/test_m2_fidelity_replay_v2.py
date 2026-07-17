from __future__ import annotations

import copy
from pathlib import Path
import runpy

import pytest
import yaml

from experiments.m1 import fidelity
from experiments.m1.contracts import M1ConfigError, file_sha256
from infra.backend.policy import (
    PolicyError,
    canonical_hash,
    replay_key_is_sensitive,
    validate_launch,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_V1 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v2.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_V2.read_text(encoding="utf-8"))


def _payload(config: dict) -> dict:
    return {
        "run_id": "dftr-test",
        "config": config,
        "config_hash": canonical_hash(config),
        "git_sha": "a" * 40,
        "budget_class": "screen",
        "preregistration": {
            "kind": "prereg",
            "comparison": config["run"]["comparison_id"],
            "status": "open",
        },
        "human_scaleup_approved": False,
    }


def test_v2_binds_original_merge_and_separate_submitted_snapshot_audit() -> None:
    config = _config()
    fingerprints, seeds = fidelity.validate_replay_spec(config)
    manifest, path, digest = fidelity.load_snapshot_identity_audit(config)

    assert len(fingerprints) == 16
    assert seeds == [101, 202, 303]
    assert config["run"]["comparison_id"] == "M2-adapter-merge-fidelity-replay-v2"
    assert config["workflow"]["protocol_version"] == "dftr.adapter_merge_replay.v2"
    assert config["artifacts"]["merged_path"] == (
        "/checkpoints/runs/dftr-1784224462-c1f83ed3/merged-model"
    )
    assert config["artifacts"]["merged_content_hash"] == "7f095c31e83f8b03"
    assert config["submitted_snapshot_audit"]["canonical_directory_hash"] == (
        "0f437f62bc1cca0c"
    )
    assert digest == file_sha256(path)
    assert manifest["relation"]["difference_files"] == [
        "generation_config.json",
        "train_config.json",
    ]
    assert set(manifest["submitted_snapshot"]["identical_file_sha256"]) == (
        set(manifest["original_merge"]["file_sha256"])
        - {"generation_config.json", "train_config.json"}
    )


def test_snapshot_metadata_hashes_are_exact_and_generation_args_remain_explicit() -> None:
    manifest, _, _ = fidelity.load_snapshot_identity_audit(_config())
    differences = manifest["submitted_snapshot"]["metadata_differences"]
    assert differences["generation_config.json"] == {
        "original_sha256": "64d86df2173901c58389974bde21f7d2ab9eb7d79f35a337753329d39cf265c0",
        "snapshot_sha256": "0ba4fa0fce9b70e3a1a830c618de7fd1a1e4adb3008eaa147fa22aa35550d0f0",
        "classification": "submission_generation_metadata",
    }
    assert differences["train_config.json"] == {
        "original_sha256": "a09d02a3fce6aa2b2e4447dd69b24493d97694b63be0151e455e513ae4b93ef2",
        "snapshot_sha256": "bdeb26ea942bd28eeae6d4522849636c3459b8406b00288380ba8b41c7a3ba18",
        "classification": "submission_provenance_metadata",
    }
    assert manifest["relation"]["generation_arguments_are_explicit"] is True
    assert manifest["relation"]["generation_arguments_authority"] == (
        "configs/m2/canonical_full_brief_generation_v1.json"
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda value: value["artifacts"].update(
                merged_content_hash="0f437f62bc1cca0c"
            ),
            "original merge identity",
        ),
        (
            lambda value: value["submitted_snapshot_audit"].update(
                canonical_directory_hash="7f095c31e83f8b03"
            ),
            "submitted snapshot identity",
        ),
        (
            lambda value: value["submitted_snapshot_audit"].update(
                metadata_difference_files=["generation_config.json"]
            ),
            "metadata difference file set",
        ),
    ],
)
def test_v2_identity_substitution_fails_closed(mutation, error) -> None:
    config = _config()
    mutation(config)
    with pytest.raises(M1ConfigError, match=error):
        fidelity.validate_replay_spec(config)


def test_backend_policy_accepts_exact_v2_and_rejects_identity_substitution() -> None:
    config = _config()
    assert validate_launch(_payload(config)).comparison_id == (
        "M2-adapter-merge-fidelity-replay-v2"
    )
    changed = copy.deepcopy(config)
    changed["submitted_snapshot_audit"]["metadata_difference_files"].reverse()
    with pytest.raises(PolicyError, match="identity repair"):
        validate_launch(_payload(changed))


def test_gpu_client_accepts_exact_v2_and_rejects_identity_substitution() -> None:
    client = runpy.run_path(str(ROOT / "infra" / "gpu"))
    validate_submit = client["_validate_submit"]
    validate_submit.__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg",
        "comparison": comparison,
        "status": "open",
    }

    comparison, preregistration = validate_submit(_config(), "screen")
    assert comparison == "M2-adapter-merge-fidelity-replay-v2"
    assert preregistration["comparison"] == comparison

    changed = _config()
    changed["submitted_snapshot_audit"]["canonical_directory_hash"] = (
        "7f095c31e83f8b03"
    )
    with pytest.raises(SystemExit):
        validate_submit(changed, "screen")


def test_v1_config_remains_byte_identical_and_historical() -> None:
    assert file_sha256(CONFIG_V1) == (
        "8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c"
    )
    historical = yaml.safe_load(CONFIG_V1.read_text(encoding="utf-8"))
    assert historical["workflow"]["protocol_version"] == "dftr.adapter_merge_replay.v1"
    assert historical["artifacts"]["merged_content_hash"] == "0f437f62bc1cca0c"


def test_only_exact_canonical_v1_identity_can_use_historical_protocol() -> None:
    historical = yaml.safe_load(CONFIG_V1.read_text(encoding="utf-8"))
    assert fidelity.validate_replay_spec(historical)[1] == [101, 202, 303]
    assert validate_launch(_payload(historical)).comparison_id == (
        "M2-adapter-merge-fidelity-replay-v1"
    )
    client = runpy.run_path(str(ROOT / "infra" / "gpu"))
    validate_submit = client["_validate_submit"]
    validate_submit.__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg", "comparison": comparison, "status": "open",
    }
    assert validate_submit(historical, "screen")[0] == (
        "M2-adapter-merge-fidelity-replay-v1"
    )

    changed = copy.deepcopy(historical)
    changed["run"]["arm"] = "prospective-substitute"
    with pytest.raises(M1ConfigError, match="canonical historical config"):
        fidelity.validate_replay_spec(changed)
    with pytest.raises(PolicyError, match="canonical historical config"):
        validate_launch(_payload(changed))
    with pytest.raises(SystemExit):
        validate_submit(changed, "screen")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: (
            value["workflow"].update(protocol_version="dftr.adapter_merge_replay.v1"),
            value["artifacts"].update(merged_content_hash="0f437f62bc1cca0c"),
        ),
        lambda value: value["submitted_snapshot_audit"].update(
            weights_tokenizer_index_identity="not-exact"
        ),
        lambda value: value["runtime"].update(
            nested={"credential": "private-value"}
        ),
    ],
)
def test_workflow_backend_and_client_reject_all_v2_launch_bypasses(mutation) -> None:
    config = _config()
    mutation(config)
    with pytest.raises(M1ConfigError):
        fidelity.validate_replay_spec(config)
    with pytest.raises(PolicyError):
        validate_launch(_payload(config))

    client = runpy.run_path(str(ROOT / "infra" / "gpu"))
    validate_submit = client["_validate_submit"]
    validate_submit.__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg", "comparison": comparison, "status": "open",
    }
    with pytest.raises(SystemExit):
        validate_submit(config, "screen")


@pytest.mark.parametrize(
    "field",
    [
        "split_special_tokens",
        "extra_special_tokens",
        "all_special_tokens",
        "all_special_tokens_extended",
        "spaces_between_special_tokens",
        "token_healing",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
    ],
)
def test_standard_public_model_token_fields_are_not_sensitive(field: str) -> None:
    assert fidelity._is_sensitive_replay_key(field) is False
    assert replay_key_is_sensitive(field) is False


def test_oauth_id_token_is_sensitive_in_worker_and_launch_policy() -> None:
    assert fidelity._is_sensitive_replay_key("id_token") is True
    assert replay_key_is_sensitive("id_token") is True
