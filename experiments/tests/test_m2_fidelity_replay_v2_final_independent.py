"""Final independent CPU-only verification for fidelity replay v2."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

import pytest
import yaml

from experiments.m1 import fidelity
from experiments.m1.contracts import M1ConfigError, file_sha256
from infra.backend.policy import PolicyError, canonical_hash, validate_launch


ROOT = Path(__file__).resolve().parents[2]
TARGET = "a32ab25181766aff589619942b27526d9778654d"
CONFIG_V1 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v2.yaml"
V1_FILE_SHA256 = "8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c"
V1_CANONICAL_HASH = "859798f2ce66b81a2db32665b7f8fda5a76f5d9e82c64789e7e1f797c4587b9f"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def launch_payload(config: dict) -> dict:
    return {
        "run_id": "final-independent-test",
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


def gpu_validator():
    client = runpy.run_path(str(ROOT / "infra" / "gpu"))
    validate = client["_validate_submit"]
    validate.__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg",
        "comparison": comparison,
        "status": "open",
    }
    return validate


def accepting_layers(config: dict) -> set[str]:
    layers = {
        "workflow": lambda: fidelity.validate_replay_spec(config),
        "backend": lambda: validate_launch(launch_payload(config)),
        "gpu_client": lambda: gpu_validator()(config, "screen"),
    }
    accepted = set()
    for name, call in layers.items():
        try:
            call()
        except (M1ConfigError, PolicyError, SystemExit):
            continue
        accepted.add(name)
    return accepted


def test_target_and_exact_historical_v1_identity_are_preserved() -> None:
    assert subprocess.check_output(
        ["git", "cat-file", "-t", TARGET], cwd=ROOT, text=True
    ).strip() == "commit"
    raw = CONFIG_V1.read_bytes()
    v1 = load_config(CONFIG_V1)
    assert hashlib.sha256(raw).hexdigest() == file_sha256(CONFIG_V1) == V1_FILE_SHA256
    assert canonical_hash(v1) == V1_CANONICAL_HASH
    assert v1["workflow"]["protocol_version"] == "dftr.adapter_merge_replay.v1"
    assert v1["run"]["comparison_id"] == "M2-adapter-merge-fidelity-replay-v1"
    assert v1["artifacts"]["merged_content_hash"] == "0f437f62bc1cca0c"
    assert accepting_layers(v1) == {"workflow", "backend", "gpu_client"}


@pytest.mark.parametrize(
    "base,mutation",
    [
        ("v2", lambda value: value["run"].update(
            comparison_id="M2-adapter-merge-fidelity-replay-v1"
        )),
        ("v2", lambda value: (
            value["workflow"].update(protocol_version="dftr.adapter_merge_replay.v1"),
            value["artifacts"].update(merged_content_hash="0f437f62bc1cca0c"),
        )),
        ("v1", lambda value: value["run"].update(
            comparison_id="M2-adapter-merge-fidelity-replay-v2"
        )),
        ("v1", lambda value: value["workflow"].update(
            protocol_version="dftr.adapter_merge_replay.v2"
        )),
        ("v1", lambda value: value["artifacts"].update(
            merged_content_hash="7f095c31e83f8b03"
        )),
        ("v1", lambda value: value["run"].update(arm="self-consistent-substitute")),
    ],
)
def test_protocol_comparison_and_v1_identity_bind_bidirectionally(base, mutation) -> None:
    config = load_config(CONFIG_V1 if base == "v1" else CONFIG_V2)
    mutation(config)
    assert accepting_layers(config) == set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["submitted_snapshot_audit"].update(
            weights_tokenizer_index_identity="semantic-equivalence"
        ),
        lambda value: value["submitted_snapshot_audit"].update(
            generation_arguments_authority="configs/m2/substitute.json"
        ),
        lambda value: value["submitted_snapshot_audit"].update(
            metadata_difference_files=["train_config.json", "generation_config.json"]
        ),
    ],
)
def test_v2_serialization_and_generation_authority_are_launch_guarded(mutation) -> None:
    config = load_config(CONFIG_V2)
    mutation(config)
    assert accepting_layers(config) == set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["workflow"].update(
            generation_contract="configs/m2/substitute.json"
        ),
        lambda value: value["workflow"].update(generation_contract_sha256="0" * 64),
    ],
)
def test_generation_contract_substitution_fails_launch_and_worker_authority(
    mutation,
) -> None:
    config = load_config(CONFIG_V2)
    mutation(config)
    assert accepting_layers(config) == {"workflow"}
    with pytest.raises(M1ConfigError, match="canonical generation contract"):
        fidelity.load_generation_contract(config)


@pytest.mark.parametrize(
    "alias",
    [
        "remoteServiceUrl",
        "privateEndpointUrl",
        "externalAuthConfig",
        "clientSecretValue",
        "gatewayAccessTokenValue",
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason="wrapped credential aliases bypass every recursive public-only scanner",
)
def test_wrapped_credential_aliases_are_rejected_recursively(alias) -> None:
    config = load_config(CONFIG_V2)
    config["runtime"]["public_metadata"] = {alias: "private-value"}
    assert accepting_layers(config) == set()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokenizer_path", "/public/model/tokenizer.json"),
        ("tokenizer_config", {"use_fast": True}),
        ("weights_tokenizer_index_identity", "exact_serialization_bytes"),
    ],
)
def test_public_tokenizer_metadata_is_not_a_credential_false_positive(field, value) -> None:
    config = load_config(CONFIG_V2)
    config["runtime"]["public_metadata"] = {field: value}
    assert accepting_layers(config) == {"workflow", "backend", "gpu_client"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("special_tokens_map", "special_tokens_map.json"),
        ("added_tokens", "added_tokens.json"),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason="public tokenizer metadata is misclassified as a credential surface",
)
def test_standard_public_token_fields_are_not_credential_false_positives(
    field, value
) -> None:
    config = load_config(CONFIG_V2)
    config["runtime"]["public_metadata"] = {field: value}
    assert accepting_layers(config) == {"workflow", "backend", "gpu_client"}


def test_tester_commit_does_not_modify_fidelity_implementation_surfaces() -> None:
    implementation_paths = [
        "experiments/m1/fidelity.py",
        "infra/backend/policy.py",
        "infra/gpu",
        "configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml",
        "configs/m2/m2_adapter_merge_fidelity_replay_v2.yaml",
        "configs/m2/manifests/m2_adapter_merge_snapshot_identity_v2.json",
    ]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", TARGET, "--", *implementation_paths],
        cwd=ROOT,
        text=True,
    )
    assert changed == ""
