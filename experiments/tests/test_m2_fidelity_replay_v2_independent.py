"""Independent public, CPU-only adversarial tests for fidelity replay v2."""
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
CONFIG_V1 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v2.yaml"
IDENTITY = ROOT / "configs" / "m2" / "manifests" / "m2_adapter_merge_snapshot_identity_v2.json"
TARGET = "092d3c0b8b93c95734a30e708dfe6a2d47c68219"


def config_v2() -> dict:
    return yaml.safe_load(CONFIG_V2.read_text(encoding="utf-8"))


def payload(config: dict) -> dict:
    return {
        "run_id": "independent-test",
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


def accepting_layers(config: dict) -> list[str]:
    layers = {
        "workflow": lambda: fidelity.validate_replay_spec(config),
        "backend": lambda: validate_launch(payload(config)),
        "gpu_client": lambda: gpu_validator()(config, "screen"),
    }
    accepted = []
    for name, call in layers.items():
        try:
            call()
        except (M1ConfigError, PolicyError, SystemExit):
            continue
        accepted.append(name)
    return accepted


def test_exact_original_and_snapshot_identities_are_distinct_and_manifest_bound() -> None:
    config = config_v2()
    manifest, path, digest = fidelity.load_snapshot_identity_audit(config)
    assert path == IDENTITY
    assert digest == file_sha256(IDENTITY) == (
        "602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c"
    )
    assert config["artifacts"]["merged_content_hash"] == "7f095c31e83f8b03"
    assert manifest["original_merge"]["canonical_directory_hash"] == "7f095c31e83f8b03"
    assert config["submitted_snapshot_audit"]["canonical_directory_hash"] == "0f437f62bc1cca0c"
    assert manifest["submitted_snapshot"]["canonical_directory_hash"] == "0f437f62bc1cca0c"
    assert manifest["original_merge"]["path"] == config["artifacts"]["merged_path"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["artifacts"].update(merged_content_hash="0f437f62bc1cca0c"),
        lambda value: value["submitted_snapshot_audit"].update(
            canonical_directory_hash="7f095c31e83f8b03"
        ),
        lambda value: value["submitted_snapshot_audit"].update(
            identity_manifest_sha256="0" * 64
        ),
        lambda value: value["submitted_snapshot_audit"].update(
            metadata_difference_files=["train_config.json", "generation_config.json"]
        ),
        lambda value: value["run"].update(
            comparison_id="M2-adapter-merge-fidelity-replay-v1"
        ),
    ],
)
def test_identity_or_v2_comparison_substitution_is_rejected_by_every_layer(mutation) -> None:
    config = config_v2()
    mutation(config)
    assert accepting_layers(config) == []


@pytest.mark.parametrize("mutation", ["drop_shared_file", "add_metadata_difference", "change_shared_hash"])
def test_manifest_file_map_and_difference_set_mutations_fail_semantic_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    config = config_v2()
    manifest = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if mutation == "drop_shared_file":
        manifest["submitted_snapshot"]["identical_file_sha256"].pop("tokenizer.json")
    elif mutation == "add_metadata_difference":
        manifest["submitted_snapshot"]["metadata_differences"]["config.json"] = {
            "original_sha256": manifest["original_merge"]["file_sha256"]["config.json"],
            "snapshot_sha256": "1" * 64,
            "classification": "metadata",
        }
    else:
        manifest["submitted_snapshot"]["identical_file_sha256"]["tokenizer.json"] = "2" * 64
    substitute = tmp_path / "identity.json"
    substitute.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    digest = file_sha256(substitute)
    monkeypatch.setattr(fidelity, "SNAPSHOT_IDENTITY_MANIFEST_PATH", "tester/identity.json")
    monkeypatch.setattr(fidelity, "SNAPSHOT_IDENTITY_MANIFEST_SHA256", digest)
    monkeypatch.setattr(fidelity, "resolve_repo_path", lambda _path: substitute)
    config["submitted_snapshot_audit"]["identity_manifest"] = "tester/identity.json"
    config["submitted_snapshot_audit"]["identity_manifest_sha256"] = digest
    with pytest.raises(M1ConfigError):
        fidelity.load_snapshot_identity_audit(config)


def test_canonical_directory_hash_and_file_map_are_byte_strict(tmp_path: Path) -> None:
    artifact = tmp_path / "original-merge"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"weight bytes")
    (artifact / "tokenizer.json").write_text('{"tokenizer":1}\n')
    (artifact / "generation_config.json").write_text('{"temperature":1}\n')
    expected = {
        name: file_sha256(artifact / name)
        for name in ("model.safetensors", "tokenizer.json", "generation_config.json")
    }
    first = fidelity.canonical_directory_hash(artifact)
    assert fidelity._verify_file_map(artifact, expected, "tester original") == expected

    (artifact / "generation_config.json").write_text('{"temperature":0}\n')
    assert fidelity.canonical_directory_hash(artifact) != first
    with pytest.raises(M1ConfigError, match="SHA-256 mismatch"):
        fidelity._verify_file_map(artifact, expected, "tester original")

    (artifact / "generation_config.json").write_text('{"temperature":1}\n')
    assert fidelity.canonical_directory_hash(artifact) == first
    (artifact / "extra.json").write_text("{}\n")
    assert fidelity.canonical_directory_hash(artifact) != first


@pytest.mark.parametrize(
    "surface",
    [
        {"sealed_endpoint": "https://example.invalid"},
        {"hidden_fixture": "private"},
        {"judge_url": "https://example.invalid"},
        {"provider_token": "private"},
        {"api_key": "private"},
    ],
)
def test_declared_forbidden_surface_aliases_are_rejected_by_every_layer(surface) -> None:
    config = config_v2()
    config["runtime"]["nested"] = surface
    assert accepting_layers(config) == []


@pytest.mark.xfail(strict=True, reason="v2 config can downgrade to v1 and substitute the snapshot hash")
def test_v2_schema_downgrade_is_rejected_by_every_layer() -> None:
    config = config_v2()
    config["workflow"]["protocol_version"] = "dftr.adapter_merge_replay.v1"
    config["artifacts"]["merged_content_hash"] = "0f437f62bc1cca0c"
    assert accepting_layers(config) == []


@pytest.mark.xfail(strict=True, reason="backend/client admit a false exact-serialization identity claim")
def test_snapshot_exact_serialization_claim_is_launch_guarded() -> None:
    config = config_v2()
    config["submitted_snapshot_audit"]["weights_tokenizer_index_identity"] = "not-exact"
    assert accepting_layers(config) == []


@pytest.mark.xfail(strict=True, reason="credential alias is outside every forbidden-surface token list")
def test_credential_surface_is_rejected_by_every_layer() -> None:
    config = config_v2()
    config["runtime"]["credential"] = "private-value"
    assert accepting_layers(config) == []


def test_v1_and_historical_artifacts_are_byte_immutable_at_target_commit() -> None:
    assert file_sha256(CONFIG_V1) == (
        "8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c"
    )
    historical_paths = [
        "configs/m1",
        ":(glob)configs/m2/m2_sealed*",
        "harness",
        "experiments/m1/tier1",
    ]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{TARGET}^", TARGET, "--", *historical_paths],
        cwd=ROOT,
        text=True,
    )
    assert changed == ""
    parent_object = subprocess.check_output(
        ["git", "rev-parse", f"{TARGET}^:configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml"],
        cwd=ROOT,
        text=True,
    ).strip()
    target_object = subprocess.check_output(
        ["git", "rev-parse", f"{TARGET}:configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert parent_object == target_object
