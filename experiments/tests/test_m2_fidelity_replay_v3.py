from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest
import yaml

from experiments.m1 import fidelity
from experiments.m1.contracts import M1ConfigError, file_sha256
from infra.backend.policy import PolicyError, canonical_hash, validate_launch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_V1 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v2.yaml"
CONFIG_V3 = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v3.yaml"
TOKENIZER_MANIFEST = (
    ROOT
    / "configs"
    / "m2"
    / "manifests"
    / "m2_adapter_merge_tokenizer_identity_v3.json"
)


def load_config(path: Path = CONFIG_V3) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def payload(config: dict) -> dict:
    return {
        "run_id": "v3-test",
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
    calls = {
        "worker": lambda: fidelity.validate_replay_spec(config),
        "backend": lambda: validate_launch(payload(config)),
        "client": lambda: gpu_validator()(config, "screen"),
    }
    accepted = set()
    for name, call in calls.items():
        try:
            call()
        except (M1ConfigError, PolicyError, SystemExit):
            continue
        accepted.add(name)
    return accepted


def test_v3_exact_config_and_tokenizer_manifest_are_frozen() -> None:
    config = load_config()
    manifest, path, digest = fidelity.load_tokenizer_identity_audit(config)

    assert file_sha256(CONFIG_V3) == fidelity.CANONICAL_REPLAY_V3_CONFIG_SHA256
    assert canonical_hash(config) == fidelity.CANONICAL_REPLAY_V3_CONFIG_HASH
    assert digest == file_sha256(path) == file_sha256(TOKENIZER_MANIFEST)
    assert config["workflow"]["protocol_version"] == "dftr.adapter_merge_replay.v3"
    assert config["run"]["comparison_id"] == "M2-adapter-merge-fidelity-replay-v3"
    assert manifest["adapter"]["file_sha256"]["tokenizer_config.json"] == (
        "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
    )
    assert manifest["original_merge"]["file_sha256"]["tokenizer_config.json"] == (
        "a32ee532e3437966f2b52bb0fe0e7c525234dc1034814718b0467d8104a09371"
    )
    assert accepting_layers(config) == {"worker", "backend", "client"}


def test_v3_permits_only_the_exact_tokenizer_config_json_additions() -> None:
    manifest, _, _ = fidelity.load_tokenizer_identity_audit(load_config())
    relation = manifest["relation"]
    assert relation["metadata_difference_files"] == ["tokenizer_config.json"]
    assert relation["tokenizer_config_json_difference"] == {
        "adapter_only_fields": {},
        "merged_only_fields": {
            "max_length": 384,
            "stride": 0,
            "truncation_side": "right",
            "truncation_strategy": "longest_first",
        },
        "changed_fields": {},
    }
    assert relation["runtime_authority"] == (
        "independent_adapter_and_merge_tokenization"
    )
    assert relation["runtime_attestation"] == (
        "exact_prompt_token_and_attention_mask_before_diagnostics"
    )


def test_runtime_json_difference_recomputation_is_exact() -> None:
    adapter = {"shared": 1, "nested": {"same": True}}
    merged = {
        **adapter,
        "max_length": 384,
        "stride": 0,
        "truncation_side": "right",
        "truncation_strategy": "longest_first",
    }
    assert fidelity._tokenizer_config_json_difference(adapter, merged) == {
        "adapter_only_fields": {},
        "merged_only_fields": fidelity.MERGED_TOKENIZER_CONFIG_ADDITIONS,
        "changed_fields": {},
    }
    changed = copy.deepcopy(merged)
    changed["shared"] = 2
    assert fidelity._tokenizer_config_json_difference(adapter, changed)[
        "changed_fields"
    ] == {"shared": {"adapter": 1, "merged": 2}}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["workflow"].update(
            protocol_version="dftr.adapter_merge_replay.v2"
        ),
        lambda value: value["run"].update(
            comparison_id="M2-adapter-merge-fidelity-replay-v2"
        ),
        lambda value: value["adapter_merge_tokenizer_audit"].update(
            adapter_tokenizer_config_sha256=(
                "a32ee532e3437966f2b52bb0fe0e7c525234dc1034814718b0467d8104a09371"
            )
        ),
        lambda value: value["adapter_merge_tokenizer_audit"].update(
            merged_tokenizer_config_sha256=(
                "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
            )
        ),
        lambda value: value["adapter_merge_tokenizer_audit"].update(
            tokenizer_metadata_difference_files=[]
        ),
        lambda value: value["adapter_merge_tokenizer_audit"].update(
            shared_file_identity="semantic_equivalence"
        ),
        lambda value: value["adapter_merge_tokenizer_audit"].update(
            runtime_attestation="diagnostics_only"
        ),
        lambda value: value["adapter_merge_tokenizer_audit"].update(
            identity_manifest_sha256="0" * 64
        ),
    ],
)
def test_v3_config_substitutions_fail_closed_at_every_layer(mutation) -> None:
    config = load_config()
    mutation(config)
    assert accepting_layers(config) == set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["relation"]["metadata_difference_files"].append(
            "special_tokens_map.json"
        ),
        lambda value: value["relation"]["tokenizer_config_json_difference"][
            "merged_only_fields"
        ].update(max_length=1024),
        lambda value: value["relation"].update(
            runtime_attestation="exact_prompt_only"
        ),
    ],
)
def test_rehashed_manifest_semantic_substitutions_still_fail_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    manifest = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    mutation(manifest)
    substitute = tmp_path / "tokenizer-identity.json"
    substitute.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(substitute.read_bytes()).hexdigest()
    config = load_config()
    config["adapter_merge_tokenizer_audit"].update(
        identity_manifest="tester/tokenizer-identity.json",
        identity_manifest_sha256=digest,
    )
    original_resolve = fidelity.resolve_repo_path
    monkeypatch.setattr(
        fidelity,
        "resolve_repo_path",
        lambda value: substitute
        if value == "tester/tokenizer-identity.json"
        else original_resolve(value),
    )
    monkeypatch.setattr(
        fidelity, "TOKENIZER_IDENTITY_MANIFEST_PATH", "tester/tokenizer-identity.json"
    )
    monkeypatch.setattr(fidelity, "TOKENIZER_IDENTITY_MANIFEST_SHA256", digest)

    with pytest.raises(M1ConfigError, match="tokenizer"):
        fidelity.load_tokenizer_identity_audit(config)


def test_v1_and_v2_configs_and_manifest_remain_byte_identical() -> None:
    assert file_sha256(CONFIG_V1) == (
        "8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c"
    )
    assert file_sha256(CONFIG_V2) == (
        "a5f0504dfdcfd12cfda5081e068919a603a395c7155a725bd9e7c13016ba1d8c"
    )
    assert file_sha256(
        ROOT
        / "configs"
        / "m2"
        / "manifests"
        / "m2_adapter_merge_snapshot_identity_v2.json"
    ) == "602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c"
    assert accepting_layers(load_config(CONFIG_V1)) == {"worker", "backend", "client"}
    assert accepting_layers(load_config(CONFIG_V2)) == {"worker", "backend", "client"}


@pytest.mark.parametrize(
    "extra_name",
    ["tokenizer.model", "spiece.model", "sentencepiece.bpe.model"],
)
def test_tokenizer_map_discovers_additional_model_artifacts(
    tmp_path: Path, extra_name: str
) -> None:
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    adapter.mkdir()
    merged.mkdir()
    (adapter / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (merged / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (adapter / extra_name).write_text("adapter-only\n", encoding="utf-8")

    assert fidelity._tokenizer_file_map(adapter) != fidelity._tokenizer_file_map(merged)
