"""Independent fail-closed verification for prospective fidelity replay v3."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from experiments.m1 import fidelity
from experiments.m1.contracts import M1ConfigError, file_sha256
from infra.backend.policy import PolicyError, canonical_hash, validate_launch


ROOT = Path(__file__).resolve().parents[2]
TARGET = "3c082b6"
CONFIGS = {
    "dftr.adapter_merge_replay.v1": ROOT / "configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml",
    "dftr.adapter_merge_replay.v2": ROOT / "configs/m2/m2_adapter_merge_fidelity_replay_v2.yaml",
    "dftr.adapter_merge_replay.v3": ROOT / "configs/m2/m2_adapter_merge_fidelity_replay_v3.yaml",
}
COMPARISONS = {
    "dftr.adapter_merge_replay.v1": "M2-adapter-merge-fidelity-replay-v1",
    "dftr.adapter_merge_replay.v2": "M2-adapter-merge-fidelity-replay-v2",
    "dftr.adapter_merge_replay.v3": "M2-adapter-merge-fidelity-replay-v3",
}
TOKENIZER_MANIFEST = (
    ROOT / "configs/m2/manifests/m2_adapter_merge_tokenizer_identity_v3.json"
)
V2_MANIFEST = ROOT / "configs/m2/manifests/m2_adapter_merge_snapshot_identity_v2.json"
EXPECTED_SHARED = {
    "added_tokens.json": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer.json": "e7504a40cfacbd5e4662b7e9c171317c7d5f431a052e18381b2df19769cd8f2f",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
EXPECTED_ADAPTER_CONFIG = "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
EXPECTED_MERGED_CONFIG = "a32ee532e3437966f2b52bb0fe0e7c525234dc1034814718b0467d8104a09371"
EXPECTED_ADDITIONS = {
    "max_length": 384,
    "stride": 0,
    "truncation_side": "right",
    "truncation_strategy": "longest_first",
}


def load_config(protocol: str = "dftr.adapter_merge_replay.v3") -> dict:
    return yaml.safe_load(CONFIGS[protocol].read_text(encoding="utf-8"))


def payload(config: dict) -> dict:
    return {
        "run_id": "v3-independent",
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
    client = runpy.run_path(str(ROOT / "infra/gpu"))
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


def load_rehashed_manifest(
    manifest: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    substitute = tmp_path / "tokenizer-identity.json"
    substitute.write_text(json.dumps(manifest), encoding="utf-8")
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
    monkeypatch.setattr(fidelity, "TOKENIZER_IDENTITY_MANIFEST_PATH", "tester/tokenizer-identity.json")
    monkeypatch.setattr(fidelity, "TOKENIZER_IDENTITY_MANIFEST_SHA256", digest)
    loaded, _, _ = fidelity.load_tokenizer_identity_audit(config)
    return loaded


def test_exact_v3_hashes_maps_and_four_field_exception() -> None:
    config = load_config()
    manifest = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    adapter = manifest["adapter"]["file_sha256"]
    merged = manifest["original_merge"]["file_sha256"]

    assert file_sha256(CONFIGS["dftr.adapter_merge_replay.v3"]) == (
        "71ac41a8cbf8eaa0fc4346e3c87cfa7c6e7ea196eeeb8797d0dba819a3d4405b"
    )
    assert canonical_hash(config) == fidelity.CANONICAL_REPLAY_V3_CONFIG_HASH
    assert file_sha256(TOKENIZER_MANIFEST) == (
        "54891d4320ee45db4f4ad08124c22b1696410b70210e63f0da5239e3958a7712"
    )
    assert adapter == {**EXPECTED_SHARED, "tokenizer_config.json": EXPECTED_ADAPTER_CONFIG}
    assert merged == {**EXPECTED_SHARED, "tokenizer_config.json": EXPECTED_MERGED_CONFIG}
    assert manifest["relation"]["exact_match_files"] == list(EXPECTED_SHARED)
    assert manifest["relation"]["metadata_difference_files"] == ["tokenizer_config.json"]
    assert manifest["relation"]["tokenizer_config_json_difference"] == {
        "adapter_only_fields": {},
        "merged_only_fields": EXPECTED_ADDITIONS,
        "changed_fields": {},
    }
    assert accepting_layers(config) == {"worker", "backend", "client"}


@pytest.mark.parametrize("protocol", list(CONFIGS))
def test_each_canonical_protocol_pair_is_accepted(protocol: str) -> None:
    assert accepting_layers(load_config(protocol)) == {"worker", "backend", "client"}


@pytest.mark.parametrize("base_protocol", list(CONFIGS))
@pytest.mark.parametrize("substitute_protocol", list(CONFIGS))
def test_protocol_and_comparison_are_bound_bidirectionally(
    base_protocol: str, substitute_protocol: str
) -> None:
    if base_protocol == substitute_protocol:
        pytest.skip("canonical pair covered separately")
    protocol_mutation = load_config(base_protocol)
    protocol_mutation["workflow"]["protocol_version"] = substitute_protocol
    comparison_mutation = load_config(base_protocol)
    comparison_mutation["run"]["comparison_id"] = COMPARISONS[substitute_protocol]
    assert accepting_layers(protocol_mutation) == set()
    assert accepting_layers(comparison_mutation) == set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(unregistered_surface={"claim": "public"}),
        lambda value: value["runtime"].update(unknown_runtime_attestation=True),
        lambda value: value["adapter_merge_tokenizer_audit"].update(extra_exception=[]),
        lambda value: value["workflow"].update(protocol_version="dftr.adapter_merge_replay.v2"),
        lambda value: value["run"].update(comparison_id="M2-adapter-merge-fidelity-replay-v2"),
        lambda value: value["adapter_merge_tokenizer_audit"].update(
            runtime_attestation="exact_prompt_hashes_before_diagnostics"
        ),
    ],
)
def test_v3_config_mutations_and_unknown_fields_fail_every_layer(mutation) -> None:
    config = load_config()
    mutation(config)
    assert accepting_layers(config) == set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["adapter"]["file_sha256"].pop("vocab.json"),
        lambda value: value["adapter"]["file_sha256"].update({"tokenizer.model": "0" * 64}),
        lambda value: value["original_merge"]["file_sha256"].pop("added_tokens.json"),
        lambda value: value["adapter"]["file_sha256"].update({"vocab.json": "0" * 64}),
        lambda value: value["relation"]["exact_match_files"].reverse(),
        lambda value: value["relation"]["exact_match_files"].pop(),
        lambda value: value["relation"]["metadata_difference_files"].append("vocab.json"),
        lambda value: value["relation"]["metadata_difference_files"].append("tokenizer_config.json"),
        lambda value: value["relation"]["tokenizer_config_json_difference"]["merged_only_fields"].pop("stride"),
        lambda value: value["relation"]["tokenizer_config_json_difference"]["merged_only_fields"].update(stride=1),
        lambda value: value["relation"]["tokenizer_config_json_difference"]["merged_only_fields"].update(extra=True),
        lambda value: value["relation"]["tokenizer_config_json_difference"]["adapter_only_fields"].update(extra=True),
        lambda value: value["relation"]["tokenizer_config_json_difference"]["changed_fields"].update(
            model_max_length={"adapter": 384, "merged": 512}
        ),
        lambda value: value["relation"].update(runtime_authority="shared_tokenizer"),
        lambda value: value["relation"].update(runtime_attestation="diagnostics_then_tokenization"),
    ],
)
def test_rehashed_manifest_map_difference_and_runtime_mutations_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    manifest = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    mutation(manifest)
    with pytest.raises(M1ConfigError, match="tokenizer"):
        load_rehashed_manifest(manifest, tmp_path, monkeypatch)


def test_json_object_field_order_is_semantically_irrelevant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    additions = manifest["relation"]["tokenizer_config_json_difference"]["merged_only_fields"]
    manifest["relation"]["tokenizer_config_json_difference"]["merged_only_fields"] = {
        key: additions[key] for key in reversed(list(additions))
    }
    assert load_rehashed_manifest(manifest, tmp_path, monkeypatch) == manifest


def test_actual_json_difference_recomputation_covers_value_order_and_set() -> None:
    adapter = {"shared": {"nested": True}, "unchanged": 1}
    merged = {
        "truncation_strategy": "longest_first",
        "unchanged": 1,
        "stride": 0,
        "shared": {"nested": True},
        "max_length": 384,
        "truncation_side": "right",
    }
    expected = {
        "adapter_only_fields": {},
        "merged_only_fields": EXPECTED_ADDITIONS,
        "changed_fields": {},
    }
    assert fidelity._tokenizer_config_json_difference(adapter, merged) == expected
    assert fidelity._tokenizer_config_json_difference(
        dict(reversed(list(adapter.items()))), dict(reversed(list(merged.items())))
    ) == expected

    changed = copy.deepcopy(merged)
    changed["shared"] = {"nested": False}
    assert fidelity._tokenizer_config_json_difference(adapter, changed)["changed_fields"] == {
        "shared": {"adapter": {"nested": True}, "merged": {"nested": False}}
    }
    extra = copy.deepcopy(merged)
    extra["new_field"] = "new"
    assert fidelity._tokenizer_config_json_difference(adapter, extra)["merged_only_fields"][
        "new_field"
    ] == "new"
    missing = copy.deepcopy(merged)
    missing.pop("unchanged")
    assert fidelity._tokenizer_config_json_difference(adapter, missing)["adapter_only_fields"] == {
        "unchanged": 1
    }


@pytest.mark.parametrize("mismatch", ["input_ids", "attention_mask"])
def test_runtime_exact_token_and_mask_mismatch_stops_before_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    config = load_config()
    fingerprints = list(config["sampling"]["dev_subset_fingerprints"])
    monkeypatch.setattr(fidelity, "validate_replay_spec", lambda value: (fingerprints, [101, 202, 303]))
    monkeypatch.setattr(fidelity, "load_generation_contract", lambda value: ({}, Path("contract"), "sha"))
    monkeypatch.setattr(fidelity, "verify_runtime_version", lambda config, contract: "4.57.6")

    expected_hashes = {
        str(config["workflow"]["historical_sampling_config"]): fidelity.CANONICAL_HISTORICAL_CONFIG_SHA256,
        str(config["workflow"]["fixed_manifest"]): str(config["workflow"]["fixed_manifest_sha256"]),
        str(config["data"]["dev_path"]): str(config["data"]["dev_briefs_sha256"]),
        str(config["archive"]["index_path"]): str(config["archive"]["index_sha256"]),
        str(config["workflow"]["serializer_source"]): str(config["workflow"]["serializer_source_sha256"]),
    }
    monkeypatch.setattr(fidelity, "resolve_repo_path", lambda value: Path(str(value)))
    monkeypatch.setattr(fidelity, "file_sha256", lambda path: expected_hashes[str(path)])
    monkeypatch.setattr(
        fidelity,
        "read_structured",
        lambda path: {"dev_briefs_sha256": config["data"]["dev_briefs_sha256"]},
    )
    monkeypatch.setattr(
        fidelity,
        "load_jsonl",
        lambda path: [{"fingerprint": fingerprint} for fingerprint in fingerprints],
    )
    monkeypatch.setattr(fidelity, "verify_artifact_identities", lambda value: {"tokenizer": {}})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(fidelity, "build_run_paths", lambda config, run_id: (tmp_path, checkpoint))
    tokenizer = SimpleNamespace(chat_template="same-template")
    monkeypatch.setattr(
        fidelity,
        "_load_models",
        lambda config, contract: (object(), object(), tokenizer, tokenizer),
    )
    adapter_encoded = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    merged_encoded = copy.deepcopy(adapter_encoded)
    merged_encoded[mismatch][-1] = 0 if mismatch == "attention_mask" else 3
    calls = iter(
        [
            ([{"fingerprint": "adapter", **adapter_encoded}], [adapter_encoded]),
            ([{"fingerprint": "adapter", **merged_encoded}], [merged_encoded]),
        ]
    )
    monkeypatch.setattr(fidelity, "_tokenize_prompts", lambda *args, **kwargs: next(calls))
    diagnostics_called = []
    monkeypatch.setattr(fidelity, "_run_diagnostics", lambda *args, **kwargs: diagnostics_called.append(True))

    with pytest.raises(M1ConfigError, match="tokenization attestations differ"):
        fidelity.replay_equivalence(config, "runtime-attestation-test", render_prompt=lambda *args: "prompt")
    assert diagnostics_called == []


def test_v1_v2_bytes_and_v3_parent_scope_are_immutable() -> None:
    assert file_sha256(CONFIGS["dftr.adapter_merge_replay.v1"]) == (
        "8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c"
    )
    assert file_sha256(CONFIGS["dftr.adapter_merge_replay.v2"]) == (
        "a5f0504dfdcfd12cfda5081e068919a603a395c7155a725bd9e7c13016ba1d8c"
    )
    assert file_sha256(V2_MANIFEST) == (
        "602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c"
    )


@pytest.mark.xfail(
    strict=True,
    reason="actual tokenizer map silently filters an unmatched tokenizer.model file",
)
def test_actual_adapter_and_merge_tokenizer_file_maps_reject_unmatched_files(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    adapter.mkdir()
    merged.mkdir()
    for name in set(EXPECTED_SHARED) | {"tokenizer_config.json"}:
        (adapter / name).write_text(f"shared-{name}", encoding="utf-8")
        (merged / name).write_text(f"shared-{name}", encoding="utf-8")
    (adapter / "tokenizer.model").write_text("adapter-only-tokenizer-surface", encoding="utf-8")

    adapter_map = fidelity._tokenizer_file_map(adapter)
    merged_map = fidelity._tokenizer_file_map(merged)
    assert adapter_map != merged_map


def test_independent_tester_does_not_modify_v3_implementation() -> None:
    paths = [
        "experiments/m1/fidelity.py",
        "infra/backend/policy.py",
        "infra/gpu",
        "configs/m2/m2_adapter_merge_fidelity_replay_v3.yaml",
        "configs/m2/manifests/m2_adapter_merge_tokenizer_identity_v3.json",
        "configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml",
        "configs/m2/m2_adapter_merge_fidelity_replay_v2.yaml",
        "configs/m2/manifests/m2_adapter_merge_snapshot_identity_v2.json",
    ]
    output = subprocess.check_output(
        ["git", "diff", "--name-only", TARGET, "--", *paths], cwd=ROOT, text=True
    )
    assert output == ""
