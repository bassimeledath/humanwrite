"""Independent edge cases for the repaired fidelity public-surface classifier."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess

import pytest

from infra.backend.policy import replay_key_is_sensitive


ROOT = Path(__file__).resolve().parents[2]
TARGET = "ec3f74ba1fa72c2cfeead5b2f25866b5065286f0"
PRIOR = runpy.run_path(
    str(ROOT / "experiments" / "tests" / "test_m2_fidelity_replay_v2_final_independent.py")
)
load_config = PRIOR["load_config"]
accepting_layers = PRIOR["accepting_layers"]
CONFIG_V2 = PRIOR["CONFIG_V2"]
fidelity = PRIOR["fidelity"]


def with_field(field: str, value="public-value") -> dict:
    config = load_config(CONFIG_V2)
    config["runtime"]["public_metadata"] = {"wrapper": [{field: value}]}
    return config


@pytest.mark.parametrize(
    "alias",
    [
        "remoteHTTPSServiceURL",
        "OPENROUTER_API_KEY",
        "oauth2ClientCredentials",
        "HF_ACCESS_TOKEN",
        "privateModelEndpoint",
        "client.authorization.header",
        "SecretStorePath",
        "provider_authentication_service",
        "refreshTokenValue",
        "bearer_token_header",
        "JWT_TOKEN_VALUE",
        "serviceAccountKey",
        "xApiKeyValue",
    ],
)
def test_additional_camel_snake_acronym_private_aliases_reject(alias) -> None:
    assert accepting_layers(with_field(alias, "private-value")) == set()


@pytest.mark.parametrize(
    "field",
    [
        "decoder_start_token_id",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "begin_suppress_tokens",
        "suppress_tokens",
        "max_new_tokens",
        "token_type_ids",
        "additional_special_tokens",
        "specialTokensMap",
        "addedTokens",
        "tokenizerClass",
        "tokenizer_config",
        "clean_up_tokenization_spaces",
        "additional_special_tokens_ids",
        "added_tokens_decoder",
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "mask_token",
        "generationTokenCount",
    ],
)
def test_public_tokenizer_fields_classify_public_but_unknown_config_rejects(field) -> None:
    assert fidelity._is_sensitive_replay_key(field) is False
    assert replay_key_is_sensitive(field) is False
    assert accepting_layers(with_field(field)) == set()


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
def test_standard_public_tokenizer_options_classify_public_but_config_rejects(
    field,
) -> None:
    assert fidelity._is_sensitive_replay_key(field) is False
    assert replay_key_is_sensitive(field) is False
    assert accepting_layers(with_field(field)) == set()


def test_oauth_id_token_alias_is_rejected() -> None:
    assert fidelity._is_sensitive_replay_key("id_token") is True
    assert replay_key_is_sensitive("id_token") is True
    assert accepting_layers(with_field("id_token", "private-value")) == set()


@pytest.mark.parametrize(
    "alias",
    [
        "password",
        "passphrase",
        "clientAssertion",
        "sessionCookie",
        "oauthClientId",
        "codeVerifier",
    ],
)
def test_standard_authentication_aliases_are_rejected(alias) -> None:
    assert accepting_layers(with_field(alias, "private-value")) == set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["runtime"].update(
            public_metadata={"innocuous_label": "public-value"}
        ),
        lambda value: value["workflow"].update(
            extension={"name": "public-value"}
        ),
        lambda value: value.update(metadata={"note": "public-value"}),
    ],
)
def test_unknown_replay_config_fields_fail_closed(mutation) -> None:
    config = load_config(CONFIG_V2)
    mutation(config)
    assert accepting_layers(config) == set()


def test_canonical_replay_configs_need_no_arbitrary_metadata() -> None:
    for config_path in (PRIOR["CONFIG_V1"], CONFIG_V2):
        config = load_config(config_path)
        assert config["runtime"] == {"transformers_version": "4.57.6"}
        assert "public_metadata" not in config
        assert config["workflow"]["generation_contract"] == (
            "configs/m2/canonical_full_brief_generation_v1.json"
        )
        assert config["workflow"]["generation_contract_sha256"] == (
            "db7c970440c451ffd21e634b53df3fa3d556b139e87257dfff7521442fe8f219"
        )
    contract = json.loads(
        (ROOT / "configs/m2/canonical_full_brief_generation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["tokenization"]["return_attention_mask"] is True
    assert contract["prospective_generation"]["max_new_tokens"] == 384


def test_surface_tester_changes_no_implementation_file() -> None:
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
