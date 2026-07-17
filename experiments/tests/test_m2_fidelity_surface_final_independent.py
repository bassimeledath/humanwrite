"""Independent edge cases for the repaired fidelity public-surface classifier."""
from __future__ import annotations

from pathlib import Path
import runpy
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
TARGET = "a32ab25181766aff589619942b27526d9778654d"
PRIOR = runpy.run_path(
    str(ROOT / "experiments" / "tests" / "test_m2_fidelity_replay_v2_final_independent.py")
)
load_config = PRIOR["load_config"]
accepting_layers = PRIOR["accepting_layers"]
CONFIG_V2 = PRIOR["CONFIG_V2"]
ALL_LAYERS = {"workflow", "backend", "gpu_client"}


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
def test_additional_public_tokenizer_and_generation_fields_pass(field) -> None:
    assert accepting_layers(with_field(field)) == ALL_LAYERS


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
@pytest.mark.xfail(
    strict=True,
    reason="standard public tokenizer options remain credential false positives",
)
def test_standard_public_tokenizer_options_are_not_false_positives(field) -> None:
    assert accepting_layers(with_field(field)) == ALL_LAYERS


@pytest.mark.xfail(
    strict=True,
    reason="OAuth id_token is mistaken for public model token-id metadata",
)
def test_oauth_id_token_alias_is_rejected() -> None:
    assert accepting_layers(with_field("id_token", "private-value")) == set()


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
