from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import re

import pytest
import yaml

from experiments.m1 import fidelity
from experiments.m1.contracts import M1ConfigError
from infra.backend.policy import PolicyError, canonical_hash, validate_launch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v1.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_replay_rejects_reordered_frozen_fingerprints() -> None:
    config = _config()
    config["sampling"]["dev_subset_fingerprints"].reverse()

    with pytest.raises(M1ConfigError, match="fingerprint"):
        fidelity.validate_replay_spec(config)


def test_replay_rejects_nested_judge_url_surface() -> None:
    config = _config()
    config["runtime"] = {"judge_url": "https://example.invalid"}

    with pytest.raises(M1ConfigError, match="paid or hidden"):
        fidelity.validate_replay_spec(config)


def test_backend_policy_rejects_nested_judge_surface() -> None:
    config = {
        "run": {
            "comparison_id": "M2-adapter-merge-fidelity-replay-v1",
            "budget_class": "smoke",
            "command": ["python", "-m", "experiments.runner"],
        },
        "model": {"base": "Qwen/Qwen3-4B", "revision": "a" * 40},
        "compute": {"gpu": "L4", "gpus": 1, "timeout_min": 10},
        "workflow": {
            "step": "replay_equivalence",
            "protocol_version": "dftr.adapter_merge_replay.v1",
            "runtime": {"judge": {"url": "https://example.invalid"}},
        },
    }
    payload = {
        "run_id": "dftr-test",
        "config": config,
        "config_hash": canonical_hash(config),
        "git_sha": "a" * 40,
        "budget_class": "smoke",
        "preregistration": {
            "kind": "prereg",
            "comparison": config["run"]["comparison_id"],
            "status": "open",
        },
        "human_scaleup_approved": False,
    }

    with pytest.raises(PolicyError, match="paid or hidden"):
        validate_launch(payload)


def test_generate_one_accepts_the_per_record_generator() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            return ",".join(str(token_id) for token_id in token_ids)

    output_ids, _ = fidelity._generate_one(
        model,
        Tokenizer(),
        {"input_ids": [1, 3], "attention_mask": [1, 1]},
        generation={
            "max_new_tokens": 2,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        generator=torch.Generator(device="cpu").manual_seed(1),
    )
    assert len(output_ids) == 2


def test_scoped_rng_restores_global_state_and_is_transformers_order_group_invariant() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            return ",".join(str(token_id) for token_id in token_ids)

    fingerprints = _config()["sampling"]["dev_subset_fingerprints"][:3]
    encoded = {"input_ids": [1, 3], "attention_mask": [1, 1]}
    generation = {
        "max_new_tokens": 3,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    def generate(grouped_order):
        result = {}
        for group in grouped_order:
            for fingerprint in group:
                seed = fidelity.derive_record_seed(101, fingerprint)
                result[fingerprint] = fidelity._generate_one(
                    model,
                    Tokenizer(),
                    encoded,
                    generation=generation,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                )[0]
        return result

    torch.manual_seed(991)
    state_before = torch.random.get_rng_state().clone()
    forward = generate([fingerprints])
    assert torch.equal(torch.random.get_rng_state(), state_before)
    reverse = generate([[value] for value in reversed(fingerprints)])
    regrouped = generate([fingerprints[:1], fingerprints[1:]])
    assert forward == reverse == regrouped
    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_replay_worker_exactly_pins_transformers() -> None:
    worker_source = (ROOT / "infra" / "backend" / "modal_app.py").read_text(encoding="utf-8")
    assert re.search(r'["\']transformers==\d+\.\d+\.\d+["\']', worker_source)


def test_historical_binding_path_and_hash_are_canonical(tmp_path: Path) -> None:
    config = _config()
    historical_path = ROOT / config["workflow"]["historical_sampling_config"]
    historical = yaml.safe_load(historical_path.read_text(encoding="utf-8"))
    config["sampling"]["dev_subset_fingerprints"].reverse()
    historical["sampling"]["dev_subset_fingerprints"] = list(
        config["sampling"]["dev_subset_fingerprints"]
    )
    substitute = tmp_path / "substitute_historical.yaml"
    substitute.write_text(yaml.safe_dump(historical, sort_keys=False), encoding="utf-8")
    config["workflow"]["historical_sampling_config"] = str(substitute)
    config["workflow"]["historical_sampling_config_sha256"] = hashlib.sha256(
        substitute.read_bytes()
    ).hexdigest()

    with pytest.raises(M1ConfigError, match="canonical|frozen historical"):
        fidelity.validate_replay_spec(config)


def test_weight_identity_fields_make_serialization_claims_only() -> None:
    source = inspect.getsource(fidelity.verify_artifact_identities)
    assert '"tensor_identity_sha256"' not in source
    assert '"weight_serialization_file_map_identity_sha256"' in source
    assert '"weight_shard_serialization_file_map_identity_sha256"' in source
