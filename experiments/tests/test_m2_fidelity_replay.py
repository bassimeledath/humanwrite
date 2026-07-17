from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.m1 import fidelity, workflow
from experiments.m1.contracts import M1ConfigError, file_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "m2" / "m2_adapter_merge_fidelity_replay_v1.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_replay_config_has_exact_cardinality_hashes_and_no_paid_surfaces() -> None:
    config = _config()
    fingerprints, seeds = fidelity.validate_replay_spec(config)
    contract, path, digest = fidelity.load_generation_contract(config)

    assert len(fingerprints) == 16
    assert len(set(fingerprints)) == 16
    assert seeds == [101, 202, 303]
    assert contract["dtype"] == "bfloat16"
    assert digest == file_sha256(path)
    assert config["artifacts"]["adapter_sha256"] == (
        "a34c14230f4847001a3a0c4362a3bc26b3a43c1d0ef049e12a7a0d029aacea91"
    )
    assert config["artifacts"]["merged_content_hash"] == "0f437f62bc1cca0c"
    serialized = json.dumps(config).casefold()
    for forbidden in ("api_key", "sealed_eval_url", "provider", "judge"):
        assert forbidden not in serialized


@pytest.mark.parametrize("mutation,error", [
    (lambda value: value["sampling"]["seeds"].reverse(), "sampling seeds"),
    (lambda value: value["sampling"]["dev_subset_fingerprints"].pop(), "16 unique"),
    (lambda value: value["sampling"].update(dev_subset_hash="0" * 64), "subset hash mismatch"),
    (lambda value: value.update(provider={"model": "paid"}), "paid or hidden"),
])
def test_replay_spec_fails_closed_on_scope_or_cardinality_tampering(mutation, error) -> None:
    config = _config()
    mutation(config)
    with pytest.raises(M1ConfigError, match=error):
        fidelity.validate_replay_spec(config)


def test_generation_contract_hash_tampering_is_rejected(tmp_path: Path) -> None:
    config = _config()
    original = ROOT / config["workflow"]["generation_contract"]
    changed = tmp_path / "contract.json"
    changed.write_bytes(original.read_bytes() + b" ")
    config["workflow"]["generation_contract"] = str(changed)

    with pytest.raises(M1ConfigError, match="generation contract SHA-256 mismatch"):
        fidelity.load_generation_contract(config)


def test_file_identity_map_rejects_byte_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    tensor = artifact / "weights.bin"
    tensor.write_bytes(b"first")
    expected = {"weights.bin": file_sha256(tensor)}
    assert fidelity._verify_file_map(artifact, expected, "test") == expected

    tensor.write_bytes(b"second")
    with pytest.raises(M1ConfigError, match="SHA-256 mismatch"):
        fidelity._verify_file_map(artifact, expected, "test")


def test_per_record_rng_is_order_and_batch_invariant() -> None:
    fingerprints = _config()["sampling"]["dev_subset_fingerprints"]
    forward = {
        fingerprint: fidelity.derive_record_seed(101, fingerprint)
        for fingerprint in fingerprints
    }
    reverse = {
        fingerprint: fidelity.derive_record_seed(101, fingerprint)
        for fingerprint in reversed(fingerprints)
    }

    assert forward == reverse
    assert len(set(forward.values())) == 16
    assert fidelity.derive_record_seed(202, fingerprints[0]) != forward[fingerprints[0]]
    assert all(0 <= value < 2**63 for value in forward.values())


def test_scoped_generation_rng_is_order_and_grouping_invariant() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
            sampled = torch.randint(
                3, 16, (input_ids.shape[0], max_new_tokens), device=input_ids.device
            )
            return torch.cat((input_ids, sampled), dim=1)

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            return ",".join(str(token_id) for token_id in token_ids)

    fingerprints = _config()["sampling"]["dev_subset_fingerprints"][:4]
    model = Model().eval()
    encoded = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    generation = {"max_new_tokens": 3, "do_sample": True}

    def generate(grouped_order):
        result = {}
        for group in grouped_order:
            for fingerprint in group:
                seed = fidelity.derive_record_seed(101, fingerprint)
                ids, _ = fidelity._generate_one(
                    model,
                    Tokenizer(),
                    encoded,
                    generation=generation,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                )
                result[fingerprint] = ids
        return result

    forward = generate([fingerprints])
    reversed_singletons = generate([[value] for value in reversed(fingerprints)])
    regrouped = generate([fingerprints[:1], fingerprints[1:3], fingerprints[3:]])
    assert forward == reversed_singletons == regrouped


def test_token_and_mask_hashes_preserve_order_and_values() -> None:
    assert fidelity.hash_token_ids([1, 2, 3]) == fidelity.hash_token_ids([1, 2, 3])
    assert fidelity.hash_token_ids([1, 2, 3]) != fidelity.hash_token_ids([3, 2, 1])
    assert fidelity.hash_token_ids([1, 0, 1]) != fidelity.hash_token_ids([1, 1, 1])


def test_workflow_dispatches_replay_without_touching_judges(monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.m1.fidelity as module

    captured = {}

    def fake(config, run_id, *, render_prompt):
        captured.update(config=config, run_id=run_id, render_prompt=render_prompt)
        return {"verdict": "test-only"}

    monkeypatch.setattr(module, "replay_equivalence", fake)
    config = {"workflow": {"step": "replay_equivalence"}}
    assert workflow.run_m1(config, "dftr-test") == {"verdict": "test-only"}
    assert captured["run_id"] == "dftr-test"
    assert captured["render_prompt"] is workflow._render_prompt
