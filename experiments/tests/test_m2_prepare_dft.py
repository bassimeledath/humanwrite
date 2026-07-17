from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

HARNESS_SRC = Path(__file__).resolve().parents[2] / "harness" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from experiments import runner
from experiments.m2 import dft, prepare_dft, representation
from experiments.m2.prepare_dft import (
    BASE_MODEL,
    BASE_REVISION,
    PREPARE_DFT_SCHEMA,
    PREPARE_DFT_STEP,
    PrepareDFTError,
    derive_training_bandwidths,
    preparation_contract_payload,
    run_prepare_dft,
    validate_prepare_dft_config,
)
from experiments.m2.representation import (
    TRAINING_BANDWIDTH_DERIVATION,
    canonical_hash,
)
from infra.backend.policy import PolicyError, canonical_hash as policy_hash, validate_launch
from infra.backend.local_backend import _count_generated_tokens


SHA = "a" * 64


def valid_prepare_config() -> dict:
    config = {
        "artifact_schema": PREPARE_DFT_SCHEMA,
        "run": {
            "comparison_id": "M2-training-bandwidths-v1",
            "arm": "training-bandwidths",
            "budget_class": "smoke",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 0,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 20},
        "model": {"base": BASE_MODEL, "revision": BASE_REVISION, "torch_dtype": "bfloat16"},
        "initial_adapter": {
            "path": "/checkpoints/runs/source/seed-11",
            "adapter_model_sha256": SHA,
            "adapter_config_sha256": SHA,
            "file_manifest_sha256": SHA,
        },
        "data": {
            "human_targets_path": "/checkpoints/data/training-humans.jsonl",
            "human_targets_sha256": SHA,
            "human_text_field": "completion",
        },
        "representation": {
            "model": BASE_MODEL,
            "revision": BASE_REVISION,
            "layer": -1,
            "pooling": "attention_masked_mean",
            "normalize": True,
            "role": "training_only_not_measurement_v2",
            "batch_size": 4,
            "max_tokens": 256,
        },
        "derivation": copy.deepcopy(TRAINING_BANDWIDTH_DERIVATION),
        "runtime": {
            "torch_version": "2.9.1",
            "transformers_version": "4.57.6",
            "peft_version": "0.18.0",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "output": {"filename": "training_bandwidths.json", "overwrite": False},
        "workflow": {
            "protocol_version": PREPARE_DFT_SCHEMA,
            "step": PREPARE_DFT_STEP,
            "preparation_contract_sha256": "0" * 64,
        },
    }
    config["workflow"]["preparation_contract_sha256"] = canonical_hash(
        preparation_contract_payload(config)
    )
    return config


def test_prepare_config_is_exact_and_hash_bound():
    config = valid_prepare_config()
    assert validate_prepare_dft_config(config) is config
    config["representation"]["max_tokens"] += 1
    with pytest.raises(PrepareDFTError, match="contract hash"):
        validate_prepare_dft_config(config)
    config = valid_prepare_config()
    config["unexpected"] = True
    with pytest.raises(PrepareDFTError, match="exactly"):
        validate_prepare_dft_config(config)


def test_bandwidth_derivation_matches_float64_median_and_scales():
    result = derive_training_bandwidths(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32),
        [0.25, 0.5, 1.0, 2.0, 4.0],
    )
    assert result["median_positive_squared_distance"] == pytest.approx(2.0)
    assert result["values"] == pytest.approx([0.125, 0.5, 2.0, 8.0, 32.0])
    assert result["total_unordered_pair_count"] == 3
    assert result["positive_pair_distance_count"] == 3
    assert result["zero_distance_count"] == 0


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]),
    ],
)
def test_bandwidth_derivation_rejects_duplicate_or_nonfinite_embeddings(embeddings):
    with pytest.raises(PrepareDFTError, match="duplicate|invalid"):
        derive_training_bandwidths(embeddings, [1.0])


def test_prepare_run_writes_self_auditing_v2_and_zero_tokens(monkeypatch, tmp_path):
    config = valid_prepare_config()
    checkpoint_dir = tmp_path / "state" / "artifacts" / "prepare-test"
    monkeypatch.setenv("DFTR_CHECKPOINT_DIR", str(checkpoint_dir))
    monkeypatch.setattr(
        prepare_dft,
        "_verify_inputs",
        lambda _config: (["human one", "human two", "human three"], {
            "torch_version": "2.9.1",
            "transformers_version": "4.57.6",
            "peft_version": "0.18.0",
        }),
    )
    monkeypatch.setattr(prepare_dft, "load_source_peft_and_tokenizer", lambda _config: (object(), object()))
    monkeypatch.setattr(
        prepare_dft,
        "frozen_base_embeddings",
        lambda *_args: torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    )
    monkeypatch.setattr(prepare_dft, "git_sha", lambda: "b" * 40)
    manifest = run_prepare_dft(config, "prepare-test")
    artifact_path = Path(manifest["training_bandwidths"]["path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["artifact_schema"] == "dftr.m2.training_bandwidths.v2"
    assert canonical_hash(artifact["producer_config"]) == artifact["producer_config_sha256"]
    assert canonical_hash(artifact["preparation_contract"]) == artifact["preparation_contract_sha256"]
    assert artifact["values"] == pytest.approx([0.125, 0.5, 2.0, 8.0, 32.0])
    assert manifest["token_accounting"] == {"generated_tokens": 0, "total_tokens": 0}
    assert _count_generated_tokens(tmp_path / "state", "prepare-test") == 0
    assert not hasattr(prepare_dft, "generate")
    with pytest.raises(PrepareDFTError, match="empty wrapper checkpoint"):
        run_prepare_dft(config, "prepare-test-two")


def test_prepare_and_training_use_same_adapter_disabled_embedding_symbol():
    assert prepare_dft.frozen_base_embeddings is representation.frozen_base_embeddings
    assert dft.frozen_base_embeddings is representation.frozen_base_embeddings
    assert dft._masked_hidden_embeddings is representation.masked_hidden_embeddings


def test_shared_representation_forces_eval_and_disable_adapter(monkeypatch):
    events = []

    class Context:
        def __enter__(self):
            events.append("disable-enter")

        def __exit__(self, *_args):
            events.append("disable-exit")

    class Model:
        training = True

        def eval(self):
            events.append("eval")
            self.training = False

        def train(self):
            events.append("train")
            self.training = True

        def disable_adapter(self):
            return Context()

    monkeypatch.setattr(
        representation, "masked_hidden_embeddings", lambda *_args: "embeddings"
    )
    assert representation.frozen_base_embeddings(Model(), object(), ["text"], {}) == "embeddings"
    assert events == ["eval", "disable-enter", "disable-exit", "train"]


def test_prepare_source_does_not_import_measurement_or_generate():
    source = (Path(__file__).resolve().parents[1] / "m2" / "prepare_dft.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("measurement_v2" in name or name.startswith("harness") for name in imports)
    assert ".generate(" not in source


def test_runner_dispatches_prepare_bidirectionally(monkeypatch, tmp_path, capsys):
    config = valid_prepare_config()
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "run_prepare_dft", lambda value, run_id: {"status": "completed"})
    assert runner.main(["--config", str(path), "--run-id", "run-1"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    config["workflow"]["step"] = "train_dft"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="requires prepare_dft"):
        runner.main(["--config", str(path), "--run-id", "run-2"])


def _prepare_payload(config: dict) -> dict:
    return {
        "run_id": "dftr-prepare-test",
        "config": config,
        "config_hash": policy_hash(config),
        "git_sha": "b" * 40,
        "budget_class": "smoke",
        "preregistration": {
            "kind": "prereg", "status": "open",
            "comparison": config["run"]["comparison_id"],
        },
    }


def _rehash_prepare_config(config: dict) -> dict:
    config["workflow"]["preparation_contract_sha256"] = canonical_hash(
        preparation_contract_payload(config)
    )
    return config


def test_gateway_accepts_only_exact_prepare_contract():
    config = valid_prepare_config()
    assert validate_launch(_prepare_payload(config)).task_kind == "experiment"
    for mutation in (
        lambda value: value["workflow"].update(protocol_version="substitute"),
        lambda value: value["model"].update(revision="main"),
        lambda value: value["output"].update(overwrite=True),
    ):
        changed = valid_prepare_config()
        mutation(changed)
        payload = _prepare_payload(changed)
        with pytest.raises(PolicyError, match="prepare_dft|immutable"):
            validate_launch(payload)
    payload = _prepare_payload(valid_prepare_config())
    payload["dft_a64_readiness"] = {"status": "ready"}
    with pytest.raises(PolicyError, match="prepare_dft"):
        validate_launch(payload)


@pytest.mark.parametrize(("field", "value"), (("gpu", "T4"), ("gpus", True)))
def test_gateway_rejects_prepare_compute_not_accepted_by_producer(field, value):
    config = valid_prepare_config()
    config["compute"][field] = value
    _rehash_prepare_config(config)
    with pytest.raises(PolicyError, match="prepare_dft"):
        validate_launch(_prepare_payload(config))


@pytest.mark.parametrize(
    ("section", "field"),
    (("initial_adapter", "path"), ("data", "human_targets_path")),
)
def test_gateway_rejects_prepare_checkpoint_traversal(section, field):
    config = valid_prepare_config()
    config[section][field] = "/checkpoints/../tmp/attacker-input"
    _rehash_prepare_config(config)
    with pytest.raises(PolicyError, match="checkpoint volume"):
        validate_launch(_prepare_payload(config))


def test_client_accepts_exact_local_prepare_contract(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(str(root / "infra" / "gpu"))
    client["_validate_submit"].__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg", "status": "open", "comparison": comparison
    }
    monkeypatch.setenv("DFTR_GPU_BACKEND", "local")
    comparison, prereg = client["_validate_submit"](valid_prepare_config(), "smoke")
    assert comparison == "M2-training-bandwidths-v1"
    assert prereg["status"] == "open"


@pytest.mark.parametrize(("field", "value"), (("gpu", "T4"), ("gpus", True)))
def test_client_rejects_prepare_compute_not_accepted_by_producer(monkeypatch, field, value):
    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(str(root / "infra" / "gpu"))
    client["_validate_submit"].__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg", "status": "open", "comparison": comparison
    }
    monkeypatch.setenv("DFTR_GPU_BACKEND", "local")
    config = valid_prepare_config()
    config["compute"][field] = value
    _rehash_prepare_config(config)
    with pytest.raises(SystemExit):
        client["_validate_submit"](config, "smoke")


@pytest.mark.parametrize(
    ("section", "field"),
    (("initial_adapter", "path"), ("data", "human_targets_path")),
)
def test_client_rejects_prepare_checkpoint_traversal(monkeypatch, section, field):
    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(str(root / "infra" / "gpu"))
    client["_validate_submit"].__globals__["_preregistration"] = lambda comparison: {
        "kind": "prereg", "status": "open", "comparison": comparison
    }
    monkeypatch.delenv("DFTR_GPU_BACKEND", raising=False)
    config = valid_prepare_config()
    config[section][field] = "/checkpoints/../tmp/attacker-input"
    _rehash_prepare_config(config)
    with pytest.raises(SystemExit):
        client["_validate_submit"](config, "smoke")
