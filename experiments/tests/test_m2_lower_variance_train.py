from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from experiments import runner
from experiments.m1.contracts import file_sha256
from experiments.m2 import lower_variance_train as train
from experiments.m2.lower_variance_train import (
    GENERATION_CONTRACT,
    LOWER_VARIANCE_SCHEMA,
    LOWER_VARIANCE_STEP,
    LowerVarianceTrainError,
    component_gradient_norm,
    deterministic_epoch_batches,
    eos_aware_completion_ids,
    matched_exposure_payload,
    method_contract_payload,
    objective_components,
    prepare_supervised_batch,
    run_lower_variance,
    validate_lower_variance_config,
)
from experiments.m2.representation import canonical_hash


SHA = "a" * 64


def valid_config() -> dict:
    config = {
        "artifact_schema": LOWER_VARIANCE_SCHEMA,
        "run": {
            "comparison_id": "M2-lower-variance-three-arm-v1",
            "arm": "SFT-vs-TOKEN_MOMENT-vs-MMD_WITNESS",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 11,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 120},
        "model": {
            "base": train.BASE_MODEL,
            "revision": train.BASE_REVISION,
            "torch_dtype": "bfloat16",
        },
        "initial_adapter": {
            "path": "/checkpoints/runs/source/seed-11",
            "adapter_model_sha256": SHA,
            "adapter_config_sha256": SHA,
            "file_manifest_sha256": SHA,
        },
        "data": {
            "anchor_path": "/checkpoints/data/lower-variance-anchors.jsonl",
            "anchor_sha256": SHA,
            "witness_generated_path": "/checkpoints/data/shared-rollout.jsonl",
            "witness_generated_sha256": SHA,
            "witness_generation_contract_sha256": canonical_hash(GENERATION_CONTRACT),
            "completion_field": "completion",
            "generated_text_field": "generated_completion",
            "prompt_format": "USER:\n{brief}\nASSISTANT:",
            "prompt_schema_version": train.FULL_BRIEF_SCHEMA,
            "prompt_serializer_sha256": train.FULL_BRIEF_SERIALIZER_SHA256,
        },
        "representation": {
            "model": train.BASE_MODEL,
            "revision": train.BASE_REVISION,
            "layer": -1,
            "pooling": "attention_masked_mean",
            "normalize": True,
            "role": "lower_variance_training_only_not_measurement_v3",
            "batch_size": 4,
            "max_tokens": 256,
        },
        "objectives": {
            "token_moments": {
                "coefficient": 0.2,
                "first_moment_weight": 1.0,
                "second_moment_weight": 0.5,
                "frequent_token_ids": [0, 2, 4],
                "vocabulary_source_sha256": SHA,
            },
            "mmd_witness": {
                "bandwidths": [0.5, 1.0, 2.0],
                "temperature": 0.7,
                "weighting": "softmax_mean_one.v1",
                "human_self_kernel": "leave_one_out",
            },
        },
        "generation": copy.deepcopy(GENERATION_CONTRACT),
        "runtime": {
            "torch_version": "2.9.1",
            "transformers_version": "4.57.6",
            "peft_version": "0.18.0",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "training": {
            "steps": 4,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "max_input_tokens": 128,
            "checkpoint_every": 2,
            "schedule": "python_random_sample_without_replacement.v1",
        },
        "arms": [
            {"id": "SFT", "sft_weighting": "uniform", "token_moment_coefficient": 0.0},
            {
                "id": "TOKEN_MOMENT",
                "sft_weighting": "uniform",
                "token_moment_coefficient": 0.2,
            },
            {
                "id": "MMD_WITNESS",
                "sft_weighting": "mmd_witness",
                "token_moment_coefficient": 0.0,
            },
        ],
        "resume": {"SFT": None, "TOKEN_MOMENT": None, "MMD_WITNESS": None},
        "execution": {"arm": "SFT"},
        "workflow": {
            "protocol_version": LOWER_VARIANCE_SCHEMA,
            "step": LOWER_VARIANCE_STEP,
            "method_contract_sha256": "0" * 64,
        },
    }
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )
    return config


def _rehash(config: dict) -> None:
    config["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(config)
    )


def record(completion: str = "human completion") -> dict:
    return {
        "user_prompt": "Write a short article",
        "use_case": "web",
        "style_kind": "plain",
        "style": "clear",
        "detail_mode": "concise",
        "target_length": 80,
        "target_length_unit": "tokens",
        "em_dashes_allowed": False,
        "outline": ["fact"],
        "completion": completion,
    }


def test_config_is_exact_hash_bound_and_freezes_all_three_arms():
    config = valid_config()
    assert validate_lower_variance_config(config) is config
    exposure = matched_exposure_payload(config)
    assert len(set(canonical_hash(value) for value in exposure["arms"].values())) == 1
    assert exposure["initial_adapter"] == config["initial_adapter"]
    assert exposure["generation"]["stop_on_eos"] is True

    for mutate, message in (
        (lambda value: value["generation"].update(stop_on_eos=False), "EOS-aware"),
        (lambda value: value["training"].update(checkpoint_every=3), "checkpoint"),
        (lambda value: value["arms"][2].update(sft_weighting="uniform"), "three-arm"),
        (lambda value: value["data"].update(anchor_path="/measurement_v3/leak.jsonl"), "training-only"),
        (lambda value: value["generation"].update(top_k=False), "generation"),
        (
            lambda value: value["arms"][0].update(token_moment_coefficient=False),
            "token_moment_coefficient",
        ),
    ):
        changed = valid_config()
        mutate(changed)
        _rehash(changed)
        with pytest.raises(LowerVarianceTrainError, match=message):
            validate_lower_variance_config(changed)
    changed = valid_config()
    changed["objectives"]["token_moments"]["coefficient"] = 0.3
    changed["arms"][1]["token_moment_coefficient"] = 0.3
    with pytest.raises(LowerVarianceTrainError, match="hash"):
        validate_lower_variance_config(changed)


def test_eos_contract_stops_at_first_eos_or_reserves_terminal_eos():
    assert eos_aware_completion_ids([4, 9, 5, 6], eos_token_id=9, max_new_tokens=4) == [4, 9]
    assert eos_aware_completion_ids([4, 5, 6, 7], eos_token_id=9, max_new_tokens=3) == [4, 5, 9]
    assert eos_aware_completion_ids([], eos_token_id=9, max_new_tokens=1) == [9]


def test_epoch_schedule_exposes_every_record_exactly_once_per_epoch():
    batches = deterministic_epoch_batches(size=12, batch_size=3, steps=8, seed=11)
    first_epoch = [index for batch in batches[:4] for index in batch]
    second_epoch = [index for batch in batches[4:] for index in batch]
    assert sorted(first_epoch) == list(range(12))
    assert sorted(second_epoch) == list(range(12))
    assert first_epoch != second_epoch
    assert batches == deterministic_epoch_batches(12, 3, 8, 11)


def test_epoch_schedule_rejects_partial_epochs_and_ragged_batches():
    with pytest.raises(LowerVarianceTrainError, match="complete equal-sized epochs"):
        deterministic_epoch_batches(size=10, batch_size=3, steps=4, seed=11)
    with pytest.raises(LowerVarianceTrainError, match="complete equal-sized epochs"):
        deterministic_epoch_batches(size=12, batch_size=3, steps=5, seed=11)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, *, add_special_tokens, truncation, **_kwargs):
        if add_special_tokens:
            return {"input_ids": [[1, 2] for _ in texts]}
        rows = []
        for text in texts:
            rows.append([4, 9, 5] if text == "has eos" else [6, 7])
        return {"input_ids": rows}

    def pad(self, features, *, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        width = max(len(row["input_ids"]) for row in features)
        ids, masks = [], []
        for row in features:
            pad = width - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            masks.append(row["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_supervised_batch_includes_one_eos_and_masks_prompt_and_post_eos():
    tokenizer = FakeTokenizer()
    batch = prepare_supervised_batch(
        tokenizer, [record("has eos"), record("without eos")], valid_config()
    )
    assert tokenizer.padding_side == "left"
    assert batch["labels"][0].tolist() == [-100, -100, 4, 9, -100]
    assert batch["labels"][1].tolist() == [-100, -100, 6, 7, 9]
    assert batch["completion_token_counts"].tolist() == [2, 3]


def test_lower_variance_prompt_preserves_token_length_semantics():
    rendered = train._render_lower_variance_prompt(record(), valid_config())
    assert "Target length: about 80 tokens" in rendered
    assert "80 words" not in rendered
    wrong = record()
    wrong["target_length_unit"] = "words"
    with pytest.raises(LowerVarianceTrainError, match="must be tokens"):
        train._render_lower_variance_prompt(wrong, valid_config())


def test_three_objectives_have_expected_decomposition_and_finite_component_gradients():
    config = valid_config()
    labels = torch.tensor(
        [[-100, 0, 1, 2], [-100, 2, 3, 4]], dtype=torch.long
    )
    weights = torch.tensor([0.4, 1.6])
    for arm in config["arms"]:
        logits = torch.randn((2, 4, 6), requires_grad=True)
        components = objective_components(
            logits,
            labels,
            config["objectives"]["token_moments"]["frequent_token_ids"],
            weights,
            arm,
            config["objectives"],
        )
        norms = {
            name: component_gradient_norm(components[name], [logits])
            for name in (
                "uniform_sft",
                "token_moment_component",
                "witness_delta_component",
                "total",
            )
        }
        assert all(torch.isfinite(torch.tensor(value)) for value in norms.values())
        assert norms["total"] > 0
        if arm["id"] == "SFT":
            assert torch.allclose(components["total"], components["uniform_sft"])
            assert norms["token_moment_component"] == 0
            assert norms["witness_delta_component"] == 0
        elif arm["id"] == "TOKEN_MOMENT":
            assert norms["token_moment_component"] > 0
            assert norms["witness_delta_component"] == 0
        else:
            assert torch.allclose(components["total"], components["weighted_sft"])
            assert norms["witness_delta_component"] > 0
            assert norms["token_moment_component"] == 0


def test_witness_artifact_is_frozen_from_shared_generated_and_human_texts(monkeypatch):
    calls = []

    def fake_embeddings(_model, _tokenizer, texts, _config):
        calls.append(list(texts))
        if texts[0].startswith("human"):
            return torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        return torch.tensor([[2.0, 0.0], [3.0, 0.0]])

    monkeypatch.setattr(train, "frozen_base_embeddings", fake_embeddings)
    config = valid_config()
    anchors = [record("human one"), record("human two")]
    generated = [
        {"generated_completion": "generated one"},
        {"generated_completion": "generated two"},
    ]
    weights, artifact = train._witness_artifact(object(), object(), anchors, generated, config)
    assert calls == [["human one", "human two"], ["generated one", "generated two"]]
    assert weights.requires_grad is False
    assert weights.mean() == pytest.approx(1.0)
    assert artifact["anchor_sha256"] == config["data"]["anchor_sha256"]
    assert artifact["weighting"] == "softmax_mean_one.v1"


class FakePolicy:
    def save_pretrained(self, target, **_kwargs):
        target = Path(target)
        (target / "adapter_model.safetensors").write_bytes(b"adapter")
        (target / "adapter_config.json").write_text("{}", encoding="utf-8")


class FakeOptimizer:
    def state_dict(self):
        return {"state": {}, "param_groups": []}


def test_checkpoint_resume_binds_config_schedule_witness_and_exact_boundary(tmp_path):
    config = valid_config()
    target = tmp_path / "step-2"
    schedule_sha = "b" * 64
    witness_sha = "c" * 64
    logs = [{"step": 0}, {"step": 1}]
    train._save_training_checkpoint(
        FakePolicy(),
        FakeOptimizer(),
        target,
        "SFT",
        2,
        logs,
        4,
        11,
        schedule_sha,
        witness_sha,
        config,
    )
    descriptor = {
        "path": str(target),
        "adapter_model_sha256": file_sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(target / "adapter_config.json"),
        "training_state_sha256": file_sha256(target / "training_state.pt"),
        "file_manifest_sha256": canonical_hash(
            train._directory_file_map(target, "test resume")
        ),
        "source_config_sha256": canonical_hash(config),
    }
    state = train._verify_resume_artifact(
        descriptor, "SFT", config, schedule_sha, witness_sha
    )
    assert state["next_step"] == 2
    changed = dict(descriptor, source_config_sha256="d" * 64)
    with pytest.raises(LowerVarianceTrainError, match="provenance"):
        train._verify_resume_artifact(changed, "SFT", config, schedule_sha, witness_sha)

    state["next_step"] = 1
    state["logs"] = state["logs"][:1]
    state["optimizer_examples"] = 2
    torch.save(state, target / "training_state.pt")
    descriptor["training_state_sha256"] = file_sha256(target / "training_state.pt")
    descriptor["file_manifest_sha256"] = canonical_hash(
        train._directory_file_map(target, "test resume boundary")
    )
    with pytest.raises(LowerVarianceTrainError, match="provenance"):
        train._verify_resume_artifact(
            descriptor, "SFT", config, schedule_sha, witness_sha
        )


def test_run_manifest_records_exact_config_hash_and_zero_generated_tokens(
    monkeypatch, tmp_path
):
    config = valid_config()
    anchors = [record("human one"), record("human two")]
    generated = [{"generated_completion": "generated"}]
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    monkeypatch.setattr(train, "_verify_inputs", lambda _config: (anchors, generated))
    monkeypatch.setattr(
        train, "build_run_paths", lambda _config, _run_id: (output_dir, checkpoint_dir)
    )
    monkeypatch.setattr(train, "git_sha", lambda: "e" * 40)
    monkeypatch.setattr(
        train,
        "_run_arm",
        lambda *_args: {
            "optimizer_examples": 8,
            "teacher_forced_completion_tokens": 24,
            "generated_tokens": 0,
        },
    )
    result = run_lower_variance(config, "lower-variance-test")
    assert result["config_sha256"] == canonical_hash(config)
    assert result["matched_exposure_contract_sha256"] == canonical_hash(
        result["matched_exposure_contract"]
    )
    assert result["token_accounting"] == {
        "generated_tokens": 0,
        "optimizer_examples": 8,
        "teacher_forced_completion_tokens": 24,
    }
    assert json.loads((checkpoint_dir / "run_manifest.json").read_text()) == result


def test_runner_dispatch_is_bidirectional(monkeypatch, tmp_path, capsys):
    config = valid_config()
    path = tmp_path / "lower-variance.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        runner, "run_lower_variance", lambda _config, _run_id: {"status": "completed"}
    )
    assert runner.main(["--config", str(path), "--run-id", "run-1"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    config["workflow"]["step"] = "substitute"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="requires train_lower_variance"):
        runner.main(["--config", str(path), "--run-id", "run-2"])

    config = valid_config()
    config["workflow"]["protocol_version"] = "substitute"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="requires the frozen three-arm"):
        runner.main(["--config", str(path), "--run-id", "run-3"])
