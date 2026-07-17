import copy
from types import SimpleNamespace

import pytest
import torch

from experiments.m2.estimator_audit import (
    ESTIMATOR_AUDIT_SCHEMA,
    EstimatorAuditError,
    audit_contract_payload,
    accumulate_score_gradient_microbatched,
    canonical_hash,
    cosine,
    count_sketch_gradients,
    derive_training_bandwidths,
    validate_estimator_audit_config,
)
from experiments.m2.sequence_v2 import (
    FULL_BRIEF_SCHEMA_V2,
    FULL_BRIEF_SERIALIZER_V2_SHA256,
)


SHA = "a" * 64


def _config():
    config = {
        "artifact_schema": ESTIMATOR_AUDIT_SCHEMA,
        "run": {
            "comparison_id": "M2-frozen-estimator-audit-v1",
            "arm": "frozen-estimator-audit",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 11,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 120},
        "model": {
            "base": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "torch_dtype": "bfloat16",
        },
        "initial_adapter": {
            "path": "/checkpoints/runs/source/seed-11",
            "adapter_model_sha256": SHA,
            "adapter_config_sha256": SHA,
            "file_manifest_sha256": SHA,
        },
        "data": {
            "rollout_path": "/checkpoints/data/train.jsonl",
            "rollout_sha256": SHA,
            "human_targets_path": "/checkpoints/data/train.jsonl",
            "human_targets_sha256": SHA,
            "human_text_field": "completion",
            "prompt_format": "USER:\n{brief}\nASSISTANT:",
            "prompt_schema_version": FULL_BRIEF_SCHEMA_V2,
            "prompt_serializer_sha256": FULL_BRIEF_SERIALIZER_V2_SHA256,
            "legacy_target_length_semantics": "provider_requested_token_estimate_missing_unit_field",
        },
        "representation": {
            "model": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "layer": -1,
            "pooling": "attention_masked_mean",
            "normalize": True,
            "role": "diagnostic_training_only_not_evaluation",
            "batch_size": 4,
            "max_tokens": 256,
        },
        "audit": {
            "replicates": 16,
            "group_sizes": [4, 8, 16, 32],
            "max_new_tokens": 64,
            "rollout_target_length_tokens": 64,
            "max_input_tokens": 1024,
            "logprob_microbatch_size": 1,
            "prompt_schedule_seed": 3101,
            "rollout_seed_start": 4101,
            "gradient_supports": ["full_humans", "rollout_horizon_humans"],
            "sketch_dimension": 256,
            "sketch_seed": 5101,
            "bandwidth_scales": [0.25, 0.5, 1.0, 2.0, 4.0],
            "go_thresholds": {
                "k32_split_half_cosine_min": 0.5,
                "k32_gradient_norm_cv_max": 1.0,
            },
        },
        "runtime": {
            "torch_version": "2.13.0+cu130",
            "transformers_version": "4.57.6",
            "peft_version": "0.19.1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "output": {"filename": "estimator_audit.json", "overwrite": False},
        "workflow": {
            "protocol_version": ESTIMATOR_AUDIT_SCHEMA,
            "step": "audit_estimator",
            "audit_contract_sha256": "",
        },
    }
    config["workflow"]["audit_contract_sha256"] = canonical_hash(
        audit_contract_payload(config)
    )
    return config


def test_estimator_audit_config_is_hash_bound_and_exact():
    config = _config()
    assert validate_estimator_audit_config(config) is config
    mutated = copy.deepcopy(config)
    mutated["audit"]["replicates"] = 8
    with pytest.raises(EstimatorAuditError, match="audit contract"):
        validate_estimator_audit_config(mutated)


def test_bandwidths_use_positive_pairwise_squared_distance_median():
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    values = derive_training_bandwidths(embeddings, [0.5, 1.0, 2.0])
    assert values == [1.0, 4.0, 16.0]


def test_count_sketch_is_deterministic_and_tracks_exact_gradient_norm():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    second = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
    (first.sum() + 2.0 * second.sum()).backward()
    sketch_a, norm_a = count_sketch_gradients(
        [("second", second), ("first", first)], dimension=16, seed=7
    )
    sketch_b, norm_b = count_sketch_gradients(
        [("first", first), ("second", second)], dimension=16, seed=7
    )
    assert torch.equal(sketch_a, sketch_b)
    assert norm_a == pytest.approx(11.0 ** 0.5)
    assert norm_b == pytest.approx(norm_a)
    assert cosine(sketch_a, sketch_b) == pytest.approx(1.0)


def test_single_sequence_accumulation_matches_full_group_gradient():
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(9, 4)
            self.projection = torch.nn.Linear(4, 9, bias=False)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask
            return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

    torch.manual_seed(13)
    full = TinyPolicy()
    micro = copy.deepcopy(full)
    sequences = torch.tensor(
        [[0, 2, 4, 5, 1], [0, 3, 6, 1, 0], [0, 4, 2, 7, 1], [0, 5, 8, 1, 0]]
    )
    prompt_mask = torch.tensor([[0, 1]] * 4)
    action_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]])
    advantages = torch.tensor([0.4, -0.2, 0.1, -0.3])
    full_logs, full_loss = accumulate_score_gradient_microbatched(
        full,
        sequences,
        prompt_mask,
        2,
        action_mask,
        advantages,
        microbatch_size=4,
    )
    micro_logs, micro_loss = accumulate_score_gradient_microbatched(
        micro,
        sequences,
        prompt_mask,
        2,
        action_mask,
        advantages,
        microbatch_size=1,
    )
    assert micro_logs == pytest.approx(full_logs)
    assert micro_loss == pytest.approx(full_loss)
    for full_parameter, micro_parameter in zip(full.parameters(), micro.parameters()):
        assert torch.allclose(full_parameter.grad, micro_parameter.grad, atol=1e-7)
