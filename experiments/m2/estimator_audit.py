"""Frozen-policy group-size and support audit for score-function MMD.

This is diagnostic-only: it never takes an optimizer step or writes an
adapter.  Nested rollout groups make K=4/8/16/32 comparable under the same
policy, prompt schedule, and random streams.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

from experiments.m1.contracts import (
    build_run_paths,
    file_sha256,
    git_sha,
    write_json,
)
from experiments.m2.dft import (
    BASE_MODEL,
    BASE_REVISION,
    _activate_adapter,
    _directory_file_map,
    _load_jsonl,
    mmd_leave_one_out_baselines,
    mmd_score_components,
    score_function_loss,
)
from experiments.m2.representation import (
    TRAINING_BANDWIDTH_SCALES,
    frozen_base_embeddings,
    load_source_peft_and_tokenizer,
)
from experiments.m2.sequence_v2 import (
    FULL_BRIEF_SCHEMA_V2,
    FULL_BRIEF_SERIALIZER_V2_SHA256,
    materialize_sampled_batch,
    normalize_legacy_brief_as_tokens,
    render_full_brief_v2,
    sample_raw_policy_eos_aware,
    sequence_log_probs_eos_aware,
)


ESTIMATOR_AUDIT_SCHEMA = "dftr.m2.frozen_estimator_audit.v1"
ESTIMATOR_AUDIT_STEP = "audit_estimator"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SUPPORTS = ("full_humans", "rollout_horizon_humans")
TOP_LEVEL_KEYS = {
    "artifact_schema",
    "run",
    "compute",
    "model",
    "initial_adapter",
    "data",
    "representation",
    "audit",
    "runtime",
    "output",
    "workflow",
}


class EstimatorAuditError(ValueError):
    pass


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def audit_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "artifact_schema",
            "run",
            "compute",
            "model",
            "initial_adapter",
            "data",
            "representation",
            "audit",
            "runtime",
            "output",
        )
    } | {
        "protocol_version": (config.get("workflow") or {}).get("protocol_version"),
        "step": (config.get("workflow") or {}).get("step"),
    }


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EstimatorAuditError(f"{field} must contain exactly {sorted(keys)}")
    return value


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise EstimatorAuditError(f"{field} must be a lowercase SHA-256")
    return text


def validate_estimator_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    _exact(config, TOP_LEVEL_KEYS, "config")
    if config.get("artifact_schema") != ESTIMATOR_AUDIT_SCHEMA:
        raise EstimatorAuditError("unexpected estimator-audit schema")
    run = _exact(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") != "frozen-estimator-audit"
        or run.get("budget_class") != "screen"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 11
    ):
        raise EstimatorAuditError("run contract is invalid")
    compute = _exact(config.get("compute"), {"gpu", "gpus", "timeout_min"}, "compute")
    if (
        str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1
        or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int)
        or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= 120
    ):
        raise EstimatorAuditError("compute contract is invalid")
    if config.get("model") != {
        "base": BASE_MODEL,
        "revision": BASE_REVISION,
        "torch_dtype": "bfloat16",
    }:
        raise EstimatorAuditError("model contract is invalid")
    adapter = _exact(
        config.get("initial_adapter"),
        {
            "path",
            "adapter_model_sha256",
            "adapter_config_sha256",
            "file_manifest_sha256",
        },
        "initial_adapter",
    )
    if not Path(str(adapter.get("path") or "")).is_absolute():
        raise EstimatorAuditError("initial adapter path must be absolute")
    for field in ("adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"):
        _sha(adapter.get(field), f"initial_adapter.{field}")
    data = _exact(
        config.get("data"),
        {
            "rollout_path",
            "rollout_sha256",
            "human_targets_path",
            "human_targets_sha256",
            "human_text_field",
            "prompt_format",
            "prompt_schema_version",
            "prompt_serializer_sha256",
            "legacy_target_length_semantics",
        },
        "data",
    )
    for path_field in ("rollout_path", "human_targets_path"):
        path = Path(str(data.get(path_field) or ""))
        if not path.is_absolute() or any(part in {"harness", "measurement_v2"} for part in path.parts):
            raise EstimatorAuditError(f"data.{path_field} must be training-only and absolute")
    for field in ("rollout_sha256", "human_targets_sha256"):
        _sha(data.get(field), f"data.{field}")
    if (
        data.get("human_text_field") != "completion"
        or data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:"
        or data.get("prompt_schema_version") != FULL_BRIEF_SCHEMA_V2
        or data.get("prompt_serializer_sha256") != FULL_BRIEF_SERIALIZER_V2_SHA256
        or data.get("legacy_target_length_semantics")
        != "provider_requested_token_estimate_missing_unit_field"
    ):
        raise EstimatorAuditError("v2 prompt/data contract is invalid")
    representation = _exact(
        config.get("representation"),
        {"model", "revision", "layer", "pooling", "normalize", "role", "batch_size", "max_tokens"},
        "representation",
    )
    if (
        representation.get("model") != BASE_MODEL
        or representation.get("revision") != BASE_REVISION
        or representation.get("layer") != -1
        or representation.get("pooling") != "attention_masked_mean"
        or representation.get("normalize") is not True
        or representation.get("role") != "diagnostic_training_only_not_evaluation"
        or any(
            not isinstance(representation.get(field), int)
            or isinstance(representation.get(field), bool)
            or representation[field] <= 0
            for field in ("batch_size", "max_tokens")
        )
    ):
        raise EstimatorAuditError("representation contract is invalid")
    audit = _exact(
        config.get("audit"),
        {
            "replicates",
            "group_sizes",
            "max_new_tokens",
            "rollout_target_length_tokens",
            "max_input_tokens",
            "logprob_microbatch_size",
            "prompt_schedule_seed",
            "rollout_seed_start",
            "gradient_supports",
            "sketch_dimension",
            "sketch_seed",
            "bandwidth_scales",
            "go_thresholds",
        },
        "audit",
    )
    thresholds = _exact(
        audit.get("go_thresholds"),
        {"k32_split_half_cosine_min", "k32_gradient_norm_cv_max"},
        "audit.go_thresholds",
    )
    if (
        audit.get("replicates") != 16
        or audit.get("group_sizes") != [4, 8, 16, 32]
        or audit.get("max_new_tokens") != 64
        or audit.get("rollout_target_length_tokens") != 64
        or not isinstance(audit.get("max_input_tokens"), int)
        or audit["max_input_tokens"] <= 0
        or audit.get("logprob_microbatch_size") != 1
        or not isinstance(audit.get("prompt_schedule_seed"), int)
        or not isinstance(audit.get("rollout_seed_start"), int)
        or audit.get("gradient_supports") != list(SUPPORTS)
        or audit.get("sketch_dimension") != 256
        or not isinstance(audit.get("sketch_seed"), int)
        or audit.get("bandwidth_scales") != TRAINING_BANDWIDTH_SCALES
        or thresholds != {
            "k32_split_half_cosine_min": 0.5,
            "k32_gradient_norm_cv_max": 1.0,
        }
    ):
        raise EstimatorAuditError("audit contract is invalid")
    runtime = _exact(
        config.get("runtime"),
        {
            "torch_version",
            "transformers_version",
            "peft_version",
            "deterministic_algorithms",
            "cublas_workspace_config",
        },
        "runtime",
    )
    if (
        any(not str(runtime.get(field) or "") for field in ("torch_version", "transformers_version", "peft_version"))
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
    ):
        raise EstimatorAuditError("runtime contract is invalid")
    if config.get("output") != {"filename": "estimator_audit.json", "overwrite": False}:
        raise EstimatorAuditError("output contract is invalid")
    workflow = _exact(
        config.get("workflow"),
        {"protocol_version", "step", "audit_contract_sha256"},
        "workflow",
    )
    if (
        workflow.get("protocol_version") != ESTIMATOR_AUDIT_SCHEMA
        or workflow.get("step") != ESTIMATOR_AUDIT_STEP
        or canonical_hash(audit_contract_payload(config))
        != _sha(workflow.get("audit_contract_sha256"), "workflow.audit_contract_sha256")
    ):
        raise EstimatorAuditError("workflow contract hash mismatch")
    return config


def derive_training_bandwidths(embeddings: Any, scales: Sequence[float]) -> list[float]:
    import torch

    values = embeddings.detach().float().cpu()
    if values.ndim != 2 or len(values) < 2:
        raise EstimatorAuditError("bandwidth derivation requires an embedding matrix")
    distances = torch.pdist(values, p=2).pow(2).double()
    positive = distances[distances > 0]
    if len(positive) != len(distances):
        raise EstimatorAuditError("bandwidth derivation found zero-distance human pairs")
    median = float(positive.median().item())
    if not math.isfinite(median) or median <= 0:
        raise EstimatorAuditError("bandwidth median must be positive")
    return [median * float(scale) ** 2 for scale in scales]


def mmd_unbiased_value(generated: Any, humans: Any, bandwidths: Sequence[float]) -> float:
    import torch

    def kernel(left: Any, right: Any) -> Any:
        squared = torch.cdist(left, right).pow(2)
        return torch.stack(
            [torch.exp(-squared / (2.0 * float(value))) for value in bandwidths]
        ).mean(0)

    xx = kernel(generated, generated)
    yy = kernel(humans, humans)
    xy = kernel(generated, humans)
    n, m = len(generated), len(humans)
    return float(
        (
            (xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
            + (yy.sum() - yy.diagonal().sum()) / (m * (m - 1))
            - 2.0 * xy.mean()
        ).item()
    )


def count_sketch_gradients(
    named_parameters: Iterable[tuple[str, Any]], *, dimension: int, seed: int
) -> tuple[Any, float]:
    """Deterministic CountSketch of all finite trainable gradients."""
    import torch

    sketch = None
    squared_norm = 0.0
    offset = 0
    seen = 0
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            offset += parameter.numel()
            continue
        flat = gradient.detach().float().reshape(-1)
        if not torch.all(torch.isfinite(flat)):
            raise EstimatorAuditError(f"non-finite gradient in {name}")
        if sketch is None:
            sketch = torch.zeros(dimension, dtype=torch.float64, device=flat.device)
        indices = torch.arange(flat.numel(), dtype=torch.int64, device=flat.device) + offset
        hashed = (indices * 1103515245 + int(seed)) & 0x7FFFFFFF
        buckets = hashed.remainder(dimension)
        signs = ((hashed // dimension).remainder(2) * 2 - 1).to(flat.dtype)
        sketch.scatter_add_(0, buckets, (flat * signs).double())
        squared_norm += float(flat.double().square().sum().item())
        offset += parameter.numel()
        seen += flat.numel()
    if sketch is None or seen == 0:
        raise EstimatorAuditError("no trainable gradients were available for sketching")
    return sketch.cpu(), math.sqrt(squared_norm)


def cosine(left: Any, right: Any) -> float:
    import torch

    denominator = float(left.norm().item() * right.norm().item())
    return 0.0 if denominator == 0 else float(torch.dot(left, right).item() / denominator)


def accumulate_score_gradient_microbatched(
    model: Any,
    sequences: Any,
    prompt_attention_mask: Any,
    prompt_width: int,
    action_mask: Any,
    advantages: Any,
    *,
    microbatch_size: int,
) -> tuple[list[float], float]:
    """Accumulate the exact mean REINFORCE gradient without a K-way graph."""
    import torch

    group_size = int(sequences.shape[0])
    if microbatch_size <= 0 or group_size != len(advantages):
        raise EstimatorAuditError("invalid score-gradient microbatch dimensions")
    detached_log_probs: list[float] = []
    detached_loss = 0.0
    for start in range(0, group_size, microbatch_size):
        stop = min(group_size, start + microbatch_size)
        micro_log_probs = sequence_log_probs_eos_aware(
            model,
            sequences[start:stop],
            prompt_attention_mask[start:stop],
            prompt_width,
            action_mask[start:stop],
        )
        micro_loss = -(
            advantages[start:stop].detach() * micro_log_probs
        ).sum() / group_size
        if not torch.isfinite(micro_loss):
            raise EstimatorAuditError("non-finite microbatched score loss")
        micro_loss.backward()
        detached_log_probs.extend(
            float(value) for value in micro_log_probs.detach().cpu()
        )
        detached_loss += float(micro_loss.detach().item())
    return detached_log_probs, detached_loss


def _truncate_texts_to_horizon(tokenizer: Any, texts: list[str], horizon: int) -> list[str]:
    token_rows = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=horizon,
    )["input_ids"]
    return [
        tokenizer.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for row in token_rows
    ]


def _verify_inputs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    import peft
    import torch
    import transformers

    expected_runtime = {
        key: config["runtime"][key]
        for key in ("torch_version", "transformers_version", "peft_version")
    }
    observed_runtime = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
    }
    if observed_runtime != expected_runtime:
        raise EstimatorAuditError(
            f"runtime version mismatch: expected {expected_runtime} observed {observed_runtime}"
        )
    adapter = config["initial_adapter"]
    adapter_path = Path(adapter["path"])
    for filename, field in (
        ("adapter_model.safetensors", "adapter_model_sha256"),
        ("adapter_config.json", "adapter_config_sha256"),
    ):
        path = adapter_path / filename
        if not path.is_file() or file_sha256(path) != adapter[field]:
            raise EstimatorAuditError(f"initial adapter {filename} hash mismatch")
    if canonical_hash(_directory_file_map(adapter_path, "initial adapter")) != adapter["file_manifest_sha256"]:
        raise EstimatorAuditError("initial adapter file manifest mismatch")
    data = config["data"]
    rollout_path = Path(data["rollout_path"])
    human_path = Path(data["human_targets_path"])
    for path, expected, label in (
        (rollout_path, data["rollout_sha256"], "rollout"),
        (human_path, data["human_targets_sha256"], "human target"),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise EstimatorAuditError(f"{label} hash mismatch")
    rollout_rows = _load_jsonl(rollout_path, "rollout data")
    human_rows = _load_jsonl(human_path, "human targets")
    human_texts = [str(row.get(data["human_text_field"]) or "") for row in human_rows]
    if any(not text.strip() for text in human_texts):
        raise EstimatorAuditError("human target text is empty")
    if len(rollout_rows) < 32 or len(human_texts) < 32:
        raise EstimatorAuditError("audit requires at least 32 rollout and human rows")
    return rollout_rows, human_texts


def _summarize(rows: list[dict[str, Any]], group_sizes: list[int]) -> dict[str, Any]:
    import numpy as np
    import torch

    summary: dict[str, Any] = {}
    for support in SUPPORTS:
        summary[support] = {}
        for group_size in group_sizes:
            selected = [
                row for row in rows
                if row["support"] == support and row["group_size"] == group_size
            ]
            sketches = [torch.tensor(row["gradient_sketch"], dtype=torch.float64) for row in selected]
            first = torch.stack(sketches[: len(sketches) // 2]).mean(0)
            second = torch.stack(sketches[len(sketches) // 2 :]).mean(0)
            norms = np.asarray([row["gradient_norm"] for row in selected], dtype=float)
            summary[support][str(group_size)] = {
                "split_half_cosine": cosine(first, second),
                "gradient_norm_mean": float(norms.mean()),
                "gradient_norm_cv": float(norms.std(ddof=1) / norms.mean()) if norms.mean() else math.inf,
                "mmd2_mean": float(np.mean([row["mmd2"] for row in selected])),
                "advantage_std_mean": float(np.mean([row["advantage_std"] for row in selected])),
                "mean_active_tokens": float(np.mean([row["mean_active_tokens"] for row in selected])),
            }
    return summary


def run_estimator_audit(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import peft
    import torch
    import transformers

    validate_estimator_audit_config(config)
    rollout_rows, full_human_texts = _verify_inputs(config)
    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    output_path = checkpoint_dir / config["output"]["filename"]
    if output_path.exists():
        raise EstimatorAuditError("estimator audit output already exists")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["runtime"]["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(int(config["run"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["run"]["seed"]))
    policy, tokenizer = load_source_peft_and_tokenizer(config)
    trainable = _activate_adapter(policy, "default", trainable=True)
    policy.config.use_cache = False
    policy.eval()
    device = next(policy.parameters()).device
    audit = config["audit"]
    horizon = int(audit["max_new_tokens"])
    human_texts = {
        "full_humans": full_human_texts,
        "rollout_horizon_humans": _truncate_texts_to_horizon(
            tokenizer, full_human_texts, horizon
        ),
    }
    human_embeddings = {
        support: frozen_base_embeddings(policy, tokenizer, texts, config)
        for support, texts in human_texts.items()
    }
    bandwidths = {
        support: derive_training_bandwidths(
            embeddings, audit["bandwidth_scales"]
        )
        for support, embeddings in human_embeddings.items()
    }
    scheduler = random.Random(int(audit["prompt_schedule_seed"]))
    logs: list[dict[str, Any]] = []
    max_group = max(audit["group_sizes"])
    prompt_format = config["data"]["prompt_format"]
    for replicate in range(int(audit["replicates"])):
        indices = scheduler.sample(range(len(rollout_rows)), max_group)
        batch_rows = [normalize_legacy_brief_as_tokens(rollout_rows[index]) for index in indices]
        for row in batch_rows:
            row["target_length"] = int(audit["rollout_target_length_tokens"])
        prompts = [prompt_format.format(brief=render_full_brief_v2(row)) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=int(audit["max_input_tokens"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        rollout_seed = int(audit["rollout_seed_start"]) + replicate
        devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(rollout_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rollout_seed)
            sampled = sample_raw_policy_eos_aware(
                policy,
                encoded["input_ids"],
                encoded["attention_mask"],
                max_new_tokens=horizon,
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
            )
        sampled = materialize_sampled_batch(sampled)
        prompt_width = int(encoded["input_ids"].shape[1])
        continuation = sampled.sequences[:, prompt_width:]
        texts = tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_embeddings = frozen_base_embeddings(policy, tokenizer, texts, config)
        for support in SUPPORTS:
            humans = human_embeddings[support]
            support_bandwidths = bandwidths[support]
            for group_size in audit["group_sizes"]:
                subset_embeddings = generated_embeddings[:group_size]
                rewards = mmd_score_components(
                    subset_embeddings, humans, support_bandwidths
                )["reward"]
                baselines = mmd_leave_one_out_baselines(
                    subset_embeddings, humans, support_bandwidths
                )
                advantages = rewards - baselines
                policy.zero_grad(set_to_none=True)
                # A full K-way differentiable causal-LM forward retains every
                # layer activation and materializes K large vocabulary logits.
                # Accumulate the exact mean score-function gradient one sample
                # at a time, matching the reviewed training topology.
                detached_log_probs, detached_loss = accumulate_score_gradient_microbatched(
                    policy,
                    sampled.sequences[:group_size],
                    encoded["attention_mask"][:group_size],
                    prompt_width,
                    sampled.action_mask[:group_size],
                    advantages,
                    microbatch_size=int(audit["logprob_microbatch_size"]),
                )
                sketch, gradient_norm = count_sketch_gradients(
                    policy.named_parameters(),
                    dimension=int(audit["sketch_dimension"]),
                    seed=int(audit["sketch_seed"]),
                )
                logs.append(
                    {
                        "replicate": replicate,
                        "rollout_seed": rollout_seed,
                        "support": support,
                        "group_size": group_size,
                        "prompt_indices": indices[:group_size],
                        "mmd2": mmd_unbiased_value(
                            subset_embeddings, humans, support_bandwidths
                        ),
                        "reward_mean": float(rewards.mean().item()),
                        "reward_std": float(rewards.std(unbiased=True).item()),
                        "advantage_mean": float(advantages.mean().item()),
                        "advantage_std": float(advantages.std(unbiased=True).item()),
                        "surrogate_loss": detached_loss,
                        "sequence_log_probability_mean": float(
                            sum(detached_log_probs) / len(detached_log_probs)
                        ),
                        "gradient_norm": gradient_norm,
                        "gradient_sketch": sketch.tolist(),
                        "mean_active_tokens": float(
                            sampled.action_mask[:group_size].float().sum(1).mean().item()
                        ),
                    }
                )
        policy.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summaries = _summarize(logs, audit["group_sizes"])
    thresholds = audit["go_thresholds"]
    k32 = summaries["rollout_horizon_humans"]["32"]
    go = (
        k32["split_half_cosine"] >= thresholds["k32_split_half_cosine_min"]
        and k32["gradient_norm_cv"] <= thresholds["k32_gradient_norm_cv_max"]
    )
    artifact = {
        "artifact_schema": "dftr.m2.frozen_estimator_audit.result.v1",
        "status": "completed",
        "run_id": run_id,
        "git_sha": git_sha(),
        "config_sha256": canonical_hash(config),
        "audit_contract_sha256": config["workflow"]["audit_contract_sha256"],
        "observed_runtime": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
        },
        "human_document_count": len(full_human_texts),
        "rollout_document_count": len(rollout_rows),
        "bandwidths": bandwidths,
        "summaries": summaries,
        "go_for_global_k32_score_function_screen": go,
        "logs": logs,
    }
    write_json(output_path, artifact)
    write_json(output_dir / config["output"]["filename"], artifact)
    manifest = {
        "artifact_schema": "dftr.m2.frozen_estimator_audit.manifest.v1",
        "status": "completed",
        "run_id": run_id,
        "git_sha": git_sha(),
        "config_sha256": canonical_hash(config),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "go_for_global_k32_score_function_screen": go,
        "token_accounting": {
            "generated_tokens_max": int(audit["replicates"]) * max_group * horizon,
        },
    }
    write_json(checkpoint_dir / "run_manifest.json", manifest)
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
