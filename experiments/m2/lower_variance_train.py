"""Versioned three-arm lower-variance training screen for Qwen3-4B."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Sequence

from experiments.m1.contracts import build_run_paths, git_sha, write_json, write_jsonl
from experiments.m2.dft import (
    _activate_adapter,
    _directory_file_map,
    _load_jsonl,
    _require_no_existing_files,
    _verify_file,
    deterministic_batches,
)
from experiments.m2.lower_variance import (
    one_round_mmd_human_witness_weights,
    teacher_forced_token_moment_loss,
    validate_frequent_token_ids,
)
from experiments.m2.representation import (
    canonical_hash,
    frozen_base_embeddings,
    load_source_peft_and_tokenizer,
)


LOWER_VARIANCE_SCHEMA = "dftr.m2.lower_variance_three_arm.v1"
LOWER_VARIANCE_STEP = "train_lower_variance"
BASE_MODEL = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ARM_IDS = ("SFT", "TOKEN_MOMENT", "MMD_WITNESS")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FULL_BRIEF_SCHEMA = "dft.full-brief.tokens.v1"
FULL_BRIEF_SERIALIZER_SHA256 = canonical_hash(
    {
        "schema": FULL_BRIEF_SCHEMA,
        "required_fields": [
            "user_prompt",
            "use_case",
            "style_kind",
            "style",
            "detail_mode",
            "target_length",
            "target_length_unit",
            "em_dashes_allowed",
            "outline",
        ],
        "target_length_unit": "tokens",
        "brief_lines": [
            "Writing request: {user_prompt}",
            "Use case: {use_case}",
            "Style category: {style_kind}",
            "Style: {style}",
            "Detail mode: {detail_mode}",
            "Target length: about {target_length} tokens",
            "Em dashes allowed: {yes_or_no}",
            "Grounding outline (use only these supported facts when non-empty): {outline_json}",
        ],
    }
)
GENERATION_CONTRACT = {
    "sampling_distribution": "raw_policy_categorical.v1",
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 64,
    "stop_on_eos": True,
    "post_eos_behavior": "pad_and_mask",
    "teacher_forced_eos": "append_if_absent_after_truncation",
}
METHOD_KEYS = (
    "artifact_schema",
    "run",
    "compute",
    "model",
    "initial_adapter",
    "data",
    "representation",
    "objectives",
    "generation",
    "runtime",
    "training",
    "arms",
)
TOP_LEVEL_KEYS = set(METHOD_KEYS) | {"resume", "execution", "workflow"}


class LowerVarianceTrainError(ValueError):
    pass


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LowerVarianceTrainError(f"{field} must contain exactly {sorted(keys)}")
    return value


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise LowerVarianceTrainError(f"{field} must be a lowercase SHA-256")
    return text


def _positive(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LowerVarianceTrainError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise LowerVarianceTrainError(f"{field} must be finite and {qualifier}")
    return number


def method_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    workflow = config.get("workflow") or {}
    return {key: config.get(key) for key in METHOD_KEYS} | {
        "protocol_version": workflow.get("protocol_version"),
        "step": workflow.get("step"),
    }


def _validate_resume_descriptor(value: Any, arm_id: str) -> None:
    if value is None:
        return
    descriptor = _exact(
        value,
        {
            "path",
            "adapter_model_sha256",
            "adapter_config_sha256",
            "training_state_sha256",
            "file_manifest_sha256",
            "source_config_sha256",
        },
        f"resume.{arm_id}",
    )
    if not Path(str(descriptor.get("path") or "")).is_absolute():
        raise LowerVarianceTrainError(f"resume.{arm_id}.path must be absolute")
    for field in (
        "adapter_model_sha256",
        "adapter_config_sha256",
        "training_state_sha256",
        "file_manifest_sha256",
        "source_config_sha256",
    ):
        _sha(descriptor.get(field), f"resume.{arm_id}.{field}")


def validate_lower_variance_config(config: dict[str, Any]) -> dict[str, Any]:
    _exact(config, TOP_LEVEL_KEYS, "lower-variance config")
    if config.get("artifact_schema") != LOWER_VARIANCE_SCHEMA:
        raise LowerVarianceTrainError("unexpected lower-variance config schema")
    run = _exact(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") != "SFT-vs-TOKEN_MOMENT-vs-MMD_WITNESS"
        or run.get("budget_class") not in {"smoke", "screen"}
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or not isinstance(run.get("seed"), int)
        or isinstance(run.get("seed"), bool)
        or run["seed"] < 0
    ):
        raise LowerVarianceTrainError("run contract is not the frozen three-arm screen")
    compute = _exact(config.get("compute"), {"gpu", "gpus", "timeout_min"}, "compute")
    timeout_limit = 20 if run["budget_class"] == "smoke" else 120
    if (
        str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1
        or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int)
        or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= timeout_limit
    ):
        raise LowerVarianceTrainError("compute must use one supported GPU within budget")
    if config.get("model") != {
        "base": BASE_MODEL,
        "revision": BASE_REVISION,
        "torch_dtype": "bfloat16",
    }:
        raise LowerVarianceTrainError("model must be the frozen Qwen3-4B revision")
    adapter = _exact(
        config.get("initial_adapter"),
        {"path", "adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"},
        "initial_adapter",
    )
    if not Path(str(adapter.get("path") or "")).is_absolute():
        raise LowerVarianceTrainError("initial_adapter.path must be absolute")
    for field in ("adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"):
        _sha(adapter.get(field), f"initial_adapter.{field}")
    data = _exact(
        config.get("data"),
        {
            "anchor_path",
            "anchor_sha256",
            "witness_generated_path",
            "witness_generated_sha256",
            "witness_generation_contract_sha256",
            "completion_field",
            "generated_text_field",
            "prompt_format",
            "prompt_schema_version",
            "prompt_serializer_sha256",
        },
        "data",
    )
    for path_field, hash_field in (
        ("anchor_path", "anchor_sha256"),
        ("witness_generated_path", "witness_generated_sha256"),
    ):
        path = Path(str(data.get(path_field) or ""))
        if not path.is_absolute() or any(part in {"harness", "measurement_v2", "measurement_v3"} for part in path.parts):
            raise LowerVarianceTrainError(f"data.{path_field} must be absolute and training-only")
        _sha(data.get(hash_field), f"data.{hash_field}")
    if (
        not isinstance(data.get("completion_field"), str)
        or not data["completion_field"]
        or not isinstance(data.get("generated_text_field"), str)
        or not data["generated_text_field"]
        or data.get("witness_generation_contract_sha256")
        != canonical_hash(GENERATION_CONTRACT)
        or data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:"
        or data.get("prompt_schema_version") != FULL_BRIEF_SCHEMA
        or data.get("prompt_serializer_sha256") != FULL_BRIEF_SERIALIZER_SHA256
    ):
        raise LowerVarianceTrainError("data must use the canonical full-brief anchor contract")
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
        or representation.get("role") != "lower_variance_training_only_not_measurement_v3"
    ):
        raise LowerVarianceTrainError("representation must be the frozen training-only base embedding")
    for field in ("batch_size", "max_tokens"):
        if (
            not isinstance(representation.get(field), int)
            or isinstance(representation.get(field), bool)
            or representation[field] <= 0
        ):
            raise LowerVarianceTrainError(f"representation.{field} must be positive integer")
    objectives = _exact(config.get("objectives"), {"token_moments", "mmd_witness"}, "objectives")
    token_moments = _exact(
        objectives.get("token_moments"),
        {
            "coefficient",
            "first_moment_weight",
            "second_moment_weight",
            "frequent_token_ids",
            "vocabulary_source_sha256",
        },
        "objectives.token_moments",
    )
    _positive(token_moments.get("coefficient"), "token moment coefficient")
    first_weight = _positive(
        token_moments.get("first_moment_weight"), "first moment weight", allow_zero=True
    )
    second_weight = _positive(
        token_moments.get("second_moment_weight"), "second moment weight", allow_zero=True
    )
    if first_weight == 0.0 and second_weight == 0.0:
        raise LowerVarianceTrainError("at least one token moment weight must be positive")
    token_ids = token_moments.get("frequent_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise LowerVarianceTrainError("frequent_token_ids must be a nonempty frozen list")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise LowerVarianceTrainError("frequent_token_ids must contain nonnegative integer IDs")
    if any(left >= right for left, right in zip(token_ids, token_ids[1:])):
        raise LowerVarianceTrainError("frequent_token_ids must be unique and strictly increasing")
    _sha(token_moments.get("vocabulary_source_sha256"), "token vocabulary source")
    witness = _exact(
        objectives.get("mmd_witness"),
        {"bandwidths", "temperature", "weighting", "human_self_kernel"},
        "objectives.mmd_witness",
    )
    bandwidths = witness.get("bandwidths")
    if not isinstance(bandwidths, list) or not bandwidths:
        raise LowerVarianceTrainError("MMD bandwidths must be nonempty")
    normalized_bandwidths = [_positive(value, "MMD bandwidth") for value in bandwidths]
    if len(set(normalized_bandwidths)) != len(normalized_bandwidths):
        raise LowerVarianceTrainError("MMD bandwidths must be unique")
    _positive(witness.get("temperature"), "MMD witness temperature")
    if (
        witness.get("weighting") != "softmax_mean_one.v1"
        or witness.get("human_self_kernel") != "leave_one_out"
    ):
        raise LowerVarianceTrainError("MMD witness contract is not frozen")
    generation = _exact(
        config.get("generation"), set(GENERATION_CONTRACT), "generation"
    )
    if (
        type(generation.get("temperature")) is not float
        or type(generation.get("top_p")) is not float
        or type(generation.get("top_k")) is not int
        or type(generation.get("max_new_tokens")) is not int
        or type(generation.get("stop_on_eos")) is not bool
        or generation.get("stop_on_eos") is not True
        or not isinstance(generation.get("sampling_distribution"), str)
        or not isinstance(generation.get("post_eos_behavior"), str)
        or not isinstance(generation.get("teacher_forced_eos"), str)
        or generation != GENERATION_CONTRACT
    ):
        raise LowerVarianceTrainError("generation must use the frozen EOS-aware contract")
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
        any(
            not isinstance(runtime.get(field), str) or not runtime[field]
            for field in ("torch_version", "transformers_version", "peft_version")
        )
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
    ):
        raise LowerVarianceTrainError("runtime must freeze versions and deterministic execution")
    training = _exact(
        config.get("training"),
        {
            "steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "max_input_tokens",
            "checkpoint_every",
            "schedule",
        },
        "training",
    )
    for field in ("steps", "batch_size", "max_input_tokens", "checkpoint_every"):
        if (
            not isinstance(training.get(field), int)
            or isinstance(training.get(field), bool)
            or training[field] <= 0
        ):
            raise LowerVarianceTrainError(f"training.{field} must be positive integer")
    for field in ("learning_rate", "gradient_clip_norm"):
        _positive(training.get(field), f"training.{field}")
    _positive(training.get("weight_decay"), "training.weight_decay", allow_zero=True)
    if (
        training["steps"] % training["checkpoint_every"] != 0
        or training.get("schedule") != "python_random_sample_without_replacement.v1"
    ):
        raise LowerVarianceTrainError("training must end on exact checkpoint boundaries")
    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ARM_IDS):
        raise LowerVarianceTrainError("arms must be the exact matched three-arm objective set")
    for index, arm in enumerate(arms):
        _exact(
            arm,
            {"id", "sft_weighting", "token_moment_coefficient"},
            f"arms[{index}]",
        )
        if (
            not isinstance(arm.get("id"), str)
            or not isinstance(arm.get("sft_weighting"), str)
        ):
            raise LowerVarianceTrainError("arms must use exact string selectors")
        _positive(
            arm.get("token_moment_coefficient"),
            f"arms[{index}].token_moment_coefficient",
            allow_zero=True,
        )
    expected_arms = [
        {"id": "SFT", "sft_weighting": "uniform", "token_moment_coefficient": 0.0},
        {
            "id": "TOKEN_MOMENT",
            "sft_weighting": "uniform",
            "token_moment_coefficient": float(token_moments["coefficient"]),
        },
        {"id": "MMD_WITNESS", "sft_weighting": "mmd_witness", "token_moment_coefficient": 0.0},
    ]
    if arms != expected_arms:
        raise LowerVarianceTrainError("arms must be the exact matched three-arm objective set")
    resume = _exact(config.get("resume"), set(ARM_IDS), "resume")
    for arm_id in ARM_IDS:
        _validate_resume_descriptor(resume[arm_id], arm_id)
    execution = _exact(config.get("execution"), {"arm"}, "execution")
    if execution.get("arm") not in ARM_IDS:
        raise LowerVarianceTrainError("execution.arm must select one frozen arm")
    workflow = _exact(
        config.get("workflow"),
        {"protocol_version", "step", "method_contract_sha256"},
        "workflow",
    )
    if (
        workflow.get("protocol_version") != LOWER_VARIANCE_SCHEMA
        or workflow.get("step") != LOWER_VARIANCE_STEP
        or canonical_hash(method_contract_payload(config))
        != _sha(workflow.get("method_contract_sha256"), "workflow.method_contract_sha256")
    ):
        raise LowerVarianceTrainError("lower-variance workflow contract hash mismatch")
    return config


def selected_arm(config: dict[str, Any]) -> dict[str, Any]:
    arm_id = str((config.get("execution") or {}).get("arm") or "")
    matches = [arm for arm in config.get("arms") or [] if arm.get("id") == arm_id]
    if len(matches) != 1:
        raise LowerVarianceTrainError("execution selector does not identify one arm")
    return matches[0]


def eos_aware_completion_ids(
    token_ids: Sequence[int], eos_token_id: int, max_new_tokens: int
) -> list[int]:
    if (
        type(eos_token_id) is not int
        or eos_token_id < 0
        or type(max_new_tokens) is not int
        or max_new_tokens < 1
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise LowerVarianceTrainError("EOS completion contract requires valid integer IDs")
    bounded = list(token_ids[:max_new_tokens])
    if eos_token_id in bounded:
        return bounded[: bounded.index(eos_token_id) + 1]
    return bounded[: max_new_tokens - 1] + [eos_token_id]


def _render_lower_variance_prompt(record: dict[str, Any], config: dict[str, Any]) -> str:
    required = (
        "user_prompt",
        "use_case",
        "style_kind",
        "style",
        "detail_mode",
        "target_length",
        "target_length_unit",
        "em_dashes_allowed",
        "outline",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise LowerVarianceTrainError(
            f"token-unit brief is missing fields: {', '.join(missing)}"
        )
    if record["target_length_unit"] != "tokens":
        raise LowerVarianceTrainError("target_length_unit must be tokens")
    target_length = record["target_length"]
    if isinstance(target_length, bool) or not isinstance(target_length, int) or target_length <= 0:
        raise LowerVarianceTrainError("target_length must be a positive token count")
    outline = record["outline"]
    if not isinstance(outline, list):
        raise LowerVarianceTrainError("token-unit brief outline must be a list")
    user_prompt = str(record["user_prompt"]).strip()
    if not user_prompt:
        raise LowerVarianceTrainError("token-unit brief user_prompt is empty")
    brief = "\n".join(
        (
            f"Writing request: {user_prompt}",
            f"Use case: {str(record['use_case']).strip()}",
            f"Style category: {str(record['style_kind']).strip()}",
            f"Style: {str(record['style']).strip()}",
            f"Detail mode: {str(record['detail_mode']).strip()}",
            f"Target length: about {target_length} tokens",
            f"Em dashes allowed: {'yes' if bool(record['em_dashes_allowed']) else 'no'}",
            "Grounding outline (use only these supported facts when non-empty): "
            + json.dumps(
                outline,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    return str(config["data"]["prompt_format"]).format(brief=brief)


def prepare_supervised_batch(
    tokenizer: Any, records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    import torch

    if not records:
        raise LowerVarianceTrainError("supervised batch cannot be empty")
    eos_token_id = tokenizer.eos_token_id
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise LowerVarianceTrainError("tokenizer.eos_token_id is required")
    prompts = [_render_lower_variance_prompt(record, config) for record in records]
    completion_field = str(config["data"]["completion_field"])
    completions = [str(record.get(completion_field) or "").strip() for record in records]
    if any(not completion for completion in completions):
        raise LowerVarianceTrainError("anchor record lacks a completion")
    prompt_ids = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=True,
        max_length=int(config["training"]["max_input_tokens"]),
    )["input_ids"]
    raw_completion_ids = tokenizer(
        completions,
        add_special_tokens=False,
        truncation=True,
        max_length=int(config["generation"]["max_new_tokens"]),
    )["input_ids"]
    completion_ids = [
        eos_aware_completion_ids(
            list(tokens), eos_token_id, int(config["generation"]["max_new_tokens"])
        )
        for tokens in raw_completion_ids
    ]
    combined = [
        {"input_ids": prefix + suffix, "attention_mask": [1] * (len(prefix) + len(suffix))}
        for prefix, suffix in zip(prompt_ids, completion_ids)
    ]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        padded = tokenizer.pad(combined, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = previous_padding_side
    labels = padded["input_ids"].clone()
    for index, prefix in enumerate(prompt_ids):
        labels[index, : len(prefix)] = -100
    labels[padded["attention_mask"] == 0] = -100
    if bool((labels != -100).sum(dim=1).eq(0).any().item()):
        raise LowerVarianceTrainError("every anchor must expose at least one completion token")
    return {
        "input_ids": padded["input_ids"].to(dtype=torch.long),
        "attention_mask": padded["attention_mask"].to(dtype=torch.long),
        "labels": labels.to(dtype=torch.long),
        "completion_token_counts": (labels != -100).sum(dim=1),
    }


def objective_components(
    logits: Any,
    labels: Any,
    frequent_token_ids: Sequence[int],
    example_weights: Any,
    arm: dict[str, Any],
    objectives: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise LowerVarianceTrainError("objective logits and labels must align")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    if bool(valid.sum(dim=1).eq(0).any().item()):
        raise LowerVarianceTrainError("each objective example needs a target token")
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    per_token = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape_as(safe_labels)
    per_document_sft = (per_token * valid).sum(dim=1) / valid.sum(dim=1)
    uniform_sft = per_document_sft.mean()
    if (
        example_weights.ndim != 1
        or example_weights.shape[0] != logits.shape[0]
        or not bool(torch.isfinite(example_weights).all().item())
        or bool((example_weights <= 0).any().item())
    ):
        raise LowerVarianceTrainError("example weights must be finite positive batch weights")
    normalized_weights = example_weights.to(device=logits.device, dtype=uniform_sft.dtype)
    weighted_sft = (per_document_sft * normalized_weights).sum() / normalized_weights.sum()
    zero = uniform_sft * 0.0
    if arm["id"] == "MMD_WITNESS":
        witness_delta = weighted_sft - uniform_sft
    else:
        witness_delta = zero
    if arm["id"] == "TOKEN_MOMENT":
        token_config = objectives["token_moments"]
        validate_frequent_token_ids(frequent_token_ids, int(logits.shape[-1]))
        raw_moment = teacher_forced_token_moment_loss(
            shifted_logits,
            safe_labels,
            valid,
            frequent_token_ids,
            first_moment_weight=float(token_config["first_moment_weight"]),
            second_moment_weight=float(token_config["second_moment_weight"]),
        )
        moment_component = float(arm["token_moment_coefficient"]) * raw_moment
    else:
        raw_moment = zero
        moment_component = zero
    total = uniform_sft + witness_delta + moment_component
    return {
        "uniform_sft": uniform_sft,
        "weighted_sft": weighted_sft,
        "raw_token_moment": raw_moment,
        "token_moment_component": moment_component,
        "witness_delta_component": witness_delta,
        "total": total,
    }


def component_gradient_norm(loss: Any, parameters: Sequence[Any]) -> float:
    import torch

    parameter_list = list(parameters)
    gradients = torch.autograd.grad(
        loss, parameter_list, retain_graph=True, allow_unused=True
    )
    squared = torch.zeros((), device=loss.device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().double().square().sum()
    norm = float(torch.sqrt(squared).item())
    if not math.isfinite(norm):
        raise LowerVarianceTrainError("component gradient norm is non-finite")
    return norm


def matched_exposure_payload(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    per_arm = {
        arm_id: {
            "steps": int(training["steps"]),
            "batch_size": int(training["batch_size"]),
            "optimizer_updates": int(training["steps"]),
        }
        for arm_id in ARM_IDS
    }
    return {
        "artifact_schema": "dftr.m2.lower_variance_matched_exposure.v1",
        "comparison_id": config["run"]["comparison_id"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "initial_adapter": config["initial_adapter"],
        "anchor_sha256": config["data"]["anchor_sha256"],
        "witness_generated_sha256": config["data"]["witness_generated_sha256"],
        "witness_generation_contract_sha256": config["data"][
            "witness_generation_contract_sha256"
        ],
        "prompt_serializer_sha256": config["data"]["prompt_serializer_sha256"],
        "generation": config["generation"],
        "schedule": training["schedule"],
        "schedule_seed": int(config["run"]["seed"]),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
        },
        "arms": per_arm,
    }


def _verify_inputs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import peft
    import torch
    import transformers

    observed_runtime = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
    }
    expected_runtime = {
        field: config["runtime"][field]
        for field in ("torch_version", "transformers_version", "peft_version")
    }
    if observed_runtime != expected_runtime:
        raise LowerVarianceTrainError(
            f"runtime version mismatch: expected {expected_runtime} observed {observed_runtime}"
        )
    adapter = config["initial_adapter"]
    adapter_path = Path(adapter["path"])
    _verify_file(
        adapter_path / "adapter_model.safetensors",
        adapter["adapter_model_sha256"],
        "initial adapter model",
    )
    _verify_file(
        adapter_path / "adapter_config.json",
        adapter["adapter_config_sha256"],
        "initial adapter config",
    )
    if canonical_hash(_directory_file_map(adapter_path, "initial adapter")) != adapter[
        "file_manifest_sha256"
    ]:
        raise LowerVarianceTrainError("initial adapter complete file manifest mismatch")
    try:
        adapter_config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise LowerVarianceTrainError(f"invalid initial adapter config: {exc.msg}") from exc
    if (
        not isinstance(adapter_config, dict)
        or adapter_config.get("base_model_name_or_path") != BASE_MODEL
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("peft_type") != "LORA"
    ):
        raise LowerVarianceTrainError("initial adapter is not the frozen 4B causal LoRA")
    data = config["data"]
    anchor_path = Path(data["anchor_path"])
    generated_path = Path(data["witness_generated_path"])
    _verify_file(anchor_path, data["anchor_sha256"], "anchor panel")
    _verify_file(generated_path, data["witness_generated_sha256"], "witness generated panel")
    anchors = _load_jsonl(anchor_path, "lower-variance anchor panel")
    generated = _load_jsonl(generated_path, "lower-variance witness panel")
    completion_field = str(data["completion_field"])
    generated_field = str(data["generated_text_field"])
    if any(not str(row.get(completion_field) or "").strip() for row in anchors):
        raise LowerVarianceTrainError("anchor panel contains an empty human completion")
    if any(not str(row.get(generated_field) or "").strip() for row in generated):
        raise LowerVarianceTrainError("witness panel contains an empty generated document")
    return anchors, generated


def _witness_artifact(
    model: Any,
    tokenizer: Any,
    anchors: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    human_texts = [str(row[config["data"]["completion_field"]]).strip() for row in anchors]
    generated_texts = [
        str(row[config["data"]["generated_text_field"]]).strip() for row in generated
    ]
    human_embeddings = frozen_base_embeddings(model, tokenizer, human_texts, config)
    generated_embeddings = frozen_base_embeddings(model, tokenizer, generated_texts, config)
    witness_config = config["objectives"]["mmd_witness"]
    result = one_round_mmd_human_witness_weights(
        generated_embeddings.detach(),
        human_embeddings.detach(),
        witness_config["bandwidths"],
        temperature=float(witness_config["temperature"]),
    )
    payload = {
        "artifact_schema": "dftr.m2.one_round_mmd_witness_weights.v1",
        "anchor_sha256": config["data"]["anchor_sha256"],
        "witness_generated_sha256": config["data"]["witness_generated_sha256"],
        "witness_generation_contract_sha256": config["data"][
            "witness_generation_contract_sha256"
        ],
        "representation_sha256": canonical_hash(config["representation"]),
        "initial_adapter_file_manifest_sha256": config["initial_adapter"][
            "file_manifest_sha256"
        ],
        "bandwidths": witness_config["bandwidths"],
        "temperature": float(witness_config["temperature"]),
        "witness": [float(value) for value in result.witness.cpu()],
        "weights": [float(value) for value in result.weights.cpu()],
        "weighting": "softmax_mean_one.v1",
        "human_self_kernel": "leave_one_out",
    }
    return result.weights, payload


def _verify_resume_artifact(
    descriptor: dict[str, Any],
    arm_id: str,
    config: dict[str, Any],
    schedule_sha256: str,
    witness_sha256: str,
) -> dict[str, Any]:
    import torch

    root = Path(descriptor["path"])
    for filename, hash_field, label in (
        ("adapter_model.safetensors", "adapter_model_sha256", "resume adapter model"),
        ("adapter_config.json", "adapter_config_sha256", "resume adapter config"),
        ("training_state.pt", "training_state_sha256", "resume training state"),
    ):
        _verify_file(root / filename, descriptor[hash_field], label)
    if canonical_hash(_directory_file_map(root, f"{arm_id} resume artifact")) != descriptor[
        "file_manifest_sha256"
    ]:
        raise LowerVarianceTrainError(f"{arm_id} resume file manifest mismatch")
    state = torch.load(root / "training_state.pt", map_location="cpu", weights_only=False)
    expected_keys = {
        "artifact_schema",
        "arm",
        "next_step",
        "method_contract_sha256",
        "config_sha256",
        "source_adapter_manifest_sha256",
        "schedule_sha256",
        "witness_weights_sha256",
        "optimizer_state",
        "cpu_rng_state",
        "cuda_rng_state_all",
        "python_rng_state",
        "logs",
        "optimizer_examples",
        "teacher_forced_completion_tokens",
    }
    next_step = state.get("next_step") if isinstance(state, dict) else None
    if (
        not isinstance(state, dict)
        or set(state) != expected_keys
        or state.get("artifact_schema") != "dftr.m2.lower_variance_training_state.v1"
        or state.get("arm") != arm_id
        or state.get("method_contract_sha256") != config["workflow"]["method_contract_sha256"]
        or state.get("config_sha256") != descriptor["source_config_sha256"]
        or state.get("source_adapter_manifest_sha256")
        != config["initial_adapter"]["file_manifest_sha256"]
        or state.get("schedule_sha256") != schedule_sha256
        or state.get("witness_weights_sha256") != witness_sha256
        or not isinstance(next_step, int)
        or isinstance(next_step, bool)
        or not 0 < next_step <= int(config["training"]["steps"])
        or next_step % int(config["training"]["checkpoint_every"]) != 0
        or not isinstance(state.get("logs"), list)
        or len(state["logs"]) != next_step
        or state.get("optimizer_examples")
        != next_step * int(config["training"]["batch_size"])
        or not isinstance(state.get("teacher_forced_completion_tokens"), int)
        or state["teacher_forced_completion_tokens"] <= 0
    ):
        raise LowerVarianceTrainError(f"{arm_id} resume provenance mismatch")
    return state


def _save_training_checkpoint(
    policy: Any,
    optimizer: Any,
    target: Path,
    arm_id: str,
    next_step: int,
    logs: list[dict[str, Any]],
    optimizer_examples: int,
    teacher_forced_completion_tokens: int,
    schedule_sha256: str,
    witness_sha256: str,
    config: dict[str, Any],
) -> None:
    import torch

    target.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(target, safe_serialization=True, selected_adapters=["default"])
    state = {
        "artifact_schema": "dftr.m2.lower_variance_training_state.v1",
        "arm": arm_id,
        "next_step": next_step,
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "config_sha256": canonical_hash(config),
        "source_adapter_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "schedule_sha256": schedule_sha256,
        "witness_weights_sha256": witness_sha256,
        "optimizer_state": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng_state": random.getstate(),
        "logs": logs,
        "optimizer_examples": optimizer_examples,
        "teacher_forced_completion_tokens": teacher_forced_completion_tokens,
    }
    torch.save(state, target / "training_state.pt")


def _run_arm(
    config: dict[str, Any],
    checkpoint_dir: Path,
    anchors: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    arm = selected_arm(config)
    arm_id = str(arm["id"])
    seed = int(config["run"]["seed"])
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    schedule = deterministic_batches(len(anchors), batch_size, steps, seed)
    schedule_sha256 = canonical_hash(schedule)
    resume_descriptor = config["resume"][arm_id]
    policy, tokenizer = load_source_peft_and_tokenizer(
        config, str(resume_descriptor["path"]) if resume_descriptor else None
    )
    trainable = _activate_adapter(policy, "default", trainable=True)
    policy.train()
    policy.config.use_cache = False
    validate_frequent_token_ids(
        config["objectives"]["token_moments"]["frequent_token_ids"],
        int(policy.config.vocab_size),
    )
    witness_weights, witness_payload = _witness_artifact(
        policy, tokenizer, anchors, generated, config
    )
    witness_sha256 = canonical_hash(witness_payload)
    arm_dir = checkpoint_dir / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    write_json(arm_dir / "one_round_witness_weights.json", witness_payload)
    resume_state = (
        _verify_resume_artifact(
            resume_descriptor,
            arm_id,
            config,
            schedule_sha256,
            witness_sha256,
        )
        if resume_descriptor
        else None
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state"])
    start_step = int(resume_state["next_step"]) if resume_state else 0
    logs = list(resume_state["logs"]) if resume_state else []
    optimizer_examples = int(resume_state["optimizer_examples"]) if resume_state else 0
    teacher_forced_tokens = (
        int(resume_state["teacher_forced_completion_tokens"]) if resume_state else 0
    )
    if resume_state:
        random.setstate(resume_state["python_rng_state"])
        torch.set_rng_state(resume_state["cpu_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
    frequent_token_ids = config["objectives"]["token_moments"]["frequent_token_ids"]
    for step in range(start_step, steps):
        indices = schedule[step]
        batch = prepare_supervised_batch(
            tokenizer, [anchors[index] for index in indices], config
        )
        device = next(policy.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = policy(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).logits
        if arm_id == "MMD_WITNESS":
            batch_weights = witness_weights[indices].to(device)
        else:
            batch_weights = torch.ones(len(indices), device=device)
        components = objective_components(
            logits,
            labels,
            frequent_token_ids,
            batch_weights,
            arm,
            config["objectives"],
        )
        if not bool(torch.isfinite(components["total"]).item()):
            raise LowerVarianceTrainError("lower-variance objective is non-finite")
        optimizer.zero_grad(set_to_none=True)
        gradient_components = {
            "uniform_sft": component_gradient_norm(components["uniform_sft"], trainable),
            "token_moment": component_gradient_norm(
                components["token_moment_component"], trainable
            ),
            "witness_delta": component_gradient_norm(
                components["witness_delta_component"], trainable
            ),
            "total": component_gradient_norm(components["total"], trainable),
        }
        components["total"].backward()
        preclip_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["training"]["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(preclip_norm).item()):
            raise LowerVarianceTrainError("lower-variance gradient is non-finite")
        optimizer.step()
        optimizer_examples += batch_size
        teacher_forced_tokens += int(batch["completion_token_counts"].sum().item())
        logs.append(
            {
                "arm": arm_id,
                "step": step,
                "anchor_indices": indices,
                "optimizer_examples": batch_size,
                "teacher_forced_completion_tokens": int(
                    batch["completion_token_counts"].sum().item()
                ),
                "uniform_sft_loss": float(components["uniform_sft"].detach()),
                "weighted_sft_loss": float(components["weighted_sft"].detach()),
                "raw_token_moment_loss": float(components["raw_token_moment"].detach()),
                "token_moment_component": float(
                    components["token_moment_component"].detach()
                ),
                "witness_delta_component": float(
                    components["witness_delta_component"].detach()
                ),
                "total_loss": float(components["total"].detach()),
                "component_gradient_norms": gradient_components,
                "preclip_total_gradient_norm": float(preclip_norm),
                "batch_weight_min": float(batch_weights.min()),
                "batch_weight_max": float(batch_weights.max()),
                "batch_weight_mean": float(batch_weights.mean()),
            }
        )
        if (step + 1) % int(config["training"]["checkpoint_every"]) == 0:
            _save_training_checkpoint(
                policy,
                optimizer,
                arm_dir / f"step-{step + 1}",
                arm_id,
                step + 1,
                logs,
                optimizer_examples,
                teacher_forced_tokens,
                schedule_sha256,
                witness_sha256,
                config,
            )
    policy.save_pretrained(arm_dir, safe_serialization=True, selected_adapters=["default"])
    tokenizer.save_pretrained(arm_dir)
    write_jsonl(arm_dir / "training_steps.jsonl", logs)
    file_map = _directory_file_map(arm_dir, f"{arm_id} lower-variance output")
    manifest = {
        "artifact_schema": "dftr.m2.lower_variance_adapter_checkpoint.v1",
        "arm": arm_id,
        "status": "completed",
        "adapter_native": True,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_adapter_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "config_sha256": canonical_hash(config),
        "schedule_sha256": schedule_sha256,
        "witness_weights_sha256": witness_sha256,
        "checkpoint_dir": str(arm_dir.resolve()),
        "file_sha256": file_map,
        "file_map_excludes": ["checkpoint_manifest.json"],
        "steps": steps,
        "resumed_from_step": start_step,
        "optimizer_examples": optimizer_examples,
        "teacher_forced_completion_tokens": teacher_forced_tokens,
        "generated_tokens": 0,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "git_sha": git_sha(),
    }
    write_json(arm_dir / "checkpoint_manifest.json", manifest)
    return manifest


def run_lower_variance(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Train one selected arm under the frozen matched three-arm contract."""
    validate_lower_variance_config(config)
    if not SAFE_ID_RE.fullmatch(str(run_id)):
        raise LowerVarianceTrainError("run_id is not a safe artifact identifier")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["runtime"]["cublas_workspace_config"]
    anchors, generated = _verify_inputs(config)
    if len(anchors) < int(config["training"]["batch_size"]):
        raise LowerVarianceTrainError("anchor panel is smaller than one training batch")
    if len(generated) < 1:
        raise LowerVarianceTrainError("witness generated panel is empty")
    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    _require_no_existing_files(checkpoint_dir, "lower-variance checkpoint directory")
    if output_dir.resolve() != checkpoint_dir.resolve():
        existing_output_files = [
            item
            for item in output_dir.rglob("*")
            if (item.is_file() or item.is_symlink())
            and checkpoint_dir.resolve() not in item.parents
        ]
        if existing_output_files:
            raise LowerVarianceTrainError("run output directory already contains artifacts")
    write_json(checkpoint_dir / "config.json", config)
    arm_manifest = _run_arm(config, checkpoint_dir, anchors, generated)
    exposure = matched_exposure_payload(config)
    result = {
        "artifact_schema": "dftr.m2.lower_variance_three_arm_result.v1",
        "run_id": run_id,
        "comparison_id": config["run"]["comparison_id"],
        "status": "completed",
        "executed_arm": config["execution"]["arm"],
        "git_sha": git_sha(),
        "config_sha256": canonical_hash(config),
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "arm": arm_manifest,
        "matched_exposure_contract": exposure,
        "matched_exposure_contract_sha256": canonical_hash(exposure),
        "token_accounting": {
            "generated_tokens": 0,
            "optimizer_examples": arm_manifest["optimizer_examples"],
            "teacher_forced_completion_tokens": arm_manifest[
                "teacher_forced_completion_tokens"
            ],
        },
        "scientific_interpretation": "requires independent non-training representations and matched held-out evaluation",
    }
    write_json(checkpoint_dir / "run_manifest.json", result)
    write_json(output_dir / "lower_variance_manifest.json", result)
    return result


__all__ = [
    "ARM_IDS",
    "GENERATION_CONTRACT",
    "LOWER_VARIANCE_SCHEMA",
    "LOWER_VARIANCE_STEP",
    "LowerVarianceTrainError",
    "component_gradient_norm",
    "eos_aware_completion_ids",
    "matched_exposure_payload",
    "method_contract_payload",
    "objective_components",
    "prepare_supervised_batch",
    "run_lower_variance",
    "selected_arm",
    "validate_lower_variance_config",
]
