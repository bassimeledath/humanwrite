"""Prospective adapter-native score-function MMD training for the first M2 screen."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Sequence

from experiments.m1.contracts import (
    build_run_paths,
    file_sha256,
    git_sha,
    write_json,
    write_jsonl,
)
from experiments.m2.readiness import ReadinessError, verify_a64_readiness
from experiments.m2.prepare_dft import (
    preparation_contract_payload,
    validate_prepare_dft_config,
)
from experiments.m2.representation import (
    TRAINING_BANDWIDTH_DERIVATION,
    TRAINING_BANDWIDTH_PARAMETERIZATION,
    frozen_base_embeddings,
    load_source_peft_and_tokenizer,
    masked_hidden_embeddings as _masked_hidden_embeddings,
    representation_execution_payload,
)
from experiments.tier0.metrics import length_stats, outline_fact_recall, unsupported_claim_rate


DFT_SCHEMA = "dftr.m2.score_function_mmd.v1"
DFT_STEP = "train_dft"
BASE_MODEL = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ARM_IDS = ["A0", "A64"]
GENERATED_TOKENS = 64
FULL_BRIEF_SCHEMA = "dft.full-brief.v1"
A64_READINESS_ENV = "DFTR_M2_A64_READINESS_MANIFEST"
A64_READINESS_SHA_ENV = "DFTR_M2_A64_READINESS_SHA256"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALLOWED_TOP_LEVEL = {
    "artifact_schema",
    "run",
    "compute",
    "model",
    "initial_adapter",
    "data",
    "representation",
    "kernel",
    "runtime",
    "training",
    "arms",
    "stop",
    "readiness_trust",
    "resume",
    "execution",
    "workflow",
}


class M2ConfigError(ValueError):
    pass


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


FULL_BRIEF_SERIALIZER_SHA256 = canonical_hash({
    "schema": FULL_BRIEF_SCHEMA,
    "required_fields": [
        "user_prompt", "use_case", "style_kind", "style", "detail_mode",
        "target_length", "em_dashes_allowed", "outline",
    ],
    "brief_lines": [
        "Writing request: {user_prompt}",
        "Use case: {use_case}",
        "Style category: {style_kind}",
        "Style: {style}",
        "Detail mode: {detail_mode}",
        "Target length: about {target_length} words",
        "Em dashes allowed: {yes_or_no}",
        "Grounding outline (use only these supported facts when non-empty): {outline_json}",
    ],
    "outline_json": {"ensure_ascii": False, "sort_keys": True, "separators": [",", ":"]},
    "prompt_format": "USER:\n{brief}\nASSISTANT:",
})


def _require_exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise M2ConfigError(f"{field} must contain exactly {sorted(keys)}")
    return value


def _require_sha(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise M2ConfigError(f"{field} must be a lowercase SHA-256")
    return text


def _require_positive(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M2ConfigError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise M2ConfigError(f"{field} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


def method_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
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
            "kernel",
            "runtime",
            "training",
            "arms",
            "stop",
            "readiness_trust",
        )
    } | {
        "protocol_version": (config.get("workflow") or {}).get("protocol_version"),
        "step": (config.get("workflow") or {}).get("step"),
    }


def validate_dft_config(config: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(config, ALLOWED_TOP_LEVEL, "config")
    if config.get("artifact_schema") != DFT_SCHEMA:
        raise M2ConfigError("unexpected M2 DFT config schema")
    run = _require_exact_keys(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or
        run.get("arm") != "A0-vs-A64"
        or run.get("task_kind") != "experiment"
        or run.get("budget_class") not in {"smoke", "screen"}
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or not isinstance(run.get("seed"), int)
        or isinstance(run.get("seed"), bool)
        or int(run["seed"]) < 0
    ):
        raise M2ConfigError("run must freeze one deterministic A0-vs-A64 experiment")
    compute = _require_exact_keys(
        config.get("compute"), {"gpu", "gpus", "timeout_min"}, "compute"
    )
    if (
        str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or not isinstance(compute.get("gpus"), int)
        or isinstance(compute.get("gpus"), bool)
        or compute.get("gpus") != 1
        or not isinstance(compute.get("timeout_min"), int)
        or isinstance(compute.get("timeout_min"), bool)
        or int(compute["timeout_min"]) <= 0
    ):
        raise M2ConfigError("M2 DFT requires one supported training GPU and a positive timeout")
    timeout_limit = 20 if run["budget_class"] == "smoke" else 120
    if int(compute["timeout_min"]) > timeout_limit:
        raise M2ConfigError("compute timeout exceeds the frozen budget class")
    model = _require_exact_keys(
        config.get("model"), {"base", "revision", "torch_dtype"}, "model"
    )
    if model != {"base": BASE_MODEL, "revision": BASE_REVISION, "torch_dtype": "bfloat16"}:
        raise M2ConfigError("M2 DFT requires the frozen Qwen3-4B bfloat16 base")
    adapter = _require_exact_keys(
        config.get("initial_adapter"),
        {"path", "adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"},
        "initial_adapter",
    )
    if not Path(str(adapter.get("path") or "")).is_absolute():
        raise M2ConfigError("initial adapter must use an absolute artifact path")
    for field in ("adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"):
        _require_sha(adapter.get(field), f"initial_adapter.{field}")
    data = _require_exact_keys(
        config.get("data"),
        {
            "rollout_path", "rollout_sha256", "sft_anchor_path", "sft_anchor_sha256",
            "human_targets_path", "human_targets_sha256",
            "completion_field", "human_text_field", "prompt_format",
            "prompt_schema_version", "prompt_serializer_sha256",
        },
        "data",
    )
    for field in ("rollout_sha256", "sft_anchor_sha256", "human_targets_sha256"):
        _require_sha(data.get(field), f"data.{field}")
    for field in ("rollout_path", "sft_anchor_path", "human_targets_path"):
        path = Path(str(data.get(field) or ""))
        if not path.is_absolute() or "measurement_v2" in path.parts or "harness" in path.parts:
            raise M2ConfigError(f"data.{field} must be an absolute training-only artifact path")
    if any(
        not isinstance(data.get(field), str) or not data[field]
        for field in ("completion_field", "human_text_field")
    ):
        raise M2ConfigError("data text field names must be nonempty strings")
    if data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:":
        raise M2ConfigError("M2 DFT requires the canonical full-brief prompt format")
    if (
        data.get("prompt_schema_version") != FULL_BRIEF_SCHEMA
        or data.get("prompt_serializer_sha256") != FULL_BRIEF_SERIALIZER_SHA256
    ):
        raise M2ConfigError("M2 DFT requires the hash-bound canonical full-brief serializer")
    representation = _require_exact_keys(
        config.get("representation"),
        {
            "model", "revision", "layer", "pooling", "normalize", "role",
            "batch_size", "max_tokens",
        },
        "representation",
    )
    if (
        representation.get("model") != BASE_MODEL
        or representation.get("revision") != BASE_REVISION
        or representation.get("pooling") != "attention_masked_mean"
        or representation.get("normalize") is not True
        or representation.get("role") != "training_only_not_measurement_v2"
        or representation.get("layer") != -1
    ):
        raise M2ConfigError("training representation must be the frozen base hidden-state contract")
    for field in ("batch_size", "max_tokens"):
        value = representation.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise M2ConfigError(f"representation.{field} must be a positive integer")
    kernel = _require_exact_keys(
        config.get("kernel"), {"bandwidths_path", "bandwidths_sha256", "source"}, "kernel"
    )
    _require_sha(kernel.get("bandwidths_sha256"), "kernel.bandwidths_sha256")
    bandwidths_path = Path(str(kernel.get("bandwidths_path") or ""))
    if (
        not bandwidths_path.is_absolute()
        or "measurement_v2" in bandwidths_path.parts
        or "harness" in bandwidths_path.parts
    ):
        raise M2ConfigError("kernel bandwidths must use an absolute training-only artifact path")
    if kernel.get("source") != "training_humans_only":
        raise M2ConfigError("kernel bandwidths must be frozen from training humans only")
    runtime = _require_exact_keys(
        config.get("runtime"),
        {
            "torch_version", "transformers_version", "peft_version",
            "deterministic_algorithms", "cublas_workspace_config",
        },
        "runtime",
    )
    if any(
        not str(runtime.get(field) or "")
        for field in ("torch_version", "transformers_version", "peft_version")
    ):
        raise M2ConfigError("runtime package versions must be frozen")
    if (
        runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
    ):
        raise M2ConfigError("runtime must freeze deterministic CUDA algorithms")
    training = _require_exact_keys(
        config.get("training"),
        {
            "steps", "rollout_batch_size", "sft_batch_size", "learning_rate",
            "max_input_tokens", "generated_tokens", "sampling_distribution",
            "kl_coefficient", "sft_coefficient", "gradient_clip_norm",
            "checkpoint_every",
        },
        "training",
    )
    for field in ("steps", "rollout_batch_size", "sft_batch_size", "max_input_tokens", "checkpoint_every"):
        value = training.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise M2ConfigError(f"training.{field} must be a positive integer")
    if int(training["rollout_batch_size"]) < 3:
        raise M2ConfigError("independent leave-one-out rewards require rollout_batch_size >= 3")
    if training.get("generated_tokens") != GENERATED_TOKENS:
        raise M2ConfigError("the first genuine DFT path is fixed to 64 generated tokens")
    for field in (
        "learning_rate", "kl_coefficient",
        "sft_coefficient", "gradient_clip_norm",
    ):
        _require_positive(training.get(field), f"training.{field}")
    if training.get("sampling_distribution") != "raw_policy_categorical":
        raise M2ConfigError("score-function training requires raw on-policy categorical sampling")
    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) != 2 or any(
        not isinstance(arm, dict) for arm in arms
    ):
        raise M2ConfigError("arms must be exposure-matched A0=0 and nonzero A64")
    if (
        [arm.get("id") for arm in arms] != ARM_IDS
        or any(set(arm) != {"id", "mmd_coefficient"} for arm in arms)
        or isinstance(arms[0].get("mmd_coefficient"), bool)
        or not isinstance(arms[0].get("mmd_coefficient"), (int, float))
        or float(arms[0]["mmd_coefficient"]) != 0.0
        or _require_positive(arms[1].get("mmd_coefficient"), "arms.A64.mmd_coefficient") <= 0
    ):
        raise M2ConfigError("arms must be exposure-matched A0=0 and nonzero A64")
    stop = _require_exact_keys(
        config.get("stop"),
        {
            "max_kl", "min_unique_fraction", "max_repeated_trigram_fraction",
            "min_outline_fact_recall", "max_unsupported_claim_rate",
            "max_mean_abs_target_error",
        },
        "stop",
    )
    _require_positive(stop.get("max_kl"), "stop.max_kl")
    unique = _require_positive(stop.get("min_unique_fraction"), "stop.min_unique_fraction")
    if unique > 1:
        raise M2ConfigError("stop.min_unique_fraction cannot exceed 1")
    repeated = _require_positive(
        stop.get("max_repeated_trigram_fraction"), "stop.max_repeated_trigram_fraction"
    )
    if repeated > 1:
        raise M2ConfigError("stop.max_repeated_trigram_fraction cannot exceed 1")
    recall = _require_positive(
        stop.get("min_outline_fact_recall"), "stop.min_outline_fact_recall", allow_zero=True
    )
    unsupported = _require_positive(
        stop.get("max_unsupported_claim_rate"), "stop.max_unsupported_claim_rate",
        allow_zero=True,
    )
    _require_positive(stop.get("max_mean_abs_target_error"), "stop.max_mean_abs_target_error")
    if recall > 1 or unsupported > 1:
        raise M2ConfigError("factuality stop rates cannot exceed 1")
    readiness_trust = _require_exact_keys(
        config.get("readiness_trust"),
        {
            "trusted_public_keys_path", "trusted_public_keys_sha256",
            "protocol_signer_key_id", "blind_signer_key_id",
        },
        "readiness_trust",
    )
    if not Path(str(readiness_trust.get("trusted_public_keys_path") or "")).is_absolute():
        raise M2ConfigError("readiness trust store path must be absolute")
    _require_sha(
        readiness_trust.get("trusted_public_keys_sha256"),
        "readiness_trust.trusted_public_keys_sha256",
    )
    protocol_key = str(readiness_trust.get("protocol_signer_key_id") or "")
    blind_key = str(readiness_trust.get("blind_signer_key_id") or "")
    if not protocol_key or not blind_key or protocol_key == blind_key:
        raise M2ConfigError("readiness trust requires distinct protocol and blind signer keys")
    resume = _require_exact_keys(config.get("resume"), set(ARM_IDS), "resume")
    for arm_id in ARM_IDS:
        descriptor = resume[arm_id]
        if descriptor is None:
            continue
        descriptor = _require_exact_keys(
            descriptor,
            {
                "path", "adapter_model_sha256", "adapter_config_sha256",
                "training_state_sha256", "file_manifest_sha256",
            },
            f"resume.{arm_id}",
        )
        if not Path(str(descriptor.get("path") or "")).is_absolute():
            raise M2ConfigError(f"resume.{arm_id}.path must be absolute")
        for field in (
            "adapter_model_sha256", "adapter_config_sha256", "training_state_sha256",
            "file_manifest_sha256",
        ):
            _require_sha(descriptor.get(field), f"resume.{arm_id}.{field}")
    execution = _require_exact_keys(config.get("execution"), {"arm"}, "execution")
    if execution.get("arm") not in ARM_IDS:
        raise M2ConfigError("execution.arm must select exactly A0 or A64")
    workflow = _require_exact_keys(
        config.get("workflow"), {"protocol_version", "step", "method_contract_sha256"}, "workflow"
    )
    if workflow.get("protocol_version") != DFT_SCHEMA or workflow.get("step") != DFT_STEP:
        raise M2ConfigError("workflow must use the prospective score-function MMD protocol")
    expected = _require_sha(workflow.get("method_contract_sha256"), "workflow.method_contract_sha256")
    if canonical_hash(method_contract_payload(config)) != expected:
        raise M2ConfigError("M2 DFT method contract hash mismatch")
    return config


def _rbf_kernel(left: Any, right: Any, bandwidths: Sequence[float]) -> Any:
    import torch

    squared = torch.cdist(left, right).pow(2)
    kernels = [torch.exp(-squared / (2.0 * float(bandwidth))) for bandwidth in bandwidths]
    return torch.stack(kernels).mean(dim=0)


def mmd_score_components(
    generated: Any, humans: Any, bandwidths: Sequence[float]
) -> dict[str, Any]:
    """Conditional score terms for a shared-parameter policy.

    The generated/generated term has factor two because either member of an
    ordered pair can carry the policy score.  Averaging ``-reward * score``
    therefore estimates the gradient of MMD^2, excluding its human constant.
    """
    import torch

    if generated.ndim != 2 or humans.ndim != 2 or generated.shape[1] != humans.shape[1]:
        raise M2ConfigError("MMD embeddings must be compatible matrices")
    n = int(generated.shape[0])
    if n < 2 or int(humans.shape[0]) < 2:
        raise M2ConfigError("unbiased score-function MMD requires at least two samples per panel")
    xx = _rbf_kernel(generated, generated, bandwidths)
    xy = _rbf_kernel(generated, humans, bandwidths)
    generated_loo = (xx.sum(dim=1) - torch.diagonal(xx)) / (n - 1)
    human_mean = xy.mean(dim=1)
    generated_term = 2.0 * generated_loo
    human_term = 2.0 * human_mean
    return {
        "generated_pair_similarity": generated_term,
        "human_cross_similarity": human_term,
        "reward": -(generated_term - human_term),
    }


def mmd_score_rewards(generated: Any, humans: Any, bandwidths: Sequence[float]) -> Any:
    return mmd_score_components(generated, humans, bandwidths)["reward"]


def mmd_leave_one_out_baselines(
    generated: Any, humans: Any, bandwidths: Sequence[float]
) -> Any:
    """Controls independent of sample i, built only from the panel without i."""
    import torch

    if generated.ndim != 2 or humans.ndim != 2 or generated.shape[1] != humans.shape[1]:
        raise M2ConfigError("MMD embeddings must be compatible matrices")
    n = int(generated.shape[0])
    if n < 3 or int(humans.shape[0]) < 2:
        raise M2ConfigError("independent leave-one-out controls require three generated samples")
    xx = _rbf_kernel(generated, generated, bandwidths)
    xy = _rbf_kernel(generated, humans, bandwidths)
    controls = []
    for held_out in range(n):
        keep = torch.arange(n, device=generated.device) != held_out
        panel_xx = xx[keep][:, keep]
        panel_xy = xy[keep]
        panel_size = n - 1
        ordered_pair_mean = (
            panel_xx.sum() - torch.diagonal(panel_xx).sum()
        ) / (panel_size * (panel_size - 1))
        human_mean = panel_xy.mean()
        controls.append(-(2.0 * ordered_pair_mean - 2.0 * human_mean))
    return torch.stack(controls)


def score_function_loss(sequence_log_probs: Any, advantages: Any, coefficient: float) -> Any:
    if sequence_log_probs.shape != advantages.shape:
        raise M2ConfigError("score-function log probabilities and advantages must align")
    return -float(coefficient) * (advantages.detach() * sequence_log_probs).mean()


def per_sample_score_loss(
    sequence_log_probability: Any,
    advantage: Any,
    per_token_log_ratio: Any,
    *,
    mmd_coefficient: float,
    kl_coefficient: float,
    batch_size: int,
) -> Any:
    if batch_size <= 0:
        raise M2ConfigError("score loss batch size must be positive")
    return (
        -float(mmd_coefficient) * advantage.detach() * sequence_log_probability
        + float(kl_coefficient)
        * (per_token_log_ratio.detach() + 1.0 / GENERATED_TOKENS)
        * sequence_log_probability
    ) / batch_size


def deterministic_schedule(size: int, count: int, seed: int) -> list[int]:
    if size <= 0 or count <= 0:
        raise M2ConfigError("deterministic schedule requires positive sizes")
    generator = random.Random(int(seed))
    order: list[int] = []
    while len(order) < count:
        epoch = list(range(size))
        generator.shuffle(epoch)
        order.extend(epoch)
    return order[:count]


def deterministic_batches(size: int, batch_size: int, steps: int, seed: int) -> list[list[int]]:
    if size <= 0 or batch_size <= 0 or steps <= 0 or batch_size > size:
        raise M2ConfigError("deterministic batches require 0 < batch_size <= panel size")
    generator = random.Random(int(seed))
    return [generator.sample(range(size), batch_size) for _ in range(steps)]


def repeated_ngram_fraction(token_ids: Sequence[int], n: int = 3) -> float:
    if n <= 0:
        raise M2ConfigError("repetition n-gram size must be positive")
    windows = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
    if not windows:
        return 0.0
    return 1.0 - len(set(windows)) / len(windows)


def training_factual_adherence_sentinels(
    texts: list[str], records: list[dict[str, Any]]
) -> dict[str, float]:
    if len(texts) != len(records) or not texts:
        raise M2ConfigError("training sentinel texts and records must align")
    outlines = [row.get("outline") for row in records]
    targets = [int(row.get("target_length") or 0) for row in records]
    if any(target <= 0 for target in targets):
        raise M2ConfigError("training sentinel target lengths must be positive")
    return {
        "outline_fact_recall": float(outline_fact_recall(texts, outlines)),
        "unsupported_claim_rate": float(unsupported_claim_rate(texts, outlines)),
        "mean_abs_target_error": float(
            length_stats(texts, targets=targets)["mean_abs_target_error"]
        ),
    }


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise M2ConfigError(
                f"invalid {label} JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise M2ConfigError(f"{label} line {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise M2ConfigError(f"{label} is empty")
    return rows


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
        raise M2ConfigError(f"{label} artifact SHA-256 mismatch")


def _verify_a64_readiness(config: dict[str, Any]) -> str | None:
    if config["execution"]["arm"] != "A64":
        return None
    manifest_path = Path(os.environ.get(A64_READINESS_ENV, ""))
    expected_sha = _require_sha(
        os.environ.get(A64_READINESS_SHA_ENV), A64_READINESS_SHA_ENV
    )
    try:
        return verify_a64_readiness(
            config=config,
            readiness_path=manifest_path,
            readiness_sha256=expected_sha,
            base_model=BASE_MODEL,
            base_revision=BASE_REVISION,
            generated_tokens_per_rollout=GENERATED_TOKENS,
        )
    except ReadinessError as error:
        raise M2ConfigError(str(error)) from error


def _directory_file_map(root: Path, label: str) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise M2ConfigError(f"{label} must be a real directory")
    result: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise M2ConfigError(f"{label} cannot contain symlinks")
        if item.is_file():
            result[item.relative_to(root).as_posix()] = file_sha256(item)
    if not result:
        raise M2ConfigError(f"{label} is empty")
    return result


def _require_no_existing_files(root: Path, label: str) -> None:
    if any(item.is_file() or item.is_symlink() for item in root.rglob("*")):
        raise M2ConfigError(f"{label} already contains artifacts")


def _verify_inputs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float]]:
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
        raise M2ConfigError(
            f"runtime version mismatch: expected {expected_runtime} observed {observed_runtime}"
        )
    adapter = config["initial_adapter"]
    adapter_path = Path(adapter["path"])
    _verify_file(adapter_path / "adapter_model.safetensors", adapter["adapter_model_sha256"], "adapter model")
    _verify_file(adapter_path / "adapter_config.json", adapter["adapter_config_sha256"], "adapter config")
    manifest = _directory_file_map(adapter_path, "initial adapter")
    if canonical_hash(manifest) != adapter["file_manifest_sha256"]:
        raise M2ConfigError("initial adapter complete file manifest mismatch")
    try:
        adapter_config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise M2ConfigError(f"invalid adapter config: {exc.msg}") from exc
    if (
        not isinstance(adapter_config, dict)
        or adapter_config.get("base_model_name_or_path") != BASE_MODEL
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("peft_type") != "LORA"
    ):
        raise M2ConfigError("initial artifact is not the frozen 4B causal-LM LoRA adapter")
    data = config["data"]
    paths = [Path(data[name]) for name in ("rollout_path", "sft_anchor_path", "human_targets_path")]
    hashes = [data[name] for name in ("rollout_sha256", "sft_anchor_sha256", "human_targets_sha256")]
    for path, digest, label in zip(paths, hashes, ("rollout", "SFT anchor", "human target")):
        _verify_file(path, digest, label)
    rollout_records = _load_jsonl(paths[0], "rollout data")
    anchor_records = _load_jsonl(paths[1], "SFT anchor data")
    human_records = _load_jsonl(paths[2], "human targets")
    bandwidth_path = Path(config["kernel"]["bandwidths_path"])
    _verify_file(bandwidth_path, config["kernel"]["bandwidths_sha256"], "bandwidth")
    try:
        bandwidth_artifact = json.loads(bandwidth_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise M2ConfigError(f"invalid bandwidth artifact: {exc.msg}") from exc
    if (
        not isinstance(bandwidth_artifact, dict)
        or set(bandwidth_artifact) != {
            "artifact_schema", "source", "human_targets_sha256",
            "human_text_sequence_sha256",
            "representation_contract_sha256", "representation_execution_contract_sha256",
            "tokenizer_file_manifest_sha256", "source_adapter_file_manifest_sha256",
            "source_adapter_model_sha256", "source_adapter_config_sha256",
            "preparation_contract_sha256", "preparation_contract",
            "producer_run_id", "producer_git_sha", "producer_config_sha256", "producer_config",
            "model_base", "model_revision", "observed_runtime", "gpu", "observed_device_name",
            "derivation", "human_document_count", "embedding_dimension",
            "total_unordered_pair_count", "positive_pair_distance_count", "zero_distance_count",
            "median_positive_squared_distance", "embedding_matrix_sha256",
            "positive_distances_sha256", "parameterization", "values", "values_sha256",
        }
        or
        bandwidth_artifact.get("artifact_schema") != "dftr.m2.training_bandwidths.v2"
        or bandwidth_artifact.get("source") != "training_humans_only"
        or bandwidth_artifact.get("human_targets_sha256") != data["human_targets_sha256"]
        or bandwidth_artifact.get("representation_contract_sha256")
        != canonical_hash(config["representation"])
        or bandwidth_artifact.get("representation_execution_contract_sha256")
        != canonical_hash(representation_execution_payload(config))
        or bandwidth_artifact.get("tokenizer_file_manifest_sha256")
        != adapter["file_manifest_sha256"]
        or bandwidth_artifact.get("source_adapter_file_manifest_sha256")
        != adapter["file_manifest_sha256"]
        or bandwidth_artifact.get("source_adapter_model_sha256")
        != adapter["adapter_model_sha256"]
        or bandwidth_artifact.get("source_adapter_config_sha256")
        != adapter["adapter_config_sha256"]
        or bandwidth_artifact.get("human_text_sequence_sha256")
        != canonical_hash([
            row.get(data["human_text_field"]) for row in human_records
        ])
        or bandwidth_artifact.get("derivation") != TRAINING_BANDWIDTH_DERIVATION
        or bandwidth_artifact.get("parameterization") != TRAINING_BANDWIDTH_PARAMETERIZATION
        or bandwidth_artifact.get("human_document_count") != len(human_records)
        or bandwidth_artifact.get("model_base") != config["model"]["base"]
        or bandwidth_artifact.get("model_revision") != config["model"]["revision"]
        or bandwidth_artifact.get("observed_runtime") != expected_runtime
        or bandwidth_artifact.get("gpu") != config["compute"]["gpu"]
        or not str(bandwidth_artifact.get("observed_device_name") or "")
    ):
        raise M2ConfigError("bandwidth artifact is not bound to training humans only")
    for field in (
        "preparation_contract_sha256", "producer_config_sha256", "embedding_matrix_sha256",
        "positive_distances_sha256", "values_sha256",
    ):
        _require_sha(bandwidth_artifact.get(field), f"bandwidth.{field}")
    producer_config = bandwidth_artifact.get("producer_config")
    try:
        validate_prepare_dft_config(producer_config)
    except (TypeError, ValueError) as error:
        raise M2ConfigError(f"bandwidth producer config is invalid: {error}") from error
    if (
        canonical_hash(producer_config) != bandwidth_artifact["producer_config_sha256"]
        or preparation_contract_payload(producer_config)
        != bandwidth_artifact.get("preparation_contract")
        or canonical_hash(bandwidth_artifact["preparation_contract"])
        != bandwidth_artifact["preparation_contract_sha256"]
        or producer_config["initial_adapter"] != config["initial_adapter"]
        or producer_config["data"] != {
            "human_targets_path": data["human_targets_path"],
            "human_targets_sha256": data["human_targets_sha256"],
            "human_text_field": data["human_text_field"],
        }
        or producer_config["model"] != config["model"]
        or producer_config["representation"] != config["representation"]
        or producer_config["runtime"] != config["runtime"]
        or producer_config["compute"]["gpu"] != config["compute"]["gpu"]
    ):
        raise M2ConfigError("bandwidth producer contract does not match training")
    producer_git = str(bandwidth_artifact.get("producer_git_sha") or "")
    if len(producer_git) != 40 or any(character not in "0123456789abcdef" for character in producer_git):
        raise M2ConfigError("bandwidth producer Git SHA is invalid")
    if not SAFE_ID_RE.fullmatch(str(bandwidth_artifact.get("producer_run_id") or "")):
        raise M2ConfigError("bandwidth producer run ID is invalid")
    document_count = bandwidth_artifact.get("human_document_count")
    embedding_dimension = bandwidth_artifact.get("embedding_dimension")
    positive_pairs = bandwidth_artifact.get("positive_pair_distance_count")
    total_pairs = bandwidth_artifact.get("total_unordered_pair_count")
    zero_pairs = bandwidth_artifact.get("zero_distance_count")
    if (
        not isinstance(document_count, int) or isinstance(document_count, bool) or document_count < 2
        or not isinstance(embedding_dimension, int) or isinstance(embedding_dimension, bool)
        or embedding_dimension <= 0
        or not isinstance(positive_pairs, int) or isinstance(positive_pairs, bool)
        or not isinstance(total_pairs, int) or isinstance(total_pairs, bool)
        or not isinstance(zero_pairs, int) or isinstance(zero_pairs, bool)
        or total_pairs != document_count * (document_count - 1) // 2
        or positive_pairs != total_pairs
        or zero_pairs != 0
    ):
        raise M2ConfigError("bandwidth derivation dimensions or pair count are invalid")
    median_distance = _require_positive(
        bandwidth_artifact.get("median_positive_squared_distance"),
        "bandwidth median positive squared distance",
    )
    bandwidths = bandwidth_artifact.get("values")
    if not isinstance(bandwidths, list) or not bandwidths:
        raise M2ConfigError("bandwidth artifact has no values")
    values = [_require_positive(value, "bandwidth value") for value in bandwidths]
    expected_values = [
        median_distance * float(scale) ** 2
        for scale in TRAINING_BANDWIDTH_DERIVATION["scales"]
    ]
    if values != expected_values:
        raise M2ConfigError("bandwidth values do not reproduce the frozen derivation")
    if canonical_hash(values) != bandwidth_artifact["values_sha256"]:
        raise M2ConfigError("bandwidth values hash mismatch")
    return (
        rollout_records,
        anchor_records,
        human_records,
        values,
    )


def _verify_resume_artifact(
    descriptor: dict[str, Any], arm_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    import torch

    root = Path(descriptor["path"])
    _verify_file(root / "adapter_model.safetensors", descriptor["adapter_model_sha256"], "resume adapter model")
    _verify_file(root / "adapter_config.json", descriptor["adapter_config_sha256"], "resume adapter config")
    _verify_file(root / "training_state.pt", descriptor["training_state_sha256"], "resume training state")
    if canonical_hash(_directory_file_map(root, f"{arm_id} resume artifact")) != descriptor[
        "file_manifest_sha256"
    ]:
        raise M2ConfigError(f"{arm_id} resume complete file manifest mismatch")
    state = torch.load(root / "training_state.pt", map_location="cpu", weights_only=False)
    expected_keys = {
        "artifact_schema", "arm", "next_step", "method_contract_sha256",
        "source_adapter_manifest_sha256", "optimizer_state", "cpu_rng_state",
        "cuda_rng_state_all", "python_rng_state", "logs", "total_tokens",
    }
    if not isinstance(state, dict) or set(state) != expected_keys:
        raise M2ConfigError(f"{arm_id} resume training-state schema mismatch")
    next_step = state.get("next_step")
    if (
        state.get("artifact_schema") != "dftr.m2.training_state.v1"
        or state.get("arm") != arm_id
        or state.get("method_contract_sha256") != config["workflow"]["method_contract_sha256"]
        or state.get("source_adapter_manifest_sha256")
        != config["initial_adapter"]["file_manifest_sha256"]
        or not isinstance(next_step, int)
        or isinstance(next_step, bool)
        or not 0 < next_step <= int(config["training"]["steps"])
        or not isinstance(state.get("logs"), list)
        or len(state["logs"]) != next_step
        or state.get("total_tokens")
        != next_step * int(config["training"]["rollout_batch_size"]) * GENERATED_TOKENS
    ):
        raise M2ConfigError(f"{arm_id} resume training-state provenance mismatch")
    return state


def _save_training_checkpoint(
    policy: Any,
    optimizer: Any,
    target: Path,
    arm_id: str,
    next_step: int,
    logs: list[dict[str, Any]],
    total_tokens: int,
    config: dict[str, Any],
) -> None:
    import torch

    target.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(
        target, safe_serialization=True, selected_adapters=["default"]
    )
    state = {
        "artifact_schema": "dftr.m2.training_state.v1",
        "arm": arm_id,
        "next_step": next_step,
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "source_adapter_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "optimizer_state": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng_state": random.getstate(),
        "logs": logs,
        "total_tokens": total_tokens,
    }
    torch.save(state, target / "training_state.pt")


def _render_prompt(record: dict[str, Any], config: dict[str, Any]) -> str:
    required = (
        "user_prompt", "use_case", "style_kind", "style", "detail_mode",
        "target_length", "em_dashes_allowed", "outline",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise M2ConfigError(f"canonical full brief is missing fields: {', '.join(missing)}")
    user_prompt = str(record["user_prompt"]).strip()
    if not user_prompt:
        raise M2ConfigError("canonical full brief is missing user_prompt")
    outline = record["outline"]
    if not isinstance(outline, list):
        raise M2ConfigError("canonical full brief outline must be a list")
    target_length = int(record["target_length"])
    if target_length <= 0:
        raise M2ConfigError("canonical full brief target_length must be positive")
    brief = "\n".join((
        f"Writing request: {user_prompt}",
        f"Use case: {str(record['use_case']).strip()}",
        f"Style category: {str(record['style_kind']).strip()}",
        f"Style: {str(record['style']).strip()}",
        f"Detail mode: {str(record['detail_mode']).strip()}",
        f"Target length: about {target_length} words",
        f"Em dashes allowed: {'yes' if bool(record['em_dashes_allowed']) else 'no'}",
        "Grounding outline (use only these supported facts when non-empty): "
        + json.dumps(outline, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ))
    return str(config["data"]["prompt_format"]).format(brief=brief)


def _sample_raw_policy(
    model: Any, input_ids: Any, attention_mask: Any, new_tokens: int
) -> Any:
    """Sample the raw policy categorical distribution without generation warpers."""
    import torch

    sequences = input_ids
    running_mask = attention_mask
    next_input = input_ids
    past_key_values = None
    with torch.inference_mode():
        for _ in range(new_tokens):
            output = model(
                input_ids=next_input,
                attention_mask=running_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = output.logits[:, -1].float()
            if not torch.all(torch.isfinite(logits)):
                raise M2ConfigError("raw policy sampler produced non-finite logits")
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            sequences = torch.cat((sequences, next_token), dim=1)
            running_mask = torch.cat(
                (
                    running_mask,
                    torch.ones(
                        (running_mask.shape[0], 1),
                        dtype=running_mask.dtype,
                        device=running_mask.device,
                    ),
                ),
                dim=1,
            )
            next_input = next_token
            past_key_values = output.past_key_values
            if past_key_values is None:
                raise M2ConfigError("raw policy sampler requires an autoregressive cache")
    return sequences


def _sequence_log_probs(
    model: Any, sequences: Any, prompt_attention_mask: Any, prompt_width: int
) -> Any:
    import torch
    import torch.nn.functional as F

    continuation_mask = torch.ones(
        (sequences.shape[0], sequences.shape[1] - prompt_width),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    attention_mask = torch.cat((prompt_attention_mask, continuation_mask), dim=1)
    output = model(input_ids=sequences, attention_mask=attention_mask, return_dict=True)
    log_probs = F.log_softmax(output.logits[:, :-1].float(), dim=-1)
    labels = sequences[:, 1:]
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return selected[:, prompt_width - 1 :].sum(dim=1)


def _sft_anchor_loss(model: Any, tokenizer: Any, records: list[dict[str, Any]], config: dict[str, Any]) -> Any:
    import torch
    import torch.nn.functional as F

    prompts = [_render_prompt(record, config) for record in records]
    completion_field = str(config["data"]["completion_field"])
    completions = [str(record.get(completion_field) or "") for record in records]
    if any(not value for value in completions):
        raise M2ConfigError("SFT anchor record lacks completion")
    prompt_tokens = tokenizer(
        prompts, add_special_tokens=True, truncation=True,
        max_length=int(config["training"]["max_input_tokens"]),
    )["input_ids"]
    completion_tokens = tokenizer(
        completions, add_special_tokens=False, truncation=True, max_length=GENERATED_TOKENS,
    )["input_ids"]
    if any(not tokens for tokens in completion_tokens):
        raise M2ConfigError("SFT anchor completion is empty after tokenization")
    combined = [
        {"input_ids": prefix + suffix, "attention_mask": [1] * (len(prefix) + len(suffix))}
        for prefix, suffix in zip(prompt_tokens, completion_tokens)
    ]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        full = tokenizer.pad(combined, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = previous_padding_side
    labels = full["input_ids"].clone()
    for index, prefix in enumerate(prompt_tokens):
        labels[index, : len(prefix)] = -100
    labels[full["attention_mask"] == 0] = -100
    device = next(model.parameters()).device
    input_ids = full["input_ids"].to(device)
    labels = labels.to(device)
    logits = model(input_ids=input_ids, attention_mask=full["attention_mask"].to(device)).logits
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)


def _activate_adapter(model: Any, adapter_name: str, *, trainable: bool) -> list[Any]:
    model.set_adapter(adapter_name)
    marker = f".{adapter_name}."
    selected = []
    for name, parameter in model.named_parameters():
        enabled = bool(trainable and marker in name)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(parameter)
    if trainable and not selected:
        raise M2ConfigError(f"adapter {adapter_name} exposes no trainable parameters")
    return selected


def _load_arm_model(config: dict[str, Any], policy_adapter_path: str | None = None) -> tuple[Any, Any]:
    policy, tokenizer = load_source_peft_and_tokenizer(config, policy_adapter_path)
    policy.load_adapter(
        config["initial_adapter"]["path"], adapter_name="reference", is_trainable=False
    )
    _activate_adapter(policy, "default", trainable=True)
    policy.train()
    policy.config.use_cache = False
    return policy, tokenizer


def _run_arm(
    config: dict[str, Any], arm: dict[str, Any], checkpoint_dir: Path,
    rollout_records: list[dict[str, Any]], anchor_records: list[dict[str, Any]],
    human_records: list[dict[str, Any]], bandwidths: list[float],
    resume_descriptor: dict[str, Any] | None,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    seed = int(config["run"]["seed"])
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    arm_id = str(arm["id"])
    resume_state = (
        _verify_resume_artifact(resume_descriptor, arm_id, config)
        if resume_descriptor is not None
        else None
    )
    policy, tokenizer = _load_arm_model(
        config, policy_adapter_path=str(resume_descriptor["path"]) if resume_descriptor else None
    )
    trainable = _activate_adapter(policy, "default", trainable=True)
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=float(config["training"]["learning_rate"]))
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
    human_field = str(config["data"]["human_text_field"])
    human_texts = [str(row.get(human_field) or "") for row in human_records]
    if any(not text for text in human_texts):
        raise M2ConfigError("human target record lacks training text")
    human_embeddings = frozen_base_embeddings(policy, tokenizer, human_texts, config)
    steps = int(config["training"]["steps"])
    rollout_batch = int(config["training"]["rollout_batch_size"])
    sft_batch = int(config["training"]["sft_batch_size"])
    rollout_schedule = deterministic_batches(len(rollout_records), rollout_batch, steps, seed)
    anchor_schedule = deterministic_batches(len(anchor_records), sft_batch, steps, seed + 1)
    arm_dir = checkpoint_dir / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = list(resume_state["logs"]) if resume_state else []
    total_tokens = int(resume_state["total_tokens"]) if resume_state else 0
    start_step = int(resume_state["next_step"]) if resume_state else 0
    if resume_state is not None:
        random.setstate(resume_state["python_rng_state"])
        torch.set_rng_state(resume_state["cpu_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
    for step in range(start_step, steps):
        rollout_batch_rows = [
            rollout_records[index]
            for index in rollout_schedule[step]
        ]
        prompts = [_render_prompt(record, config) for record in rollout_batch_rows]
        encoded = tokenizer(
            prompts, padding=True, truncation=True,
            max_length=int(config["training"]["max_input_tokens"]), return_tensors="pt",
        )
        device = next(policy.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        rollout_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{step}".encode()).digest()[:8], "big"
        ) % (2**63)
        devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(rollout_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rollout_seed)
            policy.eval()
            sequences = _sample_raw_policy(
                policy, encoded["input_ids"], encoded["attention_mask"], GENERATED_TOKENS
            )
            policy.train()
        prompt_width = int(encoded["input_ids"].shape[1])
        if int(sequences.shape[1]) - prompt_width != GENERATED_TOKENS:
            raise M2ConfigError("rollout did not produce exactly 64 tokens")
        continuation = sequences[:, prompt_width:]
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        continuation_cpu = continuation.cpu()
        effective_rollout_count = len({tuple(row.tolist()) for row in continuation_cpu})
        unique_fraction = effective_rollout_count / rollout_batch
        repetition_fractions = [
            repeated_ngram_fraction(row.tolist()) for row in continuation_cpu
        ]
        factual_sentinels = training_factual_adherence_sentinels(
            texts, rollout_batch_rows
        )
        if unique_fraction < float(config["stop"]["min_unique_fraction"]):
            raise M2ConfigError("rollout effective-sample collapse")
        if max(repetition_fractions) > float(config["stop"]["max_repeated_trigram_fraction"]):
            raise M2ConfigError("rollout trigram-repetition collapse")
        if factual_sentinels["outline_fact_recall"] < float(
            config["stop"]["min_outline_fact_recall"]
        ):
            raise M2ConfigError("training-only outline factual-recall stop failed")
        if factual_sentinels["unsupported_claim_rate"] > float(
            config["stop"]["max_unsupported_claim_rate"]
        ):
            raise M2ConfigError("training-only unsupported-claim stop failed")
        if factual_sentinels["mean_abs_target_error"] > float(
            config["stop"]["max_mean_abs_target_error"]
        ):
            raise M2ConfigError("training-only target-length adherence stop failed")
        generated_embeddings = frozen_base_embeddings(policy, tokenizer, texts, config)
        reward_components = mmd_score_components(
            generated_embeddings, human_embeddings, bandwidths
        )
        rewards = reward_components["reward"]
        baselines = mmd_leave_one_out_baselines(
            generated_embeddings, human_embeddings, bandwidths
        )
        advantages = rewards - baselines
        policy.eval()
        with torch.inference_mode():
            sequence_log_probs = _sequence_log_probs(
                policy, sequences, encoded["attention_mask"], prompt_width
            )
        _activate_adapter(policy, "reference", trainable=False)
        policy.eval()
        with torch.inference_mode():
            reference_log_probs = _sequence_log_probs(
                policy, sequences, encoded["attention_mask"], prompt_width
            )
        trainable = _activate_adapter(policy, "default", trainable=True)
        policy.eval()
        log_ratio = (sequence_log_probs - reference_log_probs).detach() / GENERATED_TOKENS
        kl_estimate = float(log_ratio.mean().item())
        if max(0.0, kl_estimate) > float(config["stop"]["max_kl"]):
            raise M2ConfigError("KL-to-reference stop exceeded")
        kl_loss = ((log_ratio + 1.0 / GENERATED_TOKENS) * sequence_log_probs).mean()
        mmd_loss = score_function_loss(
            sequence_log_probs, advantages, float(arm["mmd_coefficient"])
        )
        score_loss = (
            mmd_loss + float(config["training"]["kl_coefficient"]) * kl_loss
        )
        if not torch.isfinite(score_loss):
            raise M2ConfigError("non-finite DFT score loss")
        optimizer.zero_grad(set_to_none=True)
        for sample_index in range(rollout_batch):
            sample_log_probability = _sequence_log_probs(
                policy,
                sequences[sample_index : sample_index + 1],
                encoded["attention_mask"][sample_index : sample_index + 1],
                prompt_width,
            )[0]
            sample_score_loss = per_sample_score_loss(
                sample_log_probability,
                advantages[sample_index],
                log_ratio[sample_index],
                mmd_coefficient=float(arm["mmd_coefficient"]),
                kl_coefficient=float(config["training"]["kl_coefficient"]),
                batch_size=rollout_batch,
            )
            if not torch.isfinite(sample_score_loss):
                raise M2ConfigError("non-finite per-sample DFT score loss")
            sample_score_loss.backward()
        policy.train()
        anchors = [
            anchor_records[index]
            for index in anchor_schedule[step]
        ]
        sft_loss = _sft_anchor_loss(policy, tokenizer, anchors, config)
        loss = (
            score_loss
            + float(config["training"]["sft_coefficient"]) * sft_loss
        )
        if not torch.isfinite(loss):
            raise M2ConfigError("non-finite DFT loss")
        (float(config["training"]["sft_coefficient"]) * sft_loss).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["training"]["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise M2ConfigError("non-finite DFT gradient")
        optimizer.step()
        contribution_variance = float((advantages.detach() * sequence_log_probs.detach()).var(unbiased=False).item())
        total_tokens += rollout_batch * GENERATED_TOKENS
        logs.append({
            "arm": arm["id"], "step": step, "rollout_seed": rollout_seed,
            "rollout_indices": rollout_schedule[step],
            "anchor_indices": anchor_schedule[step],
            "generated_tokens": rollout_batch * GENERATED_TOKENS,
            "rewards": [float(value) for value in rewards.cpu()],
            "generated_pair_similarity": [
                float(value) for value in reward_components["generated_pair_similarity"].cpu()
            ],
            "human_cross_similarity": [
                float(value) for value in reward_components["human_cross_similarity"].cpu()
            ],
            "leave_one_out_baselines": [float(value) for value in baselines.cpu()],
            "advantages": [float(value) for value in advantages.cpu()],
            "mmd_loss": float(mmd_loss.detach().item()), "kl_estimate": kl_estimate,
            "sft_loss": float(sft_loss.detach().item()), "total_loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm),
            "score_function_estimator_variance": contribution_variance,
            "score_function_estimator_variance_role": "diagnostic_only_not_a_stop",
            "effective_rollout_count": effective_rollout_count,
            "unique_fraction": unique_fraction,
            "repeated_trigram_fractions": repetition_fractions,
            "training_only_factual_adherence_sentinels": factual_sentinels,
        })
        if (step + 1) % int(config["training"]["checkpoint_every"]) == 0:
            _save_training_checkpoint(
                policy, optimizer, arm_dir / f"step-{step + 1}", arm_id,
                step + 1, logs, total_tokens, config,
            )
    policy.save_pretrained(
        arm_dir, safe_serialization=True, selected_adapters=["default"]
    )
    tokenizer.save_pretrained(arm_dir)
    write_jsonl(arm_dir / "training_steps.jsonl", logs)
    file_map = _directory_file_map(arm_dir, f"{arm['id']} output adapter")
    manifest = {
        "artifact_schema": "dftr.m2.adapter_native_checkpoint.v1",
        "arm": arm_id, "status": "completed", "adapter_native": True,
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "source_adapter_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "source_adapter_model_sha256": config["initial_adapter"]["adapter_model_sha256"],
        "source_adapter_config_sha256": config["initial_adapter"]["adapter_config_sha256"],
        "git_sha": git_sha(),
        "checkpoint_dir": str(arm_dir.resolve()), "file_sha256": file_map,
        "file_map_excludes": ["checkpoint_manifest.json"],
        "generated_tokens": total_tokens, "steps": steps,
        "resumed_from_step": start_step,
        "trainable_parameter_count": trainable_parameter_count,
        "mmd_coefficient": float(arm["mmd_coefficient"]),
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
    }
    write_json(arm_dir / "checkpoint_manifest.json", manifest)
    return manifest


def selected_arm(config: dict[str, Any]) -> dict[str, Any]:
    arm_id = str((config.get("execution") or {}).get("arm") or "")
    matches = [arm for arm in config.get("arms") or [] if arm.get("id") == arm_id]
    if len(matches) != 1:
        raise M2ConfigError("execution selector does not identify one frozen comparison arm")
    return matches[0]


def matched_exposure_payload(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    expected_tokens = int(training["steps"]) * int(training["rollout_batch_size"]) * GENERATED_TOKENS
    return {
        "artifact_schema": "dftr.m2.matched_exposure_contract.v1",
        "comparison_id": config["run"]["comparison_id"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "arms": {
            arm["id"]: {"mmd_coefficient": float(arm["mmd_coefficient"])}
            for arm in config["arms"]
        },
        "seed": int(config["run"]["seed"]),
        "rollout_sha256": config["data"]["rollout_sha256"],
        "sft_anchor_sha256": config["data"]["sft_anchor_sha256"],
        "steps": int(training["steps"]),
        "rollout_batch_size": int(training["rollout_batch_size"]),
        "sft_batch_size": int(training["sft_batch_size"]),
        "generated_tokens_per_rollout": GENERATED_TOKENS,
        "generated_tokens_per_arm": expected_tokens,
        "sampling_distribution": training["sampling_distribution"],
        "rollout_schedule": "python_random_sample_without_replacement.v1",
        "rollout_seed": int(config["run"]["seed"]),
        "anchor_seed": int(config["run"]["seed"]) + 1,
    }


def run_dft(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Train one selected arm under a selector-neutral matched comparison contract."""
    validate_dft_config(config)
    if not SAFE_ID_RE.fullmatch(str(run_id)):
        raise M2ConfigError("run_id is not a safe artifact identifier")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["runtime"]["cublas_workspace_config"]
    readiness_sha256 = _verify_a64_readiness(config)
    rollout, anchors, humans, bandwidths = _verify_inputs(config)
    if len(rollout) < int(config["training"]["rollout_batch_size"]):
        raise M2ConfigError("rollout panel is smaller than one without-replacement batch")
    if len(anchors) < int(config["training"]["sft_batch_size"]):
        raise M2ConfigError("SFT anchor panel is smaller than one without-replacement batch")
    if len(humans) < 2:
        raise M2ConfigError("human target panel requires at least two documents")
    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    _require_no_existing_files(checkpoint_dir, "checkpoint directory")
    if output_dir.resolve() != checkpoint_dir.resolve():
        existing_output_files = [
            item for item in output_dir.rglob("*")
            if (item.is_file() or item.is_symlink())
            and checkpoint_dir.resolve() not in item.parents
        ]
        if existing_output_files:
            raise M2ConfigError("run output directory already contains artifacts")
    write_json(checkpoint_dir / "config.json", config)
    arm = selected_arm(config)
    manifest = _run_arm(
        config, arm, checkpoint_dir, rollout, anchors, humans, bandwidths,
        config["resume"][arm["id"]],
    )
    expected_tokens = int(config["training"]["steps"]) * int(
        config["training"]["rollout_batch_size"]
    ) * GENERATED_TOKENS
    if manifest["generated_tokens"] != expected_tokens:
        raise M2ConfigError("executed arm rollout exposure accounting mismatch")
    exposure_contract = matched_exposure_payload(config)
    result = {
        "artifact_schema": "dftr.m2.score_function_mmd_result.v1",
        "run_id": run_id,
        "comparison_id": config["run"]["comparison_id"],
        "status": "completed",
        "executed_arm": arm["id"],
        "git_sha": git_sha(),
        "config_sha256": canonical_hash(config),
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "arm": manifest,
        "matched_exposure_contract": exposure_contract,
        "matched_exposure_contract_sha256": canonical_hash(exposure_contract),
        "a64_readiness_manifest_sha256": readiness_sha256,
        "token_accounting": {
            "executed_arm": arm["id"],
            "generated_tokens": expected_tokens,
            "total_tokens": expected_tokens,
        },
        "scientific_interpretation": "requires prospective adapter-native measurement-v2 comparison",
    }
    write_json(checkpoint_dir / "run_manifest.json", result)
    write_json(output_dir / "dft_manifest.json", result)
    return result
