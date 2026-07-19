"""Pure policy and append-only state helpers used by gateway and reaper."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Any


MONTHLY_GPU_CAP_USD = 100.0
MONTHLY_API_CAP_USD = 100.0
LOWER_VARIANCE_BRIEF_PROTOCOL = "dftr.lower_variance_briefs.two_provider.v1"
LOWER_VARIANCE_METADATA_MODEL = "qwen/qwen3-32b"
LOWER_VARIANCE_OUTLINE_MODEL = "openai/gpt-5-mini"
M3_REWRITE_TASK_PROTOCOL = "humanwrite.m3.rewrite_tasks.v1"
M3_SCIENTIFIC_REWRITE_PROTOCOL = "humanwrite.m3.scientific_api_rewrites.v2"
M3_BASELINE_VERIFY_PROTOCOL = "humanwrite.m3.baseline_draft_verification.v1"
M3_EVAL_REWRITE_PROTOCOL = "humanwrite.m3.eval_rewrite_inputs.v1"
M3_REWRITE_JUDGE_PROTOCOL = "humanwrite.m3.rewrite_judge.v2"
M3_REWRITE_GENERATOR_MODELS = [
    "google/gemini-3.1-flash-lite",
    "anthropic/claude-haiku-4.5",
]
M3_REWRITE_VERIFIER_BY_GENERATOR = {
    "google/gemini-3.1-flash-lite": "qwen/qwen3-32b",
    "anthropic/claude-haiku-4.5": "qwen/qwen3-32b",
}
M3_REWRITE_JUDGE_MODELS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite",
]
M3_REWRITE_JUDGE_PAIRWISE_DIMENSIONS = {
    "human_style": (
        "Choose the response that reads more like careful human-authored prose. "
        "Favor natural variation, specific development, and context-appropriate "
        "phrasing; penalize formulaic structure, canned transitions, repetition, "
        "and conspicuous model-like artifacts."
    ),
    "overall_quality": (
        "Choose the response with better writing quality while following the rewrite "
        "assignment. Favor clear, coherent, specific, engaging prose, and penalize "
        "generic, repetitive, awkward, unsupported, or instruction-violating writing."
    ),
}
M3_REWRITE_JUDGE_RANDOMIZATION = {
    "algorithm": "sha256(master_seed:model:dimension:fingerprint).parity.v1",
    "master_seed": 8903,
}
QWEN3_14B_MODEL = "Qwen/Qwen3-14B"
QWEN3_14B_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
M3_REWRITE_SFT_SMOKE_PROTOCOL = "humanwrite.m3.rewrite_sft_14b_smoke.v1"
M3_REWRITE_SFT_SMOKE_STEP = "train_m3_rewrite_sft_smoke"
M3_REWRITE_SFT_SMOKE_METHOD_KEYS = (
    "artifact_schema",
    "run",
    "compute",
    "model",
    "data",
    "lora",
    "training",
)
M3_BASELINE_DRAFT_PROTOCOL = "humanwrite.m3.baseline_drafts_14b.v1"
M3_BASELINE_DRAFT_PROTOCOL_V2 = "humanwrite.m3.baseline_drafts_14b.v2"
M3_BASELINE_DRAFT_STEP = "generate_m3_baseline_drafts"
M3_BASELINE_DRAFT_METHOD_KEYS = (
    "artifact_schema",
    "run",
    "compute",
    "model",
    "data",
    "generation",
)
M3_REWRITE_4K_PROTOCOL = "humanwrite.m3.rewrite_train_14b_4k.v1"
M3_REWRITE_4K_STEP = "train_m3_rewrite_4k"
M3_REWRITE_4K_METHOD_KEYS = (
    "artifact_schema",
    "run",
    "compute",
    "model",
    "representation",
    "data",
    "lora",
    "training",
    "objectives",
    "generation",
)
M3_REWRITE_GENERATION_PROTOCOL = "humanwrite.m3.rewrite_generate_14b.v1"
M3_REWRITE_GENERATION_STEP = "generate_m3_rewrite_candidates"
M3_REWRITE_GENERATION_METHOD_KEYS = (
    "artifact_schema",
    "run",
    "compute",
    "model",
    "data",
    "checkpoint",
    "generation",
)
M3_REWRITE_EMBEDDING_PROTOCOL = "humanwrite.m3.rewrite_embedding_score.v1"
M3_REWRITE_EMBEDDING_STEP = "score_m3_rewrite_embeddings"
M3_REWRITE_EMBEDDING_FAMILIES = {
    "bge-small-v1": {"model_id": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "prompt_name": None, "batch_size": 32, "max_tokens": 512},
    "nemotron-8b-v1": {"model_id": "nvidia/llama-embed-nemotron-8b", "revision": "aa3b43a495a9b280d1bdb716da37c54bb495d630", "prompt_name": "document", "batch_size": 4, "max_tokens": 512},
}
LOWER_VARIANCE_TRAIN_PROTOCOL = "dftr.m2.lower_variance_three_arm.v1"
LOWER_VARIANCE_CONFIRMATION_PROTOCOL = "dftr.m2.lower_variance_confirmation.v2"
LOWER_VARIANCE_TRAIN_PROTOCOLS = {
    LOWER_VARIANCE_TRAIN_PROTOCOL,
    LOWER_VARIANCE_CONFIRMATION_PROTOCOL,
}
LOWER_VARIANCE_METHOD_KEYS = (
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
BUDGET_CLASSES = {
    "smoke": {"max_seconds": 20 * 60, "max_gpus": 1},
    "screen": {"max_seconds": 2 * 60 * 60, "max_gpus": 1},
    "promo": {"max_seconds": 8 * 60 * 60, "max_gpus": 1},
}

# Modal public list prices retrieved 2026-07-15.  The 20% reserve covers CPU,
# memory, startup, and price drift.  The Modal dashboard hard cap is still the
# authoritative outer boundary.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
ALLOWED_COMMAND_PREFIX = ["python", "-m", "experiments.runner"]
TERMINAL = {"completed", "failed", "cancelled", "reaped", "launch_failed"}
UNRESOLVED_REVISION_PREFIX = "__M1_RESOLVE_"
REPLAY_TRANSFORMERS_VERSION = "4.57.6"
REPLAY_GENERATION_CONTRACT_PATH = "configs/m2/canonical_full_brief_generation_v1.json"
REPLAY_GENERATION_CONTRACT_SHA256 = (
    "db7c970440c451ffd21e634b53df3fa3d556b139e87257dfff7521442fe8f219"
)
REPLAY_HISTORICAL_CONFIG_PATH = (
    "configs/m1/m1_realdata_adherence_directional_qwen3_4b_three_seed_v1.yaml"
)
REPLAY_HISTORICAL_CONFIG_SHA256 = (
    "a02d893eda4c5e457864e1145e5cb4a4d238ab04037bc74e269a1ab20e52a72c"
)
REPLAY_PROTOCOLS = {
    "dftr.adapter_merge_replay.v1",
    "dftr.adapter_merge_replay.v2",
    "dftr.adapter_merge_replay.v3",
}
DFT_PROTOCOL = "dftr.m2.score_function_mmd.v1"
DFT_GENERATION_PROTOCOL = "dftr.m2.adapter_native_generation.v1"
LOWER_VARIANCE_GENERATION_PROTOCOL = "dftr.m2.lower_variance_generation.v1"
ESTIMATOR_AUDIT_PROTOCOL = "dftr.m2.frozen_estimator_audit.v1"
ESTIMATOR_AUDIT_SERIALIZER_SHA256 = (
    "4b8410439f74f7653061151ffd3335a1b77b2b34e3e2dbf279791e51378cbe28"
)
DFT_FULL_BRIEF_SERIALIZER_SHA256 = "eed171580857ef228cb83d8219fbd49926cda555c86a93a085f461714149d7ec"
LOWER_VARIANCE_FULL_BRIEF_SERIALIZER_SHA256 = (
    "1f92174518dfac375abbbbcf4ceba0659b726cabb215e0561a9fbffc4036b4a1"
)
PREPARE_DFT_PROTOCOL = "dftr.m2.prepare_training_bandwidths.v1"
PREPARE_DFT_SUPPORTED_GPUS = {"L40S", "A100-80GB", "H100"}
PREPARE_DFT_METHOD_KEYS = (
    "artifact_schema", "run", "compute", "model", "initial_adapter", "data",
    "representation", "derivation", "runtime", "output",
)
PREPARE_DFT_DERIVATION = {
    "algorithm": "median_positive_unordered_human_pairwise_squared_distance.v1",
    "scales": [0.25, 0.5, 1.0, 2.0, 4.0],
    "pair_scope": "all_unordered_training_human_pairs",
    "zero_distance_policy": "fail",
    "degenerate_policy": "fail",
    "distance_dtype": "float64_cpu",
}
DFT_METHOD_KEYS = (
    "artifact_schema", "run", "compute", "model", "initial_adapter", "data",
    "representation", "kernel", "runtime", "training", "arms", "stop", "readiness_trust",
)


def _is_checkpoint_volume_path(path: Path) -> bool:
    """Return whether an absolute path resolves strictly below /checkpoints."""
    if not path.is_absolute():
        return False
    try:
        checkpoint_root = Path("/checkpoints").resolve(strict=False)
        resolved = path.resolve(strict=False)
        return resolved != checkpoint_root and resolved.is_relative_to(checkpoint_root)
    except (OSError, RuntimeError, ValueError):
        return False


def validate_dft_generation_launch_config(
    config: dict[str, Any], *, backend: str, budget_class: str, task_kind: str
) -> None:
    """Reject generation drift before any accelerator reservation is created."""
    run, compute, model = config.get("run") or {}, config.get("compute") or {}, config.get("model") or {}
    checkpoint, data = config.get("checkpoint") or {}, config.get("data") or {}
    sampling, runtime = config.get("sampling") or {}, config.get("runtime") or {}
    workflow = config.get("workflow") or {}
    exact_sets = (
        (config, {"artifact_schema", "run", "compute", "model", "checkpoint", "data", "sampling", "runtime", "output", "workflow"}),
        (run, {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"}),
        (compute, {"gpu", "gpus", "timeout_min"}),
        (model, {"base", "revision", "torch_dtype"}),
        (checkpoint, {"path", "manifest_sha256", "adapter_model_sha256", "arm", "method_contract_sha256"}),
        (data, {"prompt_briefs_path", "prompt_briefs_sha256", "prompt_format", "prompt_schema_version", "prompt_serializer_sha256"}),
        (sampling, {"training_seed", "sampling_seed", "seed_scope", "prompt_order", "distribution", "batch_size", "new_tokens", "max_input_tokens", "decode"}),
        (runtime, {"torch_version", "transformers_version", "peft_version", "deterministic_algorithms", "cublas_workspace_config"}),
        (workflow, {"protocol_version", "step", "generation_contract_sha256", "decoding_policy_sha256"}),
    )
    if any(not isinstance(value, dict) or set(value) != keys for value, keys in exact_sets):
        raise PolicyError("generate_dft exact schema mismatch")
    sha_fields = (
        checkpoint.get("manifest_sha256"), checkpoint.get("adapter_model_sha256"),
        checkpoint.get("method_contract_sha256"), data.get("prompt_briefs_sha256"),
        workflow.get("generation_contract_sha256"), workflow.get("decoding_policy_sha256"),
    )
    checkpoint_path, prompt_path = Path(str(checkpoint.get("path") or "")), Path(str(data.get("prompt_briefs_path") or ""))
    fixed_sampling = {
        "training_seed": 11, "sampling_seed": 101,
        "seed_scope": "single_global_rng_stream", "prompt_order": "sorted_prompt_id",
        "distribution": "raw_policy_categorical", "batch_size": 4,
        "new_tokens": 64, "max_input_tokens": 1024,
        "decode": {"skip_special_tokens": True},
    }
    generation_contract = {
        "artifact_schema": "dftr.m2.generation_contract.v1",
        "method_contract_sha256": checkpoint.get("method_contract_sha256"),
        "prompt_schema_version": data.get("prompt_schema_version"),
        "prompt_briefs_sha256": data.get("prompt_briefs_sha256"),
        "prompt_serializer_sha256": data.get("prompt_serializer_sha256"),
        "prompt_format": data.get("prompt_format"),
        "prompt_order": sampling.get("prompt_order"),
        "sampling_seed": sampling.get("sampling_seed"),
        "seed_scope": sampling.get("seed_scope"),
        "distribution": sampling.get("distribution"),
        "batch_size": sampling.get("batch_size"),
        "new_tokens": sampling.get("new_tokens"),
        "max_input_tokens": sampling.get("max_input_tokens"),
        "decode": sampling.get("decode"),
    }
    decoding_policy = {
        "artifact_schema": "dftr.m2.decoding_policy.v1",
        "distribution": sampling.get("distribution"), "raw_logits": True,
        "warpers": [], "stopping": "exact_new_token_count",
        "new_tokens": sampling.get("new_tokens"), "decode": sampling.get("decode"),
    }
    if (
        config.get("artifact_schema") != DFT_GENERATION_PROTOCOL
        or workflow.get("protocol_version") != DFT_GENERATION_PROTOCOL
        or workflow.get("step") != "generate_dft"
        or task_kind != "experiment" or budget_class != "screen" or backend != "modal"
        or run.get("task_kind") != "experiment" or run.get("budget_class") != "screen"
        or run.get("command") != ALLOWED_COMMAND_PREFIX or run.get("seed") != 101
        or checkpoint.get("arm") not in {"A0", "A64"}
        or run.get("arm") != f"{checkpoint.get('arm')}-generation"
        or model != {"base": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "torch_dtype": "bfloat16"}
        or str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1 or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int) or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= 120
        or not _is_checkpoint_volume_path(checkpoint_path) or not _is_checkpoint_volume_path(prompt_path)
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(value or "")) for value in sha_fields)
        or data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:"
        or data.get("prompt_schema_version") != "dft.full-brief.v1"
        or data.get("prompt_serializer_sha256") != DFT_FULL_BRIEF_SERIALIZER_SHA256
        or sampling != fixed_sampling
        or runtime != {"torch_version": "2.13.0+cu130", "transformers_version": "4.57.6", "peft_version": "0.19.1", "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"}
        or config.get("output") != {"filename": "outputs.jsonl", "overwrite": False}
        or workflow.get("generation_contract_sha256") != canonical_hash(generation_contract)
        or workflow.get("decoding_policy_sha256") != canonical_hash(decoding_policy)
    ):
        raise PolicyError("generate_dft frozen contract mismatch")


def validate_lower_variance_generation_launch_config(
    config: dict[str, Any], *, backend: str, budget_class: str, task_kind: str
) -> None:
    """Reject held-out generation drift before any accelerator reservation is created."""
    run, compute, model = config.get("run") or {}, config.get("compute") or {}, config.get("model") or {}
    checkpoint, data = config.get("checkpoint") or {}, config.get("data") or {}
    sampling, runtime = config.get("sampling") or {}, config.get("runtime") or {}
    workflow = config.get("workflow") or {}
    exact_sets = (
        (config, {"artifact_schema", "run", "compute", "model", "checkpoint", "data", "sampling", "runtime", "output", "workflow"}),
        (run, {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"}),
        (compute, {"gpu", "gpus", "timeout_min"}),
        (model, {"base", "revision", "torch_dtype"}),
        (checkpoint, {"path", "manifest_sha256", "adapter_model_sha256", "arm", "method_contract_sha256"}),
        (data, {"prompt_briefs_path", "prompt_briefs_sha256", "prompt_format", "prompt_schema_version", "prompt_serializer_sha256"}),
        (sampling, {"training_seed", "sampling_seed", "seed_scope", "prompt_order", "distribution", "batch_size", "new_tokens", "max_input_tokens", "decode"}),
        (runtime, {"torch_version", "transformers_version", "peft_version", "deterministic_algorithms", "cublas_workspace_config"}),
        (workflow, {"protocol_version", "step", "generation_contract_sha256", "decoding_policy_sha256"}),
    )
    if any(not isinstance(value, dict) or set(value) != keys for value, keys in exact_sets):
        raise PolicyError("generate_lower_variance exact schema mismatch")
    sha_fields = (
        checkpoint.get("manifest_sha256"), checkpoint.get("adapter_model_sha256"),
        checkpoint.get("method_contract_sha256"), data.get("prompt_briefs_sha256"),
        workflow.get("generation_contract_sha256"), workflow.get("decoding_policy_sha256"),
    )
    checkpoint_path = Path(str(checkpoint.get("path") or ""))
    prompt_path = Path(str(data.get("prompt_briefs_path") or ""))
    fixed_sampling = {
        "training_seed": 11, "sampling_seed": 101,
        "seed_scope": "single_global_rng_stream", "prompt_order": "sorted_prompt_id",
        "distribution": "raw_policy_categorical", "batch_size": 4,
        "new_tokens": 128, "max_input_tokens": 1024,
        "decode": {"skip_special_tokens": True},
    }
    generation_contract = {
        "artifact_schema": "dftr.m2.generation_contract.v1",
        "method_contract_sha256": checkpoint.get("method_contract_sha256"),
        "prompt_schema_version": data.get("prompt_schema_version"),
        "prompt_briefs_sha256": data.get("prompt_briefs_sha256"),
        "prompt_serializer_sha256": data.get("prompt_serializer_sha256"),
        "prompt_format": data.get("prompt_format"),
        "prompt_order": sampling.get("prompt_order"),
        "sampling_seed": sampling.get("sampling_seed"),
        "seed_scope": sampling.get("seed_scope"),
        "distribution": sampling.get("distribution"),
        "batch_size": sampling.get("batch_size"),
        "new_tokens": sampling.get("new_tokens"),
        "max_input_tokens": sampling.get("max_input_tokens"),
        "decode": sampling.get("decode"),
    }
    decoding_policy = {
        "artifact_schema": "dftr.m2.decoding_policy.v1",
        "distribution": sampling.get("distribution"), "raw_logits": True,
        "warpers": [], "stopping": "exact_new_token_count",
        "new_tokens": sampling.get("new_tokens"), "decode": sampling.get("decode"),
    }
    if (
        config.get("artifact_schema") != LOWER_VARIANCE_GENERATION_PROTOCOL
        or workflow.get("protocol_version") != LOWER_VARIANCE_GENERATION_PROTOCOL
        or workflow.get("step") != "generate_lower_variance"
        or task_kind != "experiment" or budget_class != "screen" or backend != "modal"
        or run.get("task_kind") != "experiment" or run.get("budget_class") != "screen"
        or run.get("command") != ALLOWED_COMMAND_PREFIX or run.get("seed") != 101
        or checkpoint.get("arm") not in {"SFT", "MMD_WITNESS"}
        or run.get("arm") != f"{checkpoint.get('arm')}-generation"
        or model != {"base": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "torch_dtype": "bfloat16"}
        or str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1 or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int) or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= 120
        or not _is_checkpoint_volume_path(checkpoint_path) or not _is_checkpoint_volume_path(prompt_path)
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(value or "")) for value in sha_fields)
        or data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:"
        or data.get("prompt_schema_version") != "dft.full-brief.tokens.v1"
        or data.get("prompt_serializer_sha256") != LOWER_VARIANCE_FULL_BRIEF_SERIALIZER_SHA256
        or sampling != fixed_sampling
        or runtime != {"torch_version": "2.13.0+cu130", "transformers_version": "4.57.6", "peft_version": "0.19.1", "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"}
        or config.get("output") != {"filename": "outputs.jsonl", "overwrite": False}
        or workflow.get("generation_contract_sha256") != canonical_hash(generation_contract)
        or workflow.get("decoding_policy_sha256") != canonical_hash(decoding_policy)
    ):
        raise PolicyError("generate_lower_variance frozen contract mismatch")
REPLAY_PROTOCOL_V1 = "dftr.adapter_merge_replay.v1"
REPLAY_PROTOCOL_V2 = "dftr.adapter_merge_replay.v2"
REPLAY_PROTOCOL_V3 = "dftr.adapter_merge_replay.v3"
REPLAY_COMPARISON_V1 = "M2-adapter-merge-fidelity-replay-v1"
REPLAY_COMPARISON_V2 = "M2-adapter-merge-fidelity-replay-v2"
REPLAY_COMPARISON_V3 = "M2-adapter-merge-fidelity-replay-v3"
REPLAY_CANONICAL_V1_CONFIG_HASH = (
    "859798f2ce66b81a2db32665b7f8fda5a76f5d9e82c64789e7e1f797c4587b9f"
)
REPLAY_CANONICAL_V2_CONFIG_HASH = (
    "ee76ca0ecda72321f07cecd1c70fba5905779321e3169579e357bafdad4cd1da"
)
REPLAY_CANONICAL_V3_CONFIG_HASH = (
    "82ef89e5f78f205083392ad2a74f3a4795debc5856cd7ce5f7fe906f728fd6b9"
)
REPLAY_SNAPSHOT_IDENTITY_PATH = (
    "configs/m2/manifests/m2_adapter_merge_snapshot_identity_v2.json"
)
REPLAY_SNAPSHOT_IDENTITY_SHA256 = (
    "602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c"
)
REPLAY_ORIGINAL_MERGE_HASH_V2 = "7f095c31e83f8b03"
REPLAY_SUBMITTED_SNAPSHOT_HASH_V2 = "0f437f62bc1cca0c"
REPLAY_V1_MERGED_CONTENT_HASH = REPLAY_SUBMITTED_SNAPSHOT_HASH_V2
REPLAY_EXACT_SERIALIZATION_IDENTITY = "exact_serialization_bytes"
REPLAY_TOKENIZER_IDENTITY_PATH = (
    "configs/m2/manifests/m2_adapter_merge_tokenizer_identity_v3.json"
)
REPLAY_TOKENIZER_IDENTITY_SHA256 = (
    "54891d4320ee45db4f4ad08124c22b1696410b70210e63f0da5239e3958a7712"
)
REPLAY_ADAPTER_TOKENIZER_CONFIG_SHA256_V3 = (
    "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
)
REPLAY_MERGED_TOKENIZER_CONFIG_SHA256_V3 = (
    "a32ee532e3437966f2b52bb0fe0e7c525234dc1034814718b0467d8104a09371"
)
REPLAY_SENSITIVE_KEY_WORDS = {
    "api", "provider", "judge", "sealed", "hidden", "private",
    "credential", "credentials", "secret", "secrets",
    "auth", "authentication", "authorization", "key", "keys",
    "endpoint", "endpoints", "service", "services",
}
REPLAY_TOKEN_KEY_WORDS = {"token", "tokens"}
REPLAY_PUBLIC_TOKEN_METADATA_WORDS = {
    "add", "added", "additional", "all", "begin", "between", "bos", "cls",
    "count", "counts",
    "decoder", "eos", "exact", "forced", "generation", "greedy", "id", "ids",
    "image", "input", "map", "mask", "max", "min", "new", "output", "pad",
    "parity", "policy", "sep", "skip", "special", "split", "spaces", "start",
    "suppress", "teacher", "token", "tokens", "type", "types", "unk", "video",
    "vision", "extended", "extra", "healing",
}
REPLAY_PUBLIC_TOKEN_METADATA_MODIFIERS = (
    REPLAY_PUBLIC_TOKEN_METADATA_WORDS - REPLAY_TOKEN_KEY_WORDS
)
REPLAY_KEY_WORD_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+"
)


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class LaunchPolicy:
    comparison_id: str
    budget_class: str
    timeout_seconds: int
    gpu: str
    worst_case_cost_usd: float
    task_kind: str = "experiment"
    api_reserved_cost_usd: float = 0.0


def canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def revision_is_unresolved(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.startswith(UNRESOLVED_REVISION_PREFIX)


def replay_key_words(key: Any) -> tuple[str, ...]:
    words: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", str(key)):
        words.extend(
            match.casefold() for match in REPLAY_KEY_WORD_RE.findall(chunk)
        )
    return tuple(words)


def replay_key_is_sensitive(key: Any) -> bool:
    words = replay_key_words(key)
    word_set = set(words)
    if word_set & REPLAY_SENSITIVE_KEY_WORDS:
        return True
    # OAuth/OIDC uses ``id_token``; model metadata uses the inverse
    # ``*_token_id`` order (for example ``eos_token_id``).
    if words == ("id", "token"):
        return True
    if not word_set & REPLAY_TOKEN_KEY_WORDS:
        return False
    return not (
        word_set <= REPLAY_PUBLIC_TOKEN_METADATA_WORDS
        and bool(word_set & REPLAY_PUBLIC_TOKEN_METADATA_MODIFIERS)
    )


def forbidden_replay_surface_keys(value: Any) -> list[str]:
    """Find paid/private surface aliases recursively in a replay config."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if replay_key_is_sensitive(key):
                found.append(str(key))
            found.extend(forbidden_replay_surface_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(forbidden_replay_surface_keys(child))
    return found


def validate_replay_launch_contract(config: dict[str, Any]) -> None:
    """Bind replay schema, comparison, and artifact identity before launch."""
    run = config.get("run") or {}
    workflow = config.get("workflow") or {}
    artifacts = config.get("artifacts") or {}
    protocol = str(workflow.get("protocol_version") or "")
    comparison = str(run.get("comparison_id") or "")
    if protocol == REPLAY_PROTOCOL_V1:
        if (
            comparison != REPLAY_COMPARISON_V1
            or artifacts.get("merged_content_hash") != REPLAY_V1_MERGED_CONTENT_HASH
            or canonical_hash(config) != REPLAY_CANONICAL_V1_CONFIG_HASH
        ):
            raise PolicyError(
                "replay v1 is restricted to the exact canonical historical config identity"
            )
        return
    if protocol == REPLAY_PROTOCOL_V2:
        audit = config.get("submitted_snapshot_audit") or {}
        if (
            comparison != REPLAY_COMPARISON_V2
            or artifacts.get("merged_content_hash") != REPLAY_ORIGINAL_MERGE_HASH_V2
            or audit.get("identity_manifest") != REPLAY_SNAPSHOT_IDENTITY_PATH
            or audit.get("identity_manifest_sha256") != REPLAY_SNAPSHOT_IDENTITY_SHA256
            or audit.get("canonical_directory_hash") != REPLAY_SUBMITTED_SNAPSHOT_HASH_V2
            or list(audit.get("metadata_difference_files") or [])
            != ["generation_config.json", "train_config.json"]
            or audit.get("weights_tokenizer_index_identity")
            != REPLAY_EXACT_SERIALIZATION_IDENTITY
            or audit.get("generation_arguments_authority")
            != REPLAY_GENERATION_CONTRACT_PATH
            or canonical_hash(config) != REPLAY_CANONICAL_V2_CONFIG_HASH
        ):
            raise PolicyError(
                "replay v2 requires the exact canonical prospective config and artifact identity"
            )
        return
    if protocol != REPLAY_PROTOCOL_V3 or comparison != REPLAY_COMPARISON_V3:
        raise PolicyError("replay protocol and comparison identity must match bidirectionally")
    snapshot_audit = config.get("submitted_snapshot_audit") or {}
    tokenizer_audit = config.get("adapter_merge_tokenizer_audit") or {}
    if (
        artifacts.get("merged_content_hash") != REPLAY_ORIGINAL_MERGE_HASH_V2
        or snapshot_audit.get("identity_manifest") != REPLAY_SNAPSHOT_IDENTITY_PATH
        or snapshot_audit.get("identity_manifest_sha256")
        != REPLAY_SNAPSHOT_IDENTITY_SHA256
        or snapshot_audit.get("canonical_directory_hash")
        != REPLAY_SUBMITTED_SNAPSHOT_HASH_V2
        or list(snapshot_audit.get("metadata_difference_files") or [])
        != ["generation_config.json", "train_config.json"]
        or snapshot_audit.get("weights_tokenizer_index_identity")
        != REPLAY_EXACT_SERIALIZATION_IDENTITY
        or snapshot_audit.get("generation_arguments_authority")
        != REPLAY_GENERATION_CONTRACT_PATH
        or tokenizer_audit.get("identity_manifest") != REPLAY_TOKENIZER_IDENTITY_PATH
        or tokenizer_audit.get("identity_manifest_sha256")
        != REPLAY_TOKENIZER_IDENTITY_SHA256
        or tokenizer_audit.get("adapter_tokenizer_config_sha256")
        != REPLAY_ADAPTER_TOKENIZER_CONFIG_SHA256_V3
        or tokenizer_audit.get("merged_tokenizer_config_sha256")
        != REPLAY_MERGED_TOKENIZER_CONFIG_SHA256_V3
        or list(tokenizer_audit.get("tokenizer_metadata_difference_files") or [])
        != ["tokenizer_config.json"]
        or tokenizer_audit.get("shared_file_identity")
        != REPLAY_EXACT_SERIALIZATION_IDENTITY
        or tokenizer_audit.get("runtime_attestation")
        != "exact_prompt_token_and_attention_mask_before_diagnostics"
        or canonical_hash(config) != REPLAY_CANONICAL_V3_CONFIG_HASH
    ):
        raise PolicyError(
            "replay v3 requires the exact canonical tokenizer-aware config and artifact identity"
        )


def validate_estimator_audit_launch_config(
    config: dict[str, Any], *, backend: str, budget_class: str, task_kind: str
) -> None:
    """Fail closed before reserving a GPU for the frozen estimator audit."""
    run = config.get("run") or {}
    compute = config.get("compute") or {}
    adapter = config.get("initial_adapter") or {}
    data = config.get("data") or {}
    representation = config.get("representation") or {}
    audit = config.get("audit") or {}
    runtime = config.get("runtime") or {}
    workflow = config.get("workflow") or {}
    exact_sets = (
        (config, {
            "artifact_schema", "run", "compute", "model", "initial_adapter", "data",
            "representation", "audit", "runtime", "output", "workflow",
        }),
        (run, {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"}),
        (compute, {"gpu", "gpus", "timeout_min"}),
        (adapter, {"path", "adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"}),
        (data, {
            "rollout_path", "rollout_sha256", "human_targets_path", "human_targets_sha256",
            "human_text_field", "prompt_format", "prompt_schema_version",
            "prompt_serializer_sha256", "legacy_target_length_semantics",
        }),
        (representation, {"model", "revision", "layer", "pooling", "normalize", "role", "batch_size", "max_tokens"}),
        (audit, {
            "replicates", "group_sizes", "max_new_tokens", "rollout_target_length_tokens",
            "max_input_tokens", "logprob_microbatch_size", "prompt_schedule_seed", "rollout_seed_start",
            "gradient_supports", "sketch_dimension", "sketch_seed", "bandwidth_scales",
            "go_thresholds",
        }),
        (runtime, {"torch_version", "transformers_version", "peft_version", "deterministic_algorithms", "cublas_workspace_config"}),
        (workflow, {"protocol_version", "step", "audit_contract_sha256"}),
    )
    if any(not isinstance(value, dict) or set(value) != keys for value, keys in exact_sets):
        raise PolicyError("audit_estimator exact schema mismatch")
    sha_values = (
        adapter.get("adapter_model_sha256"), adapter.get("adapter_config_sha256"),
        adapter.get("file_manifest_sha256"), data.get("rollout_sha256"),
        data.get("human_targets_sha256"), workflow.get("audit_contract_sha256"),
    )
    paths = (
        Path(str(adapter.get("path") or "")),
        Path(str(data.get("rollout_path") or "")),
        Path(str(data.get("human_targets_path") or "")),
    )
    contract_payload = {
        key: config.get(key)
        for key in (
            "artifact_schema", "run", "compute", "model", "initial_adapter", "data",
            "representation", "audit", "runtime", "output",
        )
    } | {
        "protocol_version": workflow.get("protocol_version"),
        "step": workflow.get("step"),
    }
    if (
        config.get("artifact_schema") != ESTIMATOR_AUDIT_PROTOCOL
        or workflow.get("protocol_version") != ESTIMATOR_AUDIT_PROTOCOL
        or workflow.get("step") != "audit_estimator"
        or canonical_hash(contract_payload) != workflow.get("audit_contract_sha256")
        or task_kind != "experiment"
        or budget_class != "screen"
        or run != {
            "comparison_id": "M2-frozen-estimator-audit-4b-v2",
            "arm": "frozen-estimator-audit",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ALLOWED_COMMAND_PREFIX,
            "seed": 11,
        }
        or str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1
        or compute.get("timeout_min") != 120
        or config.get("model") != {
            "base": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "torch_dtype": "bfloat16",
        }
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(value or "")) for value in sha_values)
        or data.get("human_text_field") != "completion"
        or data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:"
        or data.get("prompt_schema_version") != "dft.full-brief.v2"
        or data.get("prompt_serializer_sha256") != ESTIMATOR_AUDIT_SERIALIZER_SHA256
        or data.get("legacy_target_length_semantics")
        != "provider_requested_token_estimate_missing_unit_field"
        or representation != {
            "model": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "layer": -1,
            "pooling": "attention_masked_mean",
            "normalize": True,
            "role": "diagnostic_training_only_not_evaluation",
            "batch_size": 4,
            "max_tokens": 256,
        }
        or audit != {
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
        }
        or runtime != {
            "torch_version": "2.13.0+cu130",
            "transformers_version": "4.57.6",
            "peft_version": "0.19.1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        }
        or config.get("output") != {"filename": "estimator_audit.json", "overwrite": False}
        or any(not path.is_absolute() for path in paths)
        or any(any(part in {"harness", "measurement_v2"} for part in path.parts) for path in paths[1:])
        or backend == "modal" and any(not _is_checkpoint_volume_path(path) for path in paths)
    ):
        raise PolicyError("audit_estimator frozen contract mismatch")


def validate_launch(payload: dict[str, Any], *, backend: str = "modal") -> LaunchPolicy:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise PolicyError("config must be an object")
    if payload.get("config_hash") != canonical_hash(config):
        raise PolicyError("config hash mismatch")
    run = config.get("run") or {}
    comparison_id = str(run.get("comparison_id", ""))
    if not comparison_id:
        raise PolicyError("config.run.comparison_id is required")
    prereg = payload.get("preregistration") or {}
    if (
        prereg.get("kind") != "prereg"
        or prereg.get("status") != "open"
        or prereg.get("comparison") != comparison_id
    ):
        raise PolicyError("open matching preregistration required")
    budget_class = str(payload.get("budget_class", ""))
    if budget_class not in BUDGET_CLASSES:
        raise PolicyError("unknown budget class")
    if run.get("budget_class") not in (None, budget_class):
        raise PolicyError("budget class mismatch")
    compute = config.get("compute") or {}
    if int(compute.get("gpus", 1)) != 1:
        raise PolicyError("single-GPU rule violated")
    timeout_seconds = int(
        compute.get("timeout_min", BUDGET_CLASSES[budget_class]["max_seconds"] // 60)
    ) * 60
    if timeout_seconds <= 0 or timeout_seconds > BUDGET_CLASSES[budget_class]["max_seconds"]:
        raise PolicyError("requested timeout exceeds budget class")
    task_kind = str(run.get("task_kind", "experiment"))
    if task_kind not in {
        "experiment",
        "brief_synthesis",
        "document_cleaning",
        "rewrite_synthesis",
        "rewrite_judging",
        "model_cache",
        "source_materialization",
    }:
        raise PolicyError("unsupported task_kind")
    gpu = str(compute.get("gpu", "L40S")).upper() if task_kind == "experiment" else "CPU"
    if task_kind == "experiment" and gpu not in GPU_USD_PER_SECOND:
        raise PolicyError(f"unsupported GPU: {gpu}")
    base_model = str((config.get("model") or {}).get("base", ""))
    workflow_step = str((config.get("workflow") or {}).get("step", "")).casefold()
    if "14B" in base_model.upper() and not payload.get("human_scaleup_approved"):
        raise PolicyError("14B scale-up lacks human approval")
    if workflow_step in {
        "train_sft", "sample_sweep", "merge_adapter", "replay_equivalence", "train_dft",
        "prepare_dft", "generate_dft", "generate_lower_variance", "audit_estimator", "train_lower_variance",
        M3_REWRITE_SFT_SMOKE_STEP,
        M3_BASELINE_DRAFT_STEP,
        M3_REWRITE_4K_STEP,
        M3_REWRITE_GENERATION_STEP,
    } and revision_is_unresolved(
        (config.get("model") or {}).get("revision")
    ):
        raise PolicyError(
            "evidentiary experiment jobs require model.revision to be a resolved immutable revision"
        )
    if workflow_step == "train_dft" and (
        str((config.get("workflow") or {}).get("protocol_version")) != DFT_PROTOCOL
        or task_kind != "experiment"
    ):
        raise PolicyError("train_dft requires the frozen M2 DFT experiment protocol")
    if workflow_step == "train_dft":
        workflow = config.get("workflow") or {}
        method_payload = {key: config.get(key) for key in DFT_METHOD_KEYS}
        method_payload.update(
            protocol_version=workflow.get("protocol_version"), step=workflow.get("step")
        )
        if canonical_hash(method_payload) != workflow.get("method_contract_sha256"):
            raise PolicyError("train_dft method contract hash mismatch")
        execution_arm = str((config.get("execution") or {}).get("arm") or "")
        readiness = payload.get("dft_a64_readiness")
        if execution_arm == "A0":
            if readiness is not None:
                raise PolicyError("A0 cannot consume an A64 readiness attestation")
        elif execution_arm == "A64":
            if backend != "modal":
                raise PolicyError("A64 launches require the independently configured Modal gateway")
            expected_keys = {
                "kind", "status", "comparison", "method_contract_sha256",
                "manifest_path", "manifest_sha256",
            }
            if not isinstance(readiness, dict) or set(readiness) != expected_keys:
                raise PolicyError("A64 requires an exact wrapper readiness attestation")
            readiness_sha = str(readiness.get("manifest_sha256") or "")
            if (
                readiness.get("kind") != "dft_a64_readiness"
                or readiness.get("status") != "ready"
                or readiness.get("comparison") != comparison_id
                or readiness.get("method_contract_sha256")
                != (config.get("workflow") or {}).get("method_contract_sha256")
                or len(readiness_sha) != 64
                or any(character not in "0123456789abcdef" for character in readiness_sha)
            ):
                raise PolicyError("A64 wrapper readiness attestation mismatch")
            readiness_path = Path(str(readiness.get("manifest_path") or ""))
            if not readiness_path.is_absolute():
                raise PolicyError("A64 readiness manifest path must be absolute")
            if backend == "modal" and not str(readiness_path).startswith("/checkpoints/"):
                raise PolicyError("Modal A64 readiness must use the checkpoint volume")
            configured_trust_sha = os.environ.get("DFTR_M2_TRUSTED_KEYS_SHA256", "")
            if (
                not re.fullmatch(r"[0-9a-f]{64}", configured_trust_sha)
                or (config.get("readiness_trust") or {}).get("trusted_public_keys_sha256")
                != configured_trust_sha
            ):
                raise PolicyError("A64 trust store is not independently configured by the gateway")
        else:
            raise PolicyError("train_dft execution arm must be A0 or A64")
    if workflow_step == "prepare_dft":
        workflow = config.get("workflow") or {}
        expected_top = set(PREPARE_DFT_METHOD_KEYS) | {"workflow"}
        method_payload = {key: config.get(key) for key in PREPARE_DFT_METHOD_KEYS}
        method_payload.update(
            protocol_version=workflow.get("protocol_version"), step=workflow.get("step")
        )
        prepare_run = config.get("run") or {}
        prepare_compute = config.get("compute") or {}
        prepare_adapter = config.get("initial_adapter") or {}
        prepare_data = config.get("data") or {}
        prepare_representation = config.get("representation") or {}
        prepare_runtime = config.get("runtime") or {}
        adapter_path = Path(str((config.get("initial_adapter") or {}).get("path") or ""))
        human_path = Path(str((config.get("data") or {}).get("human_targets_path") or ""))
        if (
            set(config) != expected_top
            or config.get("artifact_schema") != PREPARE_DFT_PROTOCOL
            or workflow.get("protocol_version") != PREPARE_DFT_PROTOCOL
            or set(workflow) != {"protocol_version", "step", "preparation_contract_sha256"}
            or canonical_hash(method_payload) != workflow.get("preparation_contract_sha256")
            or task_kind != "experiment"
            or budget_class != "smoke"
            or set(prepare_run) != {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"}
            or prepare_run.get("arm") != "training-bandwidths"
            or prepare_run.get("seed") != 0
            or prepare_run.get("budget_class") != "smoke"
            or prepare_run.get("task_kind") != "experiment"
            or prepare_run.get("command") != ALLOWED_COMMAND_PREFIX
            or set(prepare_compute) != {"gpu", "gpus", "timeout_min"}
            or str(prepare_compute.get("gpu") or "").upper()
            not in PREPARE_DFT_SUPPORTED_GPUS
            or prepare_compute.get("gpus") != 1
            or isinstance(prepare_compute.get("gpus"), bool)
            or not isinstance(prepare_compute.get("timeout_min"), int)
            or isinstance(prepare_compute.get("timeout_min"), bool)
            or not 0 < prepare_compute["timeout_min"] <= 20
            or config.get("model") != {
                "base": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "torch_dtype": "bfloat16",
            }
            or set(prepare_adapter) != {
                "path", "adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"
            }
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(prepare_adapter.get(field) or ""))
                for field in ("adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256")
            )
            or set(prepare_data) != {"human_targets_path", "human_targets_sha256", "human_text_field"}
            or not re.fullmatch(r"[0-9a-f]{64}", str(prepare_data.get("human_targets_sha256") or ""))
            or not str(prepare_data.get("human_text_field") or "")
            or set(prepare_representation) != {
                "model", "revision", "layer", "pooling", "normalize", "role", "batch_size", "max_tokens"
            }
            or prepare_representation.get("model") != "Qwen/Qwen3-4B"
            or prepare_representation.get("revision") != "1cfa9a7208912126459214e8b04321603b3df60c"
            or prepare_representation.get("layer") != -1
            or prepare_representation.get("pooling") != "attention_masked_mean"
            or prepare_representation.get("normalize") is not True
            or prepare_representation.get("role") != "training_only_not_measurement_v2"
            or any(
                not isinstance(prepare_representation.get(field), int)
                or isinstance(prepare_representation.get(field), bool)
                or prepare_representation[field] <= 0
                for field in ("batch_size", "max_tokens")
            )
            or config.get("derivation") != PREPARE_DFT_DERIVATION
            or set(prepare_runtime) != {
                "torch_version", "transformers_version", "peft_version",
                "deterministic_algorithms", "cublas_workspace_config",
            }
            or any(
                not str(prepare_runtime.get(field) or "")
                for field in ("torch_version", "transformers_version", "peft_version")
            )
            or prepare_runtime.get("deterministic_algorithms") is not True
            or prepare_runtime.get("cublas_workspace_config") != ":4096:8"
            or config.get("output") != {"filename": "training_bandwidths.json", "overwrite": False}
            or payload.get("dft_a64_readiness") is not None
            or not adapter_path.is_absolute()
            or not human_path.is_absolute()
            or any(part in {"harness", "measurement_v2"} for part in human_path.parts)
        ):
            raise PolicyError("prepare_dft requires the exact frozen wrapper-only protocol")
        if backend == "modal" and (
            not _is_checkpoint_volume_path(adapter_path)
            or not _is_checkpoint_volume_path(human_path)
        ):
            raise PolicyError("Modal prepare_dft inputs must use the checkpoint volume")
    if workflow_step == "generate_dft":
        validate_dft_generation_launch_config(
            config, backend=backend, budget_class=budget_class, task_kind=task_kind
        )
    if workflow_step == "generate_lower_variance":
        validate_lower_variance_generation_launch_config(
            config, backend=backend, budget_class=budget_class, task_kind=task_kind
        )
    if workflow_step == "audit_estimator":
        validate_estimator_audit_launch_config(
            config, backend=backend, budget_class=budget_class, task_kind=task_kind
        )
        if payload.get("dft_a64_readiness") is not None:
            raise PolicyError("audit_estimator cannot consume an A64 readiness attestation")
    if workflow_step == "train_lower_variance":
        workflow = config.get("workflow") or {}
        method_payload = {key: config.get(key) for key in LOWER_VARIANCE_METHOD_KEYS}
        method_payload.update(
            protocol_version=workflow.get("protocol_version"), step=workflow.get("step")
        )
        data = config.get("data") or {}
        initial_adapter = config.get("initial_adapter") or {}
        if (
            config.get("artifact_schema") not in LOWER_VARIANCE_TRAIN_PROTOCOLS
            or workflow.get("protocol_version") != config.get("artifact_schema")
            or set(workflow) != {"protocol_version", "step", "method_contract_sha256"}
            or canonical_hash(method_payload) != workflow.get("method_contract_sha256")
            or task_kind != "experiment"
            or budget_class not in {"smoke", "screen"}
            or config.get("model")
            != {
                "base": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "torch_dtype": "bfloat16",
            }
            or (config.get("execution") or {}).get("arm")
            not in {"SFT", "TOKEN_MOMENT", "MMD_WITNESS"}
            or data.get("prompt_schema_version") != "dft.full-brief.tokens.v1"
            or not str(data.get("anchor_path") or "").startswith("/checkpoints/")
            or not str(data.get("witness_generated_path") or "").startswith("/checkpoints/")
            or not str(initial_adapter.get("path") or "").startswith("/checkpoints/")
            or payload.get("dft_a64_readiness") is not None
        ):
            raise PolicyError("train_lower_variance requires the frozen wrapper protocol")
    if workflow_step == M3_REWRITE_SFT_SMOKE_STEP:
        workflow = config.get("workflow") or {}
        method_payload = {
            key: config.get(key) for key in M3_REWRITE_SFT_SMOKE_METHOD_KEYS
        }
        method_payload.update(
            protocol_version=workflow.get("protocol_version"), step=workflow.get("step")
        )
        data = config.get("data") or {}
        if (
            set(config) != set(M3_REWRITE_SFT_SMOKE_METHOD_KEYS) | {"workflow"}
            or config.get("artifact_schema") != M3_REWRITE_SFT_SMOKE_PROTOCOL
            or workflow.get("protocol_version") != M3_REWRITE_SFT_SMOKE_PROTOCOL
            or set(workflow) != {"protocol_version", "step", "method_contract_sha256"}
            or canonical_hash(method_payload) != workflow.get("method_contract_sha256")
            or task_kind != "experiment"
            or budget_class != "smoke"
            or config.get("model")
            != {
                "base": QWEN3_14B_MODEL,
                "revision": QWEN3_14B_REVISION,
                "torch_dtype": "bfloat16",
            }
            or config.get("compute") != {"gpu": "H100", "gpus": 1, "timeout_min": 20}
            or (config.get("run") or {}).get("arm") != "SFT14-mechanical-smoke"
            or data.get("source_records") != 128
            or data.get("rewrite_records") != 96
            or not str(data.get("source_briefs_path") or "").startswith("/checkpoints/")
            or not str(data.get("rewrite_tasks_path") or "").startswith("/checkpoints/")
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(data.get(field) or ""))
                for field in ("source_briefs_sha256", "rewrite_tasks_sha256")
            )
            or payload.get("dft_a64_readiness") is not None
        ):
            raise PolicyError("M3 rewrite SFT smoke requires the frozen 14B wrapper protocol")
    if workflow_step == M3_BASELINE_DRAFT_STEP:
        workflow = config.get("workflow") or {}
        method_payload = {
            key: config.get(key) for key in M3_BASELINE_DRAFT_METHOD_KEYS
        }
        method_payload.update(
            protocol_version=workflow.get("protocol_version"), step=workflow.get("step")
        )
        data = config.get("data") or {}
        protocol = config.get("artifact_schema")
        expected_budget = "promo" if protocol == M3_BASELINE_DRAFT_PROTOCOL_V2 else "screen"
        expected_timeout = 300 if protocol == M3_BASELINE_DRAFT_PROTOCOL_V2 else 120
        if (
            set(config) != set(M3_BASELINE_DRAFT_METHOD_KEYS) | {"workflow"}
            or protocol not in {M3_BASELINE_DRAFT_PROTOCOL, M3_BASELINE_DRAFT_PROTOCOL_V2}
            or workflow.get("protocol_version") != protocol
            or set(workflow) != {"protocol_version", "step", "method_contract_sha256"}
            or canonical_hash(method_payload) != workflow.get("method_contract_sha256")
            or task_kind != "experiment"
            or budget_class != expected_budget
            or config.get("model")
            != {
                "base": QWEN3_14B_MODEL,
                "revision": QWEN3_14B_REVISION,
                "torch_dtype": "bfloat16",
            }
            or config.get("compute") != {"gpu": "H100", "gpus": 1, "timeout_min": expected_timeout}
            or (config.get("run") or {}).get("arm") != "base-draft-construction"
            or data.get("source_records") != 4096
            or data.get("target_records") != 819
            or not str(data.get("source_path") or "").startswith("/checkpoints/")
            or not str(data.get("output_path") or "").startswith("/checkpoints/")
            or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("source_sha256") or ""))
            or payload.get("dft_a64_readiness") is not None
        ):
            raise PolicyError("M3 baseline drafts require the frozen 14B wrapper protocol")
    if workflow_step == M3_REWRITE_4K_STEP:
        workflow = config.get("workflow") or {}
        method_payload = {
            key: config.get(key) for key in M3_REWRITE_4K_METHOD_KEYS
        }
        method_payload.update(
            protocol_version=workflow.get("protocol_version"), step=workflow.get("step")
        )
        data = config.get("data") or {}
        representation = config.get("representation") or {}
        if (
            set(config) != set(M3_REWRITE_4K_METHOD_KEYS) | {"workflow"}
            or config.get("artifact_schema") != M3_REWRITE_4K_PROTOCOL
            or workflow.get("protocol_version") != M3_REWRITE_4K_PROTOCOL
            or set(workflow) != {"protocol_version", "step", "method_contract_sha256"}
            or canonical_hash(method_payload) != workflow.get("method_contract_sha256")
            or task_kind != "experiment"
            or budget_class != "promo"
            or config.get("model")
            != {
                "base": QWEN3_14B_MODEL,
                "revision": QWEN3_14B_REVISION,
                "torch_dtype": "bfloat16",
            }
            or config.get("compute") != {"gpu": "H100", "gpus": 1, "timeout_min": 240}
            or (config.get("run") or {}).get("arm") not in {"SFT14", "HUMANWRITE14"}
            or data.get("records") != 4096
            or data.get("stage_records") != 2048
            or not str(data.get("corpus_path") or "").startswith("/checkpoints/")
            or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("corpus_sha256") or ""))
            or representation
            != {
                "embedder": "BAAI/bge-small-en-v1.5",
                "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                "max_length": 512,
                "witness_subset": 512,
            }
            or payload.get("dft_a64_readiness") is not None
        ):
            raise PolicyError("M3 4K rewrite training requires the frozen 14B wrapper protocol")
    if workflow_step == M3_REWRITE_GENERATION_STEP:
        workflow = config.get("workflow") or {}
        method_payload = {
            key: config.get(key) for key in M3_REWRITE_GENERATION_METHOD_KEYS
        }
        method_payload.update(
            protocol_version=workflow.get("protocol_version"), step=workflow.get("step")
        )
        data = config.get("data") or {}
        checkpoint = config.get("checkpoint") or {}
        arm = str(checkpoint.get("arm") or "")
        trained = arm in {"SFT14", "HUMANWRITE14"}
        if (
            set(config) != set(M3_REWRITE_GENERATION_METHOD_KEYS) | {"workflow"}
            or config.get("artifact_schema") != M3_REWRITE_GENERATION_PROTOCOL
            or workflow.get("protocol_version") != M3_REWRITE_GENERATION_PROTOCOL
            or set(workflow) != {"protocol_version", "step", "method_contract_sha256"}
            or canonical_hash(method_payload) != workflow.get("method_contract_sha256")
            or task_kind != "experiment"
            or budget_class != "screen"
            or config.get("model")
            != {
                "base": QWEN3_14B_MODEL,
                "revision": QWEN3_14B_REVISION,
                "torch_dtype": "bfloat16",
            }
            or config.get("compute") != {"gpu": "H100", "gpus": 1, "timeout_min": 120}
            or arm not in {"BASE", "SFT14", "HUMANWRITE14"}
            or (config.get("run") or {}).get("arm") != f"{arm}-fresh-rewrite-generation"
            or data.get("panel_path")
            != "/checkpoints/data/m3-rewriting-14b-v1/fresh-rewrite-eval-panel-256-v1.jsonl"
            or data.get("records") != 256
            or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("panel_sha256") or ""))
            or set(checkpoint)
            != {"arm", "training_manifest_path", "training_manifest_sha256"}
            or (
                trained
                and (
                    not str(checkpoint.get("training_manifest_path") or "").startswith("/checkpoints/")
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(checkpoint.get("training_manifest_sha256") or "")
                    )
                )
            )
            or (
                not trained
                and (
                    checkpoint.get("training_manifest_path") is not None
                    or checkpoint.get("training_manifest_sha256") is not None
                )
            )
            or payload.get("dft_a64_readiness") is not None
        ):
            raise PolicyError("M3 rewrite generation requires the frozen 14B wrapper protocol")
    if workflow_step == M3_REWRITE_EMBEDDING_STEP:
        workflow = config.get("workflow") or {}
        data = config.get("data") or {}
        representation = config.get("representation") or {}
        if (
            set(config) != {"artifact_schema", "run", "compute", "data", "representation", "output", "workflow"}
            or config.get("artifact_schema") != M3_REWRITE_EMBEDDING_PROTOCOL
            or workflow != {"protocol_version": M3_REWRITE_EMBEDDING_PROTOCOL, "step": M3_REWRITE_EMBEDDING_STEP}
            or task_kind != "experiment"
            or budget_class != "screen"
            or config.get("compute") != {"gpu": "L40S", "gpus": 1, "timeout_min": 120}
            or (config.get("run") or {}).get("arm") != "HUMANWRITE14-vs-SFT14-two-family-MMD"
            or representation != {
                "families": M3_REWRITE_EMBEDDING_FAMILIES,
                "normalization": "explicit_float32_l2.v1",
                "bandwidth_source": "two_disjoint_128-document_unused-human-floors",
                "bandwidth_scales": [0.25, 0.5, 1.0, 2.0, 4.0],
            }
            or data.get("clean_pool_records") != 640
            or data.get("panel_records") != 256
            or any(not str(data.get(field) or "").startswith("/checkpoints/") for field in ("clean_pool_path", "panel_path", "sft_path", "treatment_path"))
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(data.get(field) or "")) for field in ("clean_pool_sha256", "panel_sha256", "sft_sha256", "treatment_sha256"))
            or config.get("output") != {"report_filename": "rewrite_embedding_score.json", "embedding_filename_template": "{family_id}.npz"}
        ):
            raise PolicyError("M3 rewrite embedding score requires the frozen two-family protocol")
    if workflow_step == "replay_equivalence":
        replay_workflow = config.get("workflow") or {}
        protocol_version = str(replay_workflow.get("protocol_version"))
        if protocol_version not in REPLAY_PROTOCOLS:
            raise PolicyError("replay_equivalence requires the frozen public replay protocol")
        if task_kind != "experiment":
            raise PolicyError("replay_equivalence requires the credential-free experiment task kind")
        forbidden = forbidden_replay_surface_keys(config)
        if forbidden:
            raise PolicyError("replay_equivalence cannot expose paid or hidden surfaces")
        if str((config.get("runtime") or {}).get("transformers_version")) != REPLAY_TRANSFORMERS_VERSION:
            raise PolicyError("replay_equivalence requires the exact frozen Transformers version")
        expected_bindings = {
            "generation_contract": REPLAY_GENERATION_CONTRACT_PATH,
            "generation_contract_sha256": REPLAY_GENERATION_CONTRACT_SHA256,
            "historical_sampling_config": REPLAY_HISTORICAL_CONFIG_PATH,
            "historical_sampling_config_sha256": REPLAY_HISTORICAL_CONFIG_SHA256,
        }
        if any(str(replay_workflow.get(key) or "") != value for key, value in expected_bindings.items()):
            raise PolicyError("replay_equivalence requires canonical frozen contract bindings")
        validate_replay_launch_contract(config)
    api_reserved = 0.0
    if task_kind == "experiment":
        command = run.get("command", ALLOWED_COMMAND_PREFIX)
        if not isinstance(command, list) or command[:3] != ALLOWED_COMMAND_PREFIX:
            raise PolicyError("command is outside the allowlist")
        worst = round(timeout_seconds * GPU_USD_PER_SECOND[gpu] * 1.20, 6)
    elif task_kind == "model_cache":
        if set(config) != {"run", "compute", "model"}:
            raise PolicyError("model_cache exact schema mismatch")
        if (
            budget_class != "promo"
            or compute != {"gpus": 1, "timeout_min": 240}
            or config.get("model")
            != {
                "base": QWEN3_14B_MODEL,
                "revision": QWEN3_14B_REVISION,
            }
        ):
            raise PolicyError("model_cache requires the pinned Qwen3-14B CPU contract")
        worst = 0.0
    elif task_kind == "brief_synthesis":
        data = config.get("data") or {}
        api = config.get("api") or {}
        for field in ("input_uri", "output_uri"):
            if not str(data.get(field, "")).startswith("modal-volume://humanwrite-checkpoints/"):
                raise PolicyError(f"brief_synthesis data.{field} must use the checkpoint volume")
        input_sha = str(data.get("input_sha256", ""))
        if len(input_sha) != 64 or any(character not in "0123456789abcdef" for character in input_sha):
            raise PolicyError("brief_synthesis requires lowercase data.input_sha256")
        max_records = int(data.get("max_records", 0))
        if max_records <= 0 or max_records > 50_000:
            raise PolicyError("brief_synthesis data.max_records must be between 1 and 50000")
        if api.get("protocol") == LOWER_VARIANCE_BRIEF_PROTOCOL:
            if (
                set(config) != {"run", "compute", "data", "api"}
                or set(data) != {"input_uri", "output_uri", "input_sha256", "max_records"}
                or set(api)
                != {"protocol", "metadata_model", "outline_model", "max_cost_usd"}
            ):
                raise PolicyError("lower-variance brief exact schema mismatch")
            if (
                api.get("metadata_model") != LOWER_VARIANCE_METADATA_MODEL
                or api.get("outline_model") != LOWER_VARIANCE_OUTLINE_MODEL
            ):
                raise PolicyError("lower-variance brief provider models are frozen")
            if data.get("input_uri") == data.get("output_uri"):
                raise PolicyError("lower-variance brief input and output URIs must be distinct")
            if max_records not in {128, 1024, 4096, 16384, 46080} or max_records % 4:
                raise PolicyError(
                    "lower-variance brief corpus must use a preregistered scale"
                )
            if budget_class != "promo":
                raise PolicyError("lower-variance brief synthesis requires promo budget")
        else:
            if not str(api.get("model", "")):
                raise PolicyError("brief_synthesis requires a frozen api.model")
            if api.get("prompt_repair_only") is True:
                if api.get("force_empty_quotations") is True:
                    raise PolicyError("prompt repair cannot also run quote-free recovery")
                if data.get("input_uri") == data.get("output_uri"):
                    raise PolicyError("prompt repair input and output URIs must be distinct")
                if max_records > 320:
                    raise PolicyError("prompt repair is limited to the frozen 320-record corpus")
            if api.get("force_empty_quotations") is True:
                max_missing = int((config.get("recovery") or {}).get("max_missing_records", 0))
                if budget_class != "smoke" or not 1 <= max_missing <= 16:
                    raise PolicyError(
                        "quote-free recovery requires smoke budget and 1..16 max missing records"
                    )
        api_reserved = float(api.get("max_cost_usd", 0.0))
        if api_reserved <= 0 or api_reserved > MONTHLY_API_CAP_USD:
            raise PolicyError("brief_synthesis requires api.max_cost_usd within the monthly cap")
        worst = 0.0
    elif task_kind == "document_cleaning":
        data = config.get("data") or {}
        api = config.get("api") or {}
        quality = config.get("quality") or {}
        if set(config) != {"run", "compute", "data", "api", "quality"}:
            raise PolicyError("document_cleaning exact schema mismatch")
        if set(data) != {"input_uri", "output_uri", "input_sha256", "max_records", "target_records"}:
            raise PolicyError("document_cleaning data schema mismatch")
        if set(api) != {"model", "max_cost_usd"} or set(quality) != {"min_word_count", "max_word_count"}:
            raise PolicyError("document_cleaning API or quality schema mismatch")
        if any(
            not str(data.get(field) or "").startswith("modal-volume://humanwrite-checkpoints/")
            for field in ("input_uri", "output_uri")
        ) or data.get("input_uri") == data.get("output_uri"):
            raise PolicyError("document_cleaning requires distinct checkpoint-volume paths")
        if not re.fullmatch(r"[0-9a-f]{64}", str(data.get("input_sha256") or "")):
            raise PolicyError("document_cleaning requires lowercase data.input_sha256")
        records = data.get("max_records")
        if not isinstance(records, int) or isinstance(records, bool) or not 1 <= records <= 50_000:
            raise PolicyError("document_cleaning max_records must be between 1 and 50000")
        target_records = data.get("target_records")
        if (
            not isinstance(target_records, int)
            or isinstance(target_records, bool)
            or not 1 <= target_records <= records
        ):
            raise PolicyError("document_cleaning target_records must be within max_records")
        if api.get("model") != "qwen/qwen3-32b":
            raise PolicyError("document_cleaning requires qwen/qwen3-32b")
        allowed_quality_bounds = (
            {"min_word_count": 80, "max_word_count": 220},
            {"min_word_count": 40, "max_word_count": 500},
        )
        if quality not in allowed_quality_bounds:
            raise PolicyError("document_cleaning requires frozen word-count bounds")
        api_reserved = float(api.get("max_cost_usd") or 0.0)
        if api_reserved <= 0 or api_reserved > MONTHLY_API_CAP_USD:
            raise PolicyError("document_cleaning requires api.max_cost_usd within the monthly cap")
        worst = 0.0
    elif task_kind == "rewrite_synthesis":
        data = config.get("data") or {}
        api = config.get("api") or {}
        tokenizer = config.get("tokenizer") or {}
        if set(config) != {"run", "compute", "data", "api", "tokenizer"}:
            raise PolicyError("rewrite_synthesis exact schema mismatch")
        if set(data) != {
            "input_uri",
            "output_uri",
            "input_sha256",
            "max_records",
            "target_records",
        }:
            raise PolicyError("rewrite_synthesis data schema mismatch")
        protocol = api.get("protocol")
        expected_api_keys = (
            {
                "protocol",
                "verifier_model",
                "max_cost_usd",
                "max_attempts",
                "semantic_similarity_min",
                "concurrency",
            }
            if protocol == M3_BASELINE_VERIFY_PROTOCOL
            else {
                "protocol",
                "generator_models",
                "verifier_by_generator",
                "max_cost_usd",
                "max_attempts",
                "semantic_similarity_min",
                "concurrency",
            }
        )
        if (
            protocol == M3_EVAL_REWRITE_PROTOCOL
            and (config.get("run") or {}).get("arm")
            == "cross-provider-public-eval-input-literal-recovery-v3"
        ):
            expected_api_keys = expected_api_keys | {"literal_inventory"}
        if (
            protocol == M3_EVAL_REWRITE_PROTOCOL
            and (config.get("run") or {}).get("arm")
            == "cross-provider-public-eval-input-placeholder-recovery-v4"
        ):
            expected_api_keys = expected_api_keys | {
                "literal_placeholders",
                "target_token_count",
            }
        if set(api) != expected_api_keys:
            raise PolicyError("rewrite_synthesis API schema mismatch")
        if set(tokenizer) != {"model", "revision"}:
            raise PolicyError("rewrite_synthesis tokenizer schema mismatch")
        if any(
            not str(data.get(field) or "").startswith(
                "modal-volume://humanwrite-checkpoints/"
            )
            for field in ("input_uri", "output_uri")
        ) or data.get("input_uri") == data.get("output_uri"):
            raise PolicyError("rewrite_synthesis requires distinct checkpoint-volume paths")
        if not re.fullmatch(r"[0-9a-f]{64}", str(data.get("input_sha256") or "")):
            raise PolicyError("rewrite_synthesis requires lowercase data.input_sha256")
        max_records = data.get("max_records")
        target_records = data.get("target_records")
        protocol = api.get("protocol")
        expected_target = None
        if isinstance(max_records, int) and not isinstance(max_records, bool):
            expected_target = (
                max_records // 2
                if protocol == M3_SCIENTIFIC_REWRITE_PROTOCOL
                else (max_records * 3) // 4
            )
        if protocol == M3_BASELINE_VERIFY_PROTOCOL:
            valid_scale = (
                isinstance(max_records, int)
                and not isinstance(max_records, bool)
                and max_records in {819, 3277, 9216}
                and target_records == max_records
            )
        elif protocol == M3_EVAL_REWRITE_PROTOCOL:
            valid_scale = max_records == 640 and target_records == 224
        else:
            valid_scale = (
                isinstance(max_records, int)
                and not isinstance(max_records, bool)
                and max_records in {128, 4096, 16384, 46080}
                and target_records == expected_target
            )
        if not valid_scale:
            raise PolicyError("rewrite_synthesis target count does not match its frozen protocol")
        provider_contract_valid = (
            api.get("verifier_model") == "qwen/qwen3-32b"
            if protocol == M3_BASELINE_VERIFY_PROTOCOL
            else (
                protocol in {
                    M3_REWRITE_TASK_PROTOCOL,
                    M3_SCIENTIFIC_REWRITE_PROTOCOL,
                    M3_EVAL_REWRITE_PROTOCOL,
                }
                and api.get("generator_models") == M3_REWRITE_GENERATOR_MODELS
                and api.get("verifier_by_generator") == M3_REWRITE_VERIFIER_BY_GENERATOR
            )
        )
        eval_tail_attempt_contract = (
            protocol == M3_EVAL_REWRITE_PROTOCOL
            and (config.get("run") or {}).get("arm")
            == "cross-provider-public-eval-input-tail-recovery-v2"
            and api.get("max_attempts") == 12
        )
        eval_literal_recovery_contract = (
            protocol == M3_EVAL_REWRITE_PROTOCOL
            and (config.get("run") or {}).get("arm")
            == "cross-provider-public-eval-input-literal-recovery-v3"
            and api.get("max_attempts") == 12
            and api.get("literal_inventory") is True
        )
        eval_placeholder_recovery_contract = (
            protocol == M3_EVAL_REWRITE_PROTOCOL
            and (config.get("run") or {}).get("arm")
            == "cross-provider-public-eval-input-placeholder-recovery-v4"
            and api.get("max_attempts") == 12
            and api.get("literal_placeholders") is True
            and api.get("target_token_count") is True
        )
        if (
            not provider_contract_valid
            or (
                api.get("max_attempts") != 4
                and not eval_tail_attempt_contract
                and not eval_literal_recovery_contract
                and not eval_placeholder_recovery_contract
            )
            or (
                "literal_inventory" in api
                and not eval_literal_recovery_contract
            )
            or (
                ({"literal_placeholders", "target_token_count"} & set(api))
                and not eval_placeholder_recovery_contract
            )
            or api.get("semantic_similarity_min") != 0.90
            or api.get("concurrency") != 16
        ):
            raise PolicyError("rewrite_synthesis frozen provider contract mismatch")
        if tokenizer != {
            "model": QWEN3_14B_MODEL,
            "revision": QWEN3_14B_REVISION,
        }:
            raise PolicyError("rewrite_synthesis requires the pinned Qwen3-14B tokenizer")
        if budget_class != "promo":
            raise PolicyError("rewrite_synthesis requires promo budget")
        api_reserved = float(api.get("max_cost_usd") or 0.0)
        if api_reserved <= 0 or api_reserved > 40.0:
            raise PolicyError("rewrite_synthesis API reservation must be within (0, 40]")
        worst = 0.0
    elif task_kind == "rewrite_judging":
        data = config.get("data") or {}
        judge = config.get("judge") or {}
        if set(config) != {"artifact_schema", "run", "compute", "data", "judge"}:
            raise PolicyError("rewrite_judging exact schema mismatch")
        if set(data) != {
            "panel_uri",
            "panel_sha256",
            "panel_records",
            "sft_uri",
            "sft_sha256",
            "treatment_uri",
            "treatment_sha256",
            "output_uri",
            "manifest_uri",
        }:
            raise PolicyError("rewrite_judging data schema mismatch")
        if set(judge) != {
            "protocol",
            "models",
            "pairwise_dimensions",
            "randomization",
            "temperature",
            "max_completion_tokens",
            "pairwise_response_contract",
            "preservation_response_contract",
            "concurrency",
            "retry_attempts",
            "max_cost_usd",
        }:
            raise PolicyError("rewrite_judging judge schema mismatch")
        uris = [
            str(data.get(field) or "")
            for field in (
                "panel_uri",
                "sft_uri",
                "treatment_uri",
                "output_uri",
                "manifest_uri",
            )
        ]
        if (
            any(not value.startswith("modal-volume://humanwrite-checkpoints/") for value in uris)
            or len(set(uris)) != len(uris)
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(data.get(field) or ""))
                for field in ("panel_sha256", "sft_sha256", "treatment_sha256")
            )
            or data.get("panel_records") != 256
        ):
            raise PolicyError("rewrite_judging artifact bindings are invalid")
        if (
            config.get("artifact_schema") != M3_REWRITE_JUDGE_PROTOCOL
            or judge.get("protocol") != M3_REWRITE_JUDGE_PROTOCOL
            or judge.get("models") != M3_REWRITE_JUDGE_MODELS
            or judge.get("pairwise_dimensions") != M3_REWRITE_JUDGE_PAIRWISE_DIMENSIONS
            or judge.get("randomization") != M3_REWRITE_JUDGE_RANDOMIZATION
            or judge.get("temperature") != 0.0
            or judge.get("max_completion_tokens") != 512
            or judge.get("pairwise_response_contract") != "single uppercase A, B, or TIE"
            or judge.get("preservation_response_contract") != "single uppercase PASS or FAIL"
            or judge.get("concurrency") != 16
            or judge.get("retry_attempts") != 3
            or budget_class != "promo"
        ):
            raise PolicyError("rewrite_judging frozen contract mismatch")
        api_reserved = float(judge.get("max_cost_usd") or 0.0)
        if api_reserved <= 0 or api_reserved > 10.0:
            raise PolicyError("rewrite_judging API reservation must be within (0, 10]")
        worst = 0.0
    else:
        source = config.get("source") or {}
        data = config.get("data") or {}
        for field in ("train_output_uri", "dev_output_uri", "manifest_output_uri"):
            if not str(data.get(field, "")).startswith("modal-volume://humanwrite-checkpoints/"):
                raise PolicyError(f"source_materialization data.{field} must use the checkpoint volume")
        required_source = ("dataset_id", "dataset_config", "revision", "split", "files")
        if any(not source.get(field) for field in required_source):
            raise PolicyError("source_materialization requires a fully pinned source")
        selection = config.get("selection") or {}
        if int(selection.get("corpus_size", 0)) <= 0 or int(selection.get("corpus_size", 0)) > 50_000:
            raise PolicyError("source_materialization corpus_size must be between 1 and 50000")
        max_per_domain = selection.get("max_records_per_domain", 1)
        if (
            not isinstance(max_per_domain, int)
            or isinstance(max_per_domain, bool)
            or max_per_domain != 1
        ):
            raise PolicyError("scale source materialization requires one record per domain")
        exclusion_uris = config.get("exclusion_input_uris") or []
        if not isinstance(exclusion_uris, list) or any(
            not str(uri).startswith("modal-volume://humanwrite-checkpoints/")
            for uri in exclusion_uris
        ):
            raise PolicyError("source_materialization exclusions must use checkpoint-volume URIs")
        worst = 0.0
    return LaunchPolicy(
        comparison_id, budget_class, timeout_seconds, gpu, worst,
        task_kind=task_kind, api_reserved_cost_usd=api_reserved,
    )


def utc_month(timestamp: float | None = None) -> str:
    instant = datetime.fromtimestamp(timestamp, timezone.utc) if timestamp else datetime.now(timezone.utc)
    return instant.strftime("%Y-%m")


def read_events(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    events = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).timestamp())
    serialized = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return row


def run_snapshot(events: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    relevant = [event for event in events if event.get("run_id") == run_id]
    if not relevant:
        return None
    result: dict[str, Any] = {}
    for event in relevant:
        result.update({key: value for key, value in event.items() if value is not None})
    return result


def budget_snapshot(events: list[dict[str, Any]], month: str | None = None) -> dict[str, float]:
    month = month or utc_month()
    run_ids = {str(event["run_id"]) for event in events if event.get("run_id")}
    gpu_committed = 0.0
    for run_id in run_ids:
        launch = next(
            (event for event in events if event.get("run_id") == run_id and event.get("kind") == "run"),
            {},
        )
        state = run_snapshot(events, run_id) or {}
        billing_month = str(launch.get("billing_month") or utc_month(float(launch.get("ts", 0))))
        if billing_month != month:
            continue
        if state.get("status") in TERMINAL:
            gpu_committed += float(state.get("actual_cost_usd") or 0.0)
        else:
            gpu_committed += float(state.get("reserved_cost_usd") or 0.0)
    api_spend = sum(
        float(event.get("cost_usd") or 0.0)
        for event in events
        if event.get("kind") == "api_cost" and utc_month(float(event.get("ts", 0))) == month
    )
    for run_id in run_ids:
        launch = next(
            (event for event in events if event.get("run_id") == run_id and event.get("kind") == "run"),
            {},
        )
        if str(launch.get("billing_month") or utc_month(float(launch.get("ts", 0)))) != month:
            continue
        state = run_snapshot(events, run_id) or {}
        if state.get("task_kind") not in {
            "brief_synthesis",
            "document_cleaning",
            "rewrite_synthesis",
            "rewrite_judging",
        }:
            continue
        if state.get("status") not in TERMINAL:
            already_reported = sum(
                float(event.get("cost_usd") or 0.0)
                for event in events
                if event.get("kind") == "api_cost" and event.get("run_id") == run_id
            )
            reservation = float(state.get("api_reserved_cost_usd") or 0.0)
            api_spend += max(0.0, reservation - already_reported)
    return {
        "gpu_cap_usd": MONTHLY_GPU_CAP_USD,
        "gpu_committed_usd": round(gpu_committed, 6),
        "gpu_remaining_usd": round(max(0.0, MONTHLY_GPU_CAP_USD - gpu_committed), 6),
        "api_cap_usd": MONTHLY_API_CAP_USD,
        "api_spend_usd": round(api_spend, 6),
        "api_remaining_usd": round(max(0.0, MONTHLY_API_CAP_USD - api_spend), 6),
    }


def has_capacity(events: list[dict[str, Any]], requested_usd: float) -> bool:
    return budget_snapshot(events)["gpu_remaining_usd"] >= requested_usd


def has_api_capacity(events: list[dict[str, Any]], requested_usd: float) -> bool:
    return budget_snapshot(events)["api_remaining_usd"] >= requested_usd


def accrued_gpu_spend(events: list[dict[str, Any]], now: float) -> float:
    """Conservative accrued GPU spend for independent reaper decisions."""
    total = 0.0
    run_ids = {str(event["run_id"]) for event in events if event.get("run_id")}
    for run_id in run_ids:
        launch = next(
            (event for event in events if event.get("run_id") == run_id and event.get("kind") == "run"),
            {},
        )
        state = run_snapshot(events, run_id) or {}
        if state.get("status") in TERMINAL:
            total += float(state.get("actual_cost_usd") or 0.0)
            continue
        reserved = float(state.get("reserved_cost_usd") or 0.0)
        timeout = max(1.0, float(state.get("timeout_seconds") or 1.0))
        elapsed = max(0.0, now - float(state.get("started_at") or now))
        total += reserved * min(1.0, elapsed / timeout)
    return round(total, 6)


def authorized(header: str | None, expected_token: str) -> bool:
    if not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:], expected_token)
