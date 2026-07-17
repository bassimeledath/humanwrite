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


MONTHLY_GPU_CAP_USD = 40.0
MONTHLY_API_CAP_USD = 100.0
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
    if task_kind not in {"experiment", "brief_synthesis", "source_materialization"}:
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
        "prepare_dft",
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
        if int(selection.get("corpus_size", 0)) <= 0 or int(selection.get("corpus_size", 0)) > 5000:
            raise PolicyError("source_materialization corpus_size must be between 1 and 5000")
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
        if state.get("task_kind") != "brief_synthesis":
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
