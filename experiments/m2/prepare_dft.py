"""Wrapper-only producer for frozen training-human MMD bandwidths."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any

from experiments.m1.contracts import file_sha256, git_sha, write_json
from experiments.m2.representation import (
    TRAINING_BANDWIDTH_DERIVATION,
    TRAINING_BANDWIDTH_PARAMETERIZATION,
    canonical_hash,
    frozen_base_embeddings,
    load_source_peft_and_tokenizer,
    representation_execution_payload,
)


PREPARE_DFT_SCHEMA = "dftr.m2.prepare_training_bandwidths.v1"
PREPARE_DFT_STEP = "prepare_dft"
TRAINING_BANDWIDTH_SCHEMA = "dftr.m2.training_bandwidths.v2"
BASE_MODEL = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TOP_LEVEL_KEYS = {
    "artifact_schema", "run", "compute", "model", "initial_adapter", "data",
    "representation", "derivation", "runtime", "output", "workflow",
}


class PrepareDFTError(ValueError):
    pass


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PrepareDFTError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _sha(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PrepareDFTError(f"{label} must be a lowercase SHA-256")
    return text


def preparation_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "artifact_schema", "run", "compute", "model", "initial_adapter", "data",
            "representation", "derivation", "runtime", "output",
        )
    } | {
        "protocol_version": (config.get("workflow") or {}).get("protocol_version"),
        "step": (config.get("workflow") or {}).get("step"),
    }


def validate_prepare_dft_config(config: dict[str, Any]) -> dict[str, Any]:
    _exact(config, TOP_LEVEL_KEYS, "prepare config")
    if config.get("artifact_schema") != PREPARE_DFT_SCHEMA:
        raise PrepareDFTError("unexpected prepare_dft config schema")
    run = _exact(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") != "training-bandwidths"
        or run.get("budget_class") != "smoke"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 0
    ):
        raise PrepareDFTError("prepare_dft run contract is invalid")
    compute = _exact(config.get("compute"), {"gpu", "gpus", "timeout_min"}, "compute")
    if (
        str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1
        or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int)
        or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= 20
    ):
        raise PrepareDFTError("prepare_dft requires one supported GPU for at most 20 minutes")
    model = _exact(config.get("model"), {"base", "revision", "torch_dtype"}, "model")
    if model != {"base": BASE_MODEL, "revision": BASE_REVISION, "torch_dtype": "bfloat16"}:
        raise PrepareDFTError("prepare_dft requires the frozen Qwen3-4B revision")
    adapter = _exact(
        config.get("initial_adapter"),
        {"path", "adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"},
        "initial_adapter",
    )
    if not Path(str(adapter.get("path") or "")).is_absolute():
        raise PrepareDFTError("initial_adapter.path must be absolute")
    for field in ("adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"):
        _sha(adapter.get(field), f"initial_adapter.{field}")
    data = _exact(
        config.get("data"),
        {"human_targets_path", "human_targets_sha256", "human_text_field"},
        "data",
    )
    human_path = Path(str(data.get("human_targets_path") or ""))
    if (
        not human_path.is_absolute()
        or "harness" in human_path.parts
        or "measurement_v2" in human_path.parts
    ):
        raise PrepareDFTError("human targets must be an absolute training-only artifact")
    _sha(data.get("human_targets_sha256"), "data.human_targets_sha256")
    if not isinstance(data.get("human_text_field"), str) or not data["human_text_field"]:
        raise PrepareDFTError("data.human_text_field must be nonempty")
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
        or representation.get("role") != "training_only_not_measurement_v2"
        or any(
            not isinstance(representation.get(field), int)
            or isinstance(representation.get(field), bool)
            or representation[field] <= 0
            for field in ("batch_size", "max_tokens")
        )
    ):
        raise PrepareDFTError("prepare_dft representation contract is invalid")
    if config.get("derivation") != TRAINING_BANDWIDTH_DERIVATION:
        raise PrepareDFTError("prepare_dft derivation must equal the frozen human-only rule")
    runtime = _exact(
        config.get("runtime"),
        {
            "torch_version", "transformers_version", "peft_version",
            "deterministic_algorithms", "cublas_workspace_config",
        },
        "runtime",
    )
    if (
        any(not str(runtime.get(field) or "") for field in ("torch_version", "transformers_version", "peft_version"))
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
    ):
        raise PrepareDFTError("prepare_dft runtime contract is invalid")
    if config.get("output") != {"filename": "training_bandwidths.json", "overwrite": False}:
        raise PrepareDFTError("prepare_dft output contract is invalid")
    workflow = _exact(
        config.get("workflow"),
        {"protocol_version", "step", "preparation_contract_sha256"},
        "workflow",
    )
    if (
        workflow.get("protocol_version") != PREPARE_DFT_SCHEMA
        or workflow.get("step") != PREPARE_DFT_STEP
        or canonical_hash(preparation_contract_payload(config))
        != _sha(workflow.get("preparation_contract_sha256"), "workflow.preparation_contract_sha256")
    ):
        raise PrepareDFTError("prepare_dft workflow contract hash mismatch")
    return config


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_humans(path: Path, text_field: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    texts: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw, object_pairs_hook=_json_object)
        except (json.JSONDecodeError, ValueError) as error:
            raise PrepareDFTError(f"invalid unique-key human JSONL row {line_number}") from error
        if not isinstance(record, dict):
            raise PrepareDFTError("human JSONL rows must be objects")
        text = record.get(text_field)
        if not isinstance(text, str) or not text:
            raise PrepareDFTError("human target text must be a nonempty string")
        records.append(record)
        texts.append(text)
    if len(texts) < 2 or len(texts) != len(set(texts)):
        raise PrepareDFTError("prepare_dft requires at least two unique human texts")
    return records, texts


def _directory_file_map(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise PrepareDFTError("initial adapter must be a regular directory")
    result: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise PrepareDFTError("initial adapter cannot contain symlinks")
        if item.is_file():
            result[item.relative_to(root).as_posix()] = file_sha256(item)
    if not result:
        raise PrepareDFTError("initial adapter is empty")
    return result


def _verify_inputs(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
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
        raise PrepareDFTError(
            f"runtime version mismatch: expected {expected_runtime} observed {observed_runtime}"
        )
    adapter = config["initial_adapter"]
    adapter_root = Path(adapter["path"])
    for filename, field in (
        ("adapter_model.safetensors", "adapter_model_sha256"),
        ("adapter_config.json", "adapter_config_sha256"),
    ):
        path = adapter_root / filename
        if not path.is_file() or path.is_symlink() or file_sha256(path) != adapter[field]:
            raise PrepareDFTError(f"initial adapter {filename} hash mismatch")
    file_map = _directory_file_map(adapter_root)
    if canonical_hash(file_map) != adapter["file_manifest_sha256"]:
        raise PrepareDFTError("initial adapter complete file map mismatch")
    try:
        adapter_config = json.loads((adapter_root / "adapter_config.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PrepareDFTError("invalid adapter_config.json") from error
    if (
        not isinstance(adapter_config, dict)
        or adapter_config.get("base_model_name_or_path") != BASE_MODEL
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("peft_type") != "LORA"
    ):
        raise PrepareDFTError("initial adapter is not the frozen causal-LM LoRA")
    human_path = Path(config["data"]["human_targets_path"])
    if (
        not human_path.is_file()
        or human_path.is_symlink()
        or file_sha256(human_path) != config["data"]["human_targets_sha256"]
    ):
        raise PrepareDFTError("human target byte hash mismatch")
    _, texts = _load_humans(human_path, config["data"]["human_text_field"])
    return texts, observed_runtime


def derive_training_bandwidths(embeddings: Any, scales: list[float]) -> dict[str, Any]:
    import torch

    matrix = embeddings.detach().cpu().to(torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not torch.isfinite(matrix).all():
        raise PrepareDFTError("human embedding matrix is invalid")
    distances = torch.pdist(matrix, p=2).pow(2)
    if distances.numel() != matrix.shape[0] * (matrix.shape[0] - 1) // 2:
        raise PrepareDFTError("unordered pair-distance cardinality mismatch")
    if not torch.isfinite(distances).all() or torch.any(distances <= 0):
        raise PrepareDFTError("human embeddings contain duplicate or nonfinite pair distances")
    values_list = [float(value) for value in distances.tolist()]
    median_distance = float(statistics.median(values_list))
    bandwidths = [median_distance * float(scale) ** 2 for scale in scales]
    if not math.isfinite(median_distance) or median_distance <= 0 or any(
        not math.isfinite(value) or value <= 0 for value in bandwidths
    ):
        raise PrepareDFTError("derived bandwidth values are invalid")
    embedding_payload = {
        "dtype": "float64", "shape": list(matrix.shape), "values": matrix.tolist()
    }
    return {
        "median_positive_squared_distance": median_distance,
        "values": bandwidths,
        "embedding_dimension": int(matrix.shape[1]),
        "total_unordered_pair_count": len(values_list),
        "positive_pair_distance_count": len(values_list),
        "zero_distance_count": 0,
        "embedding_matrix_sha256": canonical_hash(embedding_payload),
        "positive_distances_sha256": canonical_hash(values_list),
    }


def run_prepare_dft(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import torch

    validate_prepare_dft_config(config)
    if not SAFE_ID_RE.fullmatch(str(run_id)):
        raise PrepareDFTError("run_id is unsafe")
    checkpoint_value = os.environ.get("DFTR_CHECKPOINT_DIR", "")
    checkpoint_dir = Path(checkpoint_value)
    if not checkpoint_value or not checkpoint_dir.is_absolute():
        raise PrepareDFTError("prepare_dft is wrapper-only and requires DFTR_CHECKPOINT_DIR")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("DFTR_M2_A64_READINESS_MANIFEST") or os.environ.get(
        "DFTR_M2_A64_READINESS_SHA256"
    ):
        raise PrepareDFTError("prepare_dft cannot consume A64 readiness evidence")
    if any(item.is_file() or item.is_symlink() for item in checkpoint_dir.rglob("*")):
        raise PrepareDFTError("prepare_dft requires an empty wrapper checkpoint directory")
    output_path = checkpoint_dir / config["output"]["filename"]
    manifest_path = checkpoint_dir / "run_manifest.json"
    config_path = checkpoint_dir / "config.json"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["runtime"]["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    texts, observed_runtime = _verify_inputs(config)
    model, tokenizer = load_source_peft_and_tokenizer(config)
    embeddings = frozen_base_embeddings(model, tokenizer, texts, config)
    derived = derive_training_bandwidths(
        embeddings, list(config["derivation"]["scales"])
    )
    preparation_sha = config["workflow"]["preparation_contract_sha256"]
    artifact = {
        "artifact_schema": TRAINING_BANDWIDTH_SCHEMA,
        "source": "training_humans_only",
        "human_targets_sha256": config["data"]["human_targets_sha256"],
        "human_text_sequence_sha256": canonical_hash(texts),
        "representation_contract_sha256": canonical_hash(config["representation"]),
        "representation_execution_contract_sha256": canonical_hash(
            representation_execution_payload(config)
        ),
        "tokenizer_file_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "source_adapter_file_manifest_sha256": config["initial_adapter"]["file_manifest_sha256"],
        "source_adapter_model_sha256": config["initial_adapter"]["adapter_model_sha256"],
        "source_adapter_config_sha256": config["initial_adapter"]["adapter_config_sha256"],
        "preparation_contract_sha256": preparation_sha,
        "preparation_contract": preparation_contract_payload(config),
        "producer_run_id": run_id,
        "producer_git_sha": git_sha(),
        "producer_config_sha256": canonical_hash(config),
        "producer_config": config,
        "model_base": config["model"]["base"],
        "model_revision": config["model"]["revision"],
        "observed_runtime": observed_runtime,
        "gpu": config["compute"]["gpu"],
        "observed_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "derivation": config["derivation"],
        "human_document_count": len(texts),
        "embedding_dimension": derived["embedding_dimension"],
        "total_unordered_pair_count": derived["total_unordered_pair_count"],
        "positive_pair_distance_count": derived["positive_pair_distance_count"],
        "zero_distance_count": derived["zero_distance_count"],
        "median_positive_squared_distance": derived["median_positive_squared_distance"],
        "embedding_matrix_sha256": derived["embedding_matrix_sha256"],
        "positive_distances_sha256": derived["positive_distances_sha256"],
        "parameterization": TRAINING_BANDWIDTH_PARAMETERIZATION,
        "values": derived["values"],
        "values_sha256": canonical_hash(derived["values"]),
    }
    write_json(config_path, config)
    write_json(output_path, artifact)
    output_sha = file_sha256(output_path)
    manifest = {
        "artifact_schema": "dftr.m2.prepare_training_bandwidths_result.v1",
        "status": "completed",
        "run_id": run_id,
        "comparison_id": config["run"]["comparison_id"],
        "git_sha": artifact["producer_git_sha"],
        "config_sha256": artifact["producer_config_sha256"],
        "preparation_contract_sha256": preparation_sha,
        "training_bandwidths": {
            "path": str(output_path.resolve()),
            "sha256": output_sha,
            "artifact_schema": TRAINING_BANDWIDTH_SCHEMA,
        },
        "token_accounting": {"generated_tokens": 0, "total_tokens": 0},
    }
    write_json(manifest_path, manifest)
    return manifest
