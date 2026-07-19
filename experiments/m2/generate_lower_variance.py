"""Adapter-native held-out generation for frozen lower-variance checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from experiments.m1.contracts import file_sha256, git_sha, write_json, write_jsonl
from experiments.m2.dft import _sample_raw_policy
from experiments.m2.lower_variance_train import (
    BASE_MODEL,
    BASE_REVISION,
    FULL_BRIEF_SCHEMA,
    FULL_BRIEF_SERIALIZER_SHA256,
    _render_lower_variance_prompt,
)


GENERATION_SCHEMA = "dftr.m2.lower_variance_generation.v1"
GENERATION_STEP = "generate_lower_variance"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TOP_LEVEL_KEYS = {
    "artifact_schema",
    "run",
    "compute",
    "model",
    "checkpoint",
    "data",
    "sampling",
    "runtime",
    "output",
    "workflow",
}
ALLOWED_WRAPPER_FILES = {"worker.log"}


class GenerationConfigError(ValueError):
    pass


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def generation_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_schema": "dftr.m2.generation_contract.v1",
        "method_contract_sha256": config["checkpoint"]["method_contract_sha256"],
        "prompt_schema_version": config["data"]["prompt_schema_version"],
        "prompt_briefs_sha256": config["data"]["prompt_briefs_sha256"],
        "prompt_serializer_sha256": config["data"]["prompt_serializer_sha256"],
        "prompt_format": config["data"]["prompt_format"],
        "prompt_order": config["sampling"]["prompt_order"],
        "sampling_seed": config["sampling"]["sampling_seed"],
        "seed_scope": config["sampling"]["seed_scope"],
        "distribution": config["sampling"]["distribution"],
        "batch_size": config["sampling"]["batch_size"],
        "new_tokens": config["sampling"]["new_tokens"],
        "max_input_tokens": config["sampling"]["max_input_tokens"],
        "decode": config["sampling"]["decode"],
    }


def decoding_policy_payload(config: dict[str, Any]) -> dict[str, Any]:
    sampling = config["sampling"]
    return {
        "artifact_schema": "dftr.m2.decoding_policy.v1",
        "distribution": sampling["distribution"],
        "raw_logits": True,
        "warpers": [],
        "stopping": "exact_new_token_count",
        "new_tokens": sampling["new_tokens"],
        "decode": sampling["decode"],
    }


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise GenerationConfigError(f"{field} schema mismatch")
    return value


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise GenerationConfigError(f"{field} must be a lowercase SHA-256")
    return text


def validate_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    _exact(config, TOP_LEVEL_KEYS, "config")
    if config.get("artifact_schema") != GENERATION_SCHEMA:
        raise GenerationConfigError("unexpected generation schema")
    run = _exact(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") not in {"SFT-generation", "MMD_WITNESS-generation"}
        or run.get("budget_class") != "screen"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 101
    ):
        raise GenerationConfigError("run contract is invalid")
    compute = _exact(config.get("compute"), {"gpu", "gpus", "timeout_min"}, "compute")
    if (
        str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1
        or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int)
        or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= 120
    ):
        raise GenerationConfigError("compute contract is invalid")
    model = _exact(config.get("model"), {"base", "revision", "torch_dtype"}, "model")
    if model != {
        "base": BASE_MODEL,
        "revision": BASE_REVISION,
        "torch_dtype": "bfloat16",
    }:
        raise GenerationConfigError("model contract is invalid")
    checkpoint = _exact(
        config.get("checkpoint"),
        {
            "path",
            "manifest_sha256",
            "adapter_model_sha256",
            "arm",
            "method_contract_sha256",
        },
        "checkpoint",
    )
    if (
        not Path(str(checkpoint.get("path") or "")).is_absolute()
        or checkpoint.get("arm") not in {"SFT", "MMD_WITNESS"}
        or run["arm"] != f"{checkpoint['arm']}-generation"
    ):
        raise GenerationConfigError("checkpoint identity is invalid")
    for field in ("manifest_sha256", "adapter_model_sha256", "method_contract_sha256"):
        _sha(checkpoint.get(field), f"checkpoint.{field}")
    data = _exact(
        config.get("data"),
        {
            "prompt_briefs_path",
            "prompt_briefs_sha256",
            "prompt_format",
            "prompt_schema_version",
            "prompt_serializer_sha256",
        },
        "data",
    )
    if (
        not Path(str(data.get("prompt_briefs_path") or "")).is_absolute()
        or data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:"
        or data.get("prompt_schema_version") != FULL_BRIEF_SCHEMA
        or data.get("prompt_serializer_sha256") != FULL_BRIEF_SERIALIZER_SHA256
    ):
        raise GenerationConfigError("prompt contract is invalid")
    _sha(data.get("prompt_briefs_sha256"), "data.prompt_briefs_sha256")
    sampling = _exact(
        config.get("sampling"),
        {
            "training_seed",
            "sampling_seed",
            "seed_scope",
            "prompt_order",
            "distribution",
            "batch_size",
            "new_tokens",
            "max_input_tokens",
            "decode",
        },
        "sampling",
    )
    if sampling != {
        "training_seed": 11,
        "sampling_seed": 101,
        "seed_scope": "single_global_rng_stream",
        "prompt_order": "sorted_prompt_id",
        "distribution": "raw_policy_categorical",
        "batch_size": 4,
        "new_tokens": 128,
        "max_input_tokens": 1024,
        "decode": {"skip_special_tokens": True},
    }:
        raise GenerationConfigError("sampling contract is invalid")
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
        runtime.get("torch_version") != "2.13.0+cu130"
        or runtime.get("transformers_version") != "4.57.6"
        or runtime.get("peft_version") != "0.19.1"
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
    ):
        raise GenerationConfigError("runtime contract is invalid")
    if config.get("output") != {"filename": "outputs.jsonl", "overwrite": False}:
        raise GenerationConfigError("output contract is invalid")
    workflow = _exact(
        config.get("workflow"),
        {
            "protocol_version",
            "step",
            "generation_contract_sha256",
            "decoding_policy_sha256",
        },
        "workflow",
    )
    if (
        workflow.get("protocol_version") != GENERATION_SCHEMA
        or workflow.get("step") != GENERATION_STEP
        or workflow.get("generation_contract_sha256")
        != canonical_hash(generation_contract_payload(config))
        or workflow.get("decoding_policy_sha256")
        != canonical_hash(decoding_policy_payload(config))
    ):
        raise GenerationConfigError("workflow contract hash mismatch")
    return config


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GenerationConfigError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise GenerationConfigError("prompt rows must be objects")
    return rows


def _checkpoint_file_map(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise GenerationConfigError("checkpoint cannot contain symlinks")
        if item.is_file() and item.name != "checkpoint_manifest.json":
            observed[item.relative_to(root).as_posix()] = file_sha256(item)
    if not observed:
        raise GenerationConfigError("checkpoint file map is empty")
    return observed


def _require_empty_wrapper_checkpoint_dir(root: Path) -> None:
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise GenerationConfigError(
                "generate_lower_variance requires an empty wrapper checkpoint directory"
            )
        if item.is_file() and relative not in ALLOWED_WRAPPER_FILES:
            raise GenerationConfigError(
                "generate_lower_variance requires an empty wrapper checkpoint directory"
            )


def _verify_inputs(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    checkpoint = config["checkpoint"]
    checkpoint_dir = Path(checkpoint["path"])
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if (
        not checkpoint_dir.is_dir()
        or checkpoint_dir.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or file_sha256(manifest_path) != checkpoint["manifest_sha256"]
        or file_sha256(checkpoint_dir / "adapter_model.safetensors")
        != checkpoint["adapter_model_sha256"]
    ):
        raise GenerationConfigError("checkpoint byte binding failed")
    manifest = _load_json(manifest_path)
    observed_map = _checkpoint_file_map(checkpoint_dir)
    if (
        manifest.get("artifact_schema") != "dftr.m2.lower_variance_adapter_checkpoint.v2"
        or manifest.get("status") != "completed"
        or manifest.get("adapter_native") is not True
        or manifest.get("arm") != checkpoint["arm"]
        or manifest.get("base_model") != config["model"]["base"]
        or manifest.get("base_revision") != config["model"]["revision"]
        or manifest.get("method_contract_sha256")
        != checkpoint["method_contract_sha256"]
        or manifest.get("file_map_excludes") != ["checkpoint_manifest.json"]
        or manifest.get("file_sha256") != observed_map
        or observed_map.get("adapter_model.safetensors")
        != checkpoint["adapter_model_sha256"]
    ):
        raise GenerationConfigError("checkpoint manifest identity mismatch")
    prompt_path = Path(config["data"]["prompt_briefs_path"])
    if (
        not prompt_path.is_file()
        or prompt_path.is_symlink()
        or file_sha256(prompt_path) != config["data"]["prompt_briefs_sha256"]
    ):
        raise GenerationConfigError("prompt brief byte binding failed")
    rows = _load_jsonl(prompt_path)
    rows = sorted(
        rows, key=lambda row: str(row.get("prompt_id") or row.get("fingerprint") or "")
    )
    prompt_ids = [str(row.get("prompt_id") or row.get("fingerprint") or "") for row in rows]
    if len(rows) != 128 or any(not item for item in prompt_ids) or len(set(prompt_ids)) != 128:
        raise GenerationConfigError("generation requires exactly 128 unique prompt IDs")
    return checkpoint_dir, rows


def run_generate_lower_variance(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_generation_config(config)
    checkpoint_dir, rows = _verify_inputs(config)
    output_dir = Path(os.environ.get("DFTR_CHECKPOINT_DIR", ""))
    if not output_dir.is_absolute():
        raise GenerationConfigError(
            "generate_lower_variance requires an empty wrapper checkpoint directory"
        )
    if output_dir.exists():
        _require_empty_wrapper_checkpoint_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        torch.__version__ != config["runtime"]["torch_version"]
        or transformers.__version__ != config["runtime"]["transformers_version"]
        or peft.__version__ != config["runtime"]["peft_version"]
    ):
        raise GenerationConfigError("runtime version mismatch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["runtime"]["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(config["sampling"]["sampling_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["sampling"]["sampling_seed"])
    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["base"],
        revision=config["model"]["revision"],
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(
        base, checkpoint_dir, local_files_only=True, is_trainable=False
    )
    model.eval()
    generated: list[dict[str, Any]] = []
    batch_size = config["sampling"]["batch_size"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [_render_lower_variance_prompt(row, {"data": config["data"]}) for row in batch]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=config["sampling"]["max_input_tokens"],
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        sequences = _sample_raw_policy(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            config["sampling"]["new_tokens"],
        )
        continuation = sequences[:, int(encoded["input_ids"].shape[1]) :]
        if continuation.shape != (len(batch), config["sampling"]["new_tokens"]):
            raise GenerationConfigError("sampler did not produce the exact token exposure")
        texts = tokenizer.batch_decode(continuation, **config["sampling"]["decode"])
        for row, text in zip(batch, texts):
            if not text.strip():
                raise GenerationConfigError("decoded generation is empty")
            generated.append(
                {
                    "prompt_id": str(row.get("prompt_id") or row["fingerprint"]),
                    "training_seed": config["sampling"]["training_seed"],
                    "sampling_seed": config["sampling"]["sampling_seed"],
                    "text": text,
                    "checkpoint_sha256": config["checkpoint"]["adapter_model_sha256"],
                    "generation_contract_sha256": config["workflow"][
                        "generation_contract_sha256"
                    ],
                    "decoding_policy_sha256": config["workflow"][
                        "decoding_policy_sha256"
                    ],
                }
            )
    output_path = output_dir / config["output"]["filename"]
    write_jsonl(output_path, generated)
    output_sha = file_sha256(output_path)
    manifest = {
        "artifact_schema": GENERATION_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "git_sha": git_sha(),
        "config_sha256": canonical_hash(config),
        "comparison_id": config["run"]["comparison_id"],
        "arm": config["checkpoint"]["arm"],
        "adapter_native": True,
        "checkpoint_manifest_sha256": config["checkpoint"]["manifest_sha256"],
        "checkpoint_sha256": config["checkpoint"]["adapter_model_sha256"],
        "method_contract_sha256": config["checkpoint"]["method_contract_sha256"],
        "prompt_briefs_sha256": config["data"]["prompt_briefs_sha256"],
        "generation_contract_sha256": config["workflow"]["generation_contract_sha256"],
        "decoding_policy_sha256": config["workflow"]["decoding_policy_sha256"],
        "documents": len(generated),
        "generated_tokens_per_document": config["sampling"]["new_tokens"],
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "token_accounting": {
            "total_tokens": len(generated) * config["sampling"]["new_tokens"]
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
