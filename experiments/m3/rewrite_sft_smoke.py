"""Mechanical Qwen3-14B rewrite/generation LoRA training smoke."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any

from data.m3_training_tasks import assemble_mechanical_smoke_corpus
from experiments.m1.contracts import git_sha, write_json, write_jsonl
from experiments.m2.representation import canonical_hash


M3_REWRITE_SFT_SMOKE_SCHEMA = "humanwrite.m3.rewrite_sft_14b_smoke.v1"
M3_REWRITE_SFT_SMOKE_STEP = "train_m3_rewrite_sft_smoke"
BASE_MODEL = "Qwen/Qwen3-14B"
BASE_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
METHOD_KEYS = (
    "artifact_schema",
    "run",
    "compute",
    "model",
    "data",
    "lora",
    "training",
)


class M3RewriteSFTSmokeError(ValueError):
    pass


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise M3RewriteSFTSmokeError(f"{field} must contain exactly {sorted(keys)}")
    return value


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise M3RewriteSFTSmokeError(f"{field} must be a lowercase SHA-256")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    workflow = config.get("workflow") or {}
    return {key: config.get(key) for key in METHOD_KEYS} | {
        "protocol_version": workflow.get("protocol_version"),
        "step": workflow.get("step"),
    }


def validate_m3_rewrite_sft_smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    _exact(config, set(METHOD_KEYS) | {"workflow"}, "M3 rewrite SFT smoke config")
    if config.get("artifact_schema") != M3_REWRITE_SFT_SMOKE_SCHEMA:
        raise M3RewriteSFTSmokeError("unexpected M3 rewrite SFT smoke schema")
    run = _exact(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        run.get("comparison_id") != "M3-rewrite-sft-14b-mechanical-smoke-v1"
        or run.get("arm") != "SFT14-mechanical-smoke"
        or run.get("budget_class") != "smoke"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 1701
    ):
        raise M3RewriteSFTSmokeError("run is not the frozen M3 mechanical smoke")
    if config.get("compute") != {"gpu": "H100", "gpus": 1, "timeout_min": 20}:
        raise M3RewriteSFTSmokeError("mechanical smoke requires one H100 for 20 minutes")
    if config.get("model") != {
        "base": BASE_MODEL,
        "revision": BASE_REVISION,
        "torch_dtype": "bfloat16",
    }:
        raise M3RewriteSFTSmokeError("model must be the pinned Qwen3-14B revision")
    data = _exact(
        config.get("data"),
        {
            "source_briefs_path",
            "source_briefs_sha256",
            "rewrite_tasks_path",
            "rewrite_tasks_sha256",
            "source_records",
            "rewrite_records",
        },
        "data",
    )
    for path_field, sha_field in (
        ("source_briefs_path", "source_briefs_sha256"),
        ("rewrite_tasks_path", "rewrite_tasks_sha256"),
    ):
        path = Path(str(data.get(path_field) or ""))
        if not path.is_absolute() or not str(path).startswith("/checkpoints/"):
            raise M3RewriteSFTSmokeError(f"data.{path_field} must be on /checkpoints")
        _sha(data.get(sha_field), f"data.{sha_field}")
    if data.get("source_records") != 128 or data.get("rewrite_records") != 96:
        raise M3RewriteSFTSmokeError("mechanical smoke requires 128 sources and 96 rewrites")
    if config.get("lora") != {
        "r": 32,
        "alpha": 64,
        "dropout": 0.0,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }:
        raise M3RewriteSFTSmokeError("LoRA contract mismatch")
    if config.get("training") != {
        "optimizer_steps": 16,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "max_prompt_tokens": 768,
        "max_completion_tokens": 383,
        "max_sequence_tokens": 1024,
        "checkpoint_every": 8,
        "schedule": "single_seeded_epoch.v1",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }:
        raise M3RewriteSFTSmokeError("training contract mismatch")
    workflow = _exact(
        config.get("workflow"),
        {"protocol_version", "step", "method_contract_sha256"},
        "workflow",
    )
    if (
        workflow.get("protocol_version") != M3_REWRITE_SFT_SMOKE_SCHEMA
        or workflow.get("step") != M3_REWRITE_SFT_SMOKE_STEP
        or canonical_hash(method_contract_payload(config))
        != _sha(workflow.get("method_contract_sha256"), "workflow.method_contract_sha256")
    ):
        raise M3RewriteSFTSmokeError("workflow contract hash mismatch")
    return config


def deterministic_epoch_indices(records: int, seed: int) -> list[int]:
    if records != 128 or seed != 1701:
        raise M3RewriteSFTSmokeError("mechanical smoke schedule inputs are frozen")
    indices = list(range(records))
    random.Random(seed).shuffle(indices)
    return indices


def prepare_example(tokenizer: Any, row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import torch

    prompt = str(row.get("prompt") or "").strip()
    completion = str(row.get("completion") or "").strip()
    if not prompt or not completion:
        raise M3RewriteSFTSmokeError("training example prompt and completion are required")
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise M3RewriteSFTSmokeError("tokenizer EOS ID is required")
    training = config["training"]
    if len(prompt_ids) > int(training["max_prompt_tokens"]):
        raise M3RewriteSFTSmokeError("mechanical smoke prompt exceeds frozen token limit")
    if len(completion_ids) > int(training["max_completion_tokens"]):
        raise M3RewriteSFTSmokeError("mechanical smoke completion exceeds frozen token limit")
    input_ids = prompt_ids + completion_ids + [eos]
    if len(input_ids) > int(training["max_sequence_tokens"]):
        raise M3RewriteSFTSmokeError("mechanical smoke example exceeds frozen sequence limit")
    labels = [-100] * len(prompt_ids) + completion_ids + [eos]
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "completion_tokens": len(completion_ids) + 1,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise M3RewriteSFTSmokeError(f"{path} contains a non-object row")
    return rows


def _checkpoint_root(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise M3RewriteSFTSmokeError("unsafe run ID")
    root = Path(os.environ.get("DFTR_CHECKPOINT_DIR", f"/checkpoints/runs/{run_id}"))
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in root.iterdir() if path.name != "worker.log"]
    if unexpected:
        raise M3RewriteSFTSmokeError("mechanical smoke checkpoint directory is not empty")
    return root


def run_m3_rewrite_sft_smoke(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    validate_m3_rewrite_sft_smoke_config(config)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["training"]["cublas_workspace_config"]
    set_seed(int(config["run"]["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_path = Path(config["data"]["source_briefs_path"])
    rewrite_path = Path(config["data"]["rewrite_tasks_path"])
    if _file_sha256(source_path) != config["data"]["source_briefs_sha256"]:
        raise M3RewriteSFTSmokeError("source brief hash mismatch")
    if _file_sha256(rewrite_path) != config["data"]["rewrite_tasks_sha256"]:
        raise M3RewriteSFTSmokeError("rewrite task hash mismatch")
    sources = _load_jsonl(source_path)[:128]
    rewrites = _load_jsonl(rewrite_path)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tasks = assemble_mechanical_smoke_corpus(
        sources,
        rewrites,
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = config["lora"]
    policy = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias=str(lora["bias"]),
            task_type=str(lora["task_type"]),
        ),
    )
    policy.train()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    schedule = deterministic_epoch_indices(len(tasks), int(config["run"]["seed"]))
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    logs: list[dict[str, Any]] = []
    completion_tokens = 0
    root = _checkpoint_root(run_id)
    write_json(root / "config.json", config)
    optimizer.zero_grad(set_to_none=True)
    for microstep, index in enumerate(schedule, start=1):
        batch = prepare_example(tokenizer, tasks[index], config)
        device = next(policy.parameters()).device
        output = policy(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            return_dict=True,
        )
        loss = output.loss
        if not bool(torch.isfinite(loss).item()):
            raise M3RewriteSFTSmokeError("non-finite mechanical smoke loss")
        (loss / accumulation).backward()
        completion_tokens += int(batch["completion_tokens"])
        if microstep % accumulation:
            continue
        optimizer_step = microstep // accumulation
        preclip = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["training"]["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(preclip).item()):
            raise M3RewriteSFTSmokeError("non-finite mechanical smoke gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logs.append(
            {
                "optimizer_step": optimizer_step,
                "microsteps_completed": microstep,
                "last_example_index": index,
                "last_task_mode": tasks[index]["task_mode"],
                "last_loss": float(loss.detach()),
                "preclip_gradient_norm": float(preclip),
                "completion_tokens_total": completion_tokens,
            }
        )
        if optimizer_step % int(config["training"]["checkpoint_every"]) == 0:
            checkpoint = root / f"step-{optimizer_step}"
            checkpoint.mkdir(parents=False, exist_ok=False)
            policy.save_pretrained(checkpoint, safe_serialization=True)
            torch.save(
                {
                    "artifact_schema": "humanwrite.m3.rewrite_sft_smoke_state.v1",
                    "optimizer_step": optimizer_step,
                    "schedule_sha256": canonical_hash(schedule),
                    "completion_tokens": completion_tokens,
                    "optimizer_state": optimizer.state_dict(),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                },
                checkpoint / "training_state.pt",
            )
    if len(logs) != int(config["training"]["optimizer_steps"]):
        raise M3RewriteSFTSmokeError("mechanical smoke did not complete exact optimizer exposure")
    policy.save_pretrained(root, safe_serialization=True)
    tokenizer.save_pretrained(root)
    write_jsonl(root / "training_steps.jsonl", logs)
    manifest = {
        "artifact_schema": "humanwrite.m3.rewrite_sft_14b_smoke_result.v1",
        "run_id": run_id,
        "comparison_id": config["run"]["comparison_id"],
        "arm": config["run"]["arm"],
        "status": "completed",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "config_sha256": canonical_hash(config),
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "schedule_sha256": canonical_hash(schedule),
        "source_briefs_sha256": config["data"]["source_briefs_sha256"],
        "rewrite_tasks_sha256": config["data"]["rewrite_tasks_sha256"],
        "optimizer_steps": len(logs),
        "optimizer_examples": len(schedule),
        "teacher_forced_completion_tokens": completion_tokens,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "task_counts": {
            "rewrite": sum(row["task_mode"] == "rewrite" for row in tasks),
            "generate": sum(row["task_mode"] == "generate" for row in tasks),
        },
        "git_sha": git_sha(),
        "scientific_interpretation": "mechanical memory, optimizer, checkpoint, and mixed-task wiring smoke only",
    }
    write_json(root / "run_manifest.json", manifest)
    return manifest


__all__ = [
    "BASE_MODEL",
    "BASE_REVISION",
    "M3_REWRITE_SFT_SMOKE_SCHEMA",
    "M3_REWRITE_SFT_SMOKE_STEP",
    "M3RewriteSFTSmokeError",
    "deterministic_epoch_indices",
    "method_contract_payload",
    "prepare_example",
    "run_m3_rewrite_sft_smoke",
    "validate_m3_rewrite_sft_smoke_config",
]
