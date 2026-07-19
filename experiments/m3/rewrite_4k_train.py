"""Matched SFT14 and HUMANWRITE14 training for the frozen M3 4K screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any

import numpy as np

from data.m3_scientific_corpus import BASE_MODEL, BASE_REVISION
from experiments.m1.contracts import write_json, write_jsonl
from experiments.m2.representation import canonical_hash
from experiments.m3.objectives import (
    MOMENT_COEFFICIENTS,
    calibrate_moment_coefficient,
    select_frequent_tokens,
    surface_features,
    token_moment_loss,
    witness_weights,
)


SCHEMA = "humanwrite.m3.rewrite_train_14b_4k.v1"
STEP = "train_m3_rewrite_4k"
EMBEDDER = "BAAI/bge-small-en-v1.5"
EMBEDDER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CORPUS_PATH = "/checkpoints/data/m3-rewriting-14b-v1/scientific-training-corpus-4096-v1.jsonl"
ARMS = {"SFT14", "HUMANWRITE14"}
METHOD_KEYS = (
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


class M3Rewrite4KError(ValueError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise M3Rewrite4KError(f"{field} must be lowercase SHA-256")
    return text


def method_payload(config: dict[str, Any]) -> dict[str, Any]:
    workflow = config.get("workflow") or {}
    return {key: config.get(key) for key in METHOD_KEYS} | {
        "protocol_version": workflow.get("protocol_version"),
        "step": workflow.get("step"),
    }


def build_config(arm: str, corpus_sha256: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise M3Rewrite4KError("unknown 4K arm")
    _sha(corpus_sha256, "corpus_sha256")
    treatment = arm == "HUMANWRITE14"
    value = {
        "artifact_schema": SCHEMA,
        "run": {
            "comparison_id": "M3-rewriting-14b-4096-scientific-screen-v1",
            "arm": arm,
            "budget_class": "promo",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 3701,
        },
        "compute": {"gpu": "H100", "gpus": 1, "timeout_min": 240},
        "model": {
            "base": BASE_MODEL,
            "revision": BASE_REVISION,
            "torch_dtype": "bfloat16",
        },
        "representation": {
            "embedder": EMBEDDER,
            "revision": EMBEDDER_REVISION,
            "max_length": 512,
            "witness_subset": 512,
        },
        "data": {
            "corpus_path": CORPUS_PATH,
            "corpus_sha256": corpus_sha256,
            "records": 4096,
            "stage_records": 2048,
        },
        "lora": {
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
            "stages": 2,
            "merge_reset_between_stages": True,
        },
        "training": {
            "microbatch_size": 2,
            "gradient_accumulation_steps": 4,
            "optimizer_steps": 512,
            "learning_rate": 2e-5,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "max_prompt_tokens": 768,
            "max_completion_tokens": 383,
            "max_sequence_tokens": 1024,
            "checkpoint_every": 128,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "objectives": {
            "moment_enabled": treatment,
            "witness_enabled": treatment,
            "preference_enabled": False,
            "frequent_tokens": 256,
            "calibration_microbatches": 32,
            "moment_coefficient_candidates": list(MOMENT_COEFFICIENTS),
            "moment_gradient_ratio_target": 0.20,
            "witness_temperature": 0.5,
            "witness_clip": [0.5, 2.0],
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": 383,
            "enable_thinking": False,
            "normal_eos_stopping": True,
        },
    }
    value["workflow"] = {
        "protocol_version": SCHEMA,
        "step": STEP,
        "method_contract_sha256": canonical_hash(
            {key: value.get(key) for key in METHOD_KEYS}
            | {"protocol_version": SCHEMA, "step": STEP}
        ),
    }
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != set(METHOD_KEYS) | {"workflow"} or config.get("artifact_schema") != SCHEMA:
        raise M3Rewrite4KError("4K training exact schema mismatch")
    arm = str((config.get("run") or {}).get("arm") or "")
    corpus_sha = _sha((config.get("data") or {}).get("corpus_sha256"), "data.corpus_sha256")
    if config != build_config(arm, corpus_sha):
        raise M3Rewrite4KError("4K training frozen contract mismatch")
    return config


def deterministic_schedule(records: int, seed: int) -> list[int]:
    if records != 4096 or seed != 3701:
        raise M3Rewrite4KError("4K schedule inputs are frozen")
    values = list(range(records))
    random.Random(seed).shuffle(values)
    return values


def _rendered_prompt(tokenizer: Any, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def prepare_batch(tokenizer: Any, rows: list[dict[str, Any]], config: dict[str, Any]):
    import torch

    sequences, labels, completion_counts = [], [], []
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise M3Rewrite4KError("tokenizer EOS ID is required")
    limits = config["training"]
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        completion = str(row.get("completion") or "").strip()
        prompt_ids = tokenizer.encode(_rendered_prompt(tokenizer, prompt), add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        if (
            not prompt
            or not completion
            or len(prompt_ids) > limits["max_prompt_tokens"]
            or len(completion_ids) > limits["max_completion_tokens"]
            or len(prompt_ids) + len(completion_ids) + 1 > limits["max_sequence_tokens"]
        ):
            raise M3Rewrite4KError("4K example violates frozen token limits")
        sequences.append(prompt_ids + completion_ids + [eos])
        labels.append([-100] * len(prompt_ids) + completion_ids + [eos])
        completion_counts.append(len(completion_ids) + 1)
    width = max(map(len, sequences))
    pad = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    label_tensor = torch.full((len(rows), width), -100, dtype=torch.long)
    for index, (sequence, target) in enumerate(zip(sequences, labels)):
        input_ids[index, : len(sequence)] = torch.tensor(sequence)
        attention[index, : len(sequence)] = 1
        label_tensor[index, : len(target)] = torch.tensor(target)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": label_tensor,
        "completion_tokens": completion_counts,
    }


def per_example_cross_entropy(logits, labels):
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    flat = functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shifted_labels.shape)
    mask = shifted_labels.ne(-100)
    return (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _gradient_norm(parameters) -> float:
    import torch

    total = torch.zeros((), device=next(iter(parameters)).device)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().square().sum()
    return float(total.sqrt().cpu())


def _source_text(row: dict[str, Any]) -> str:
    prompt = str(row["prompt"])
    marker = "SOURCE TEXT:\n"
    if marker not in prompt:
        raise M3Rewrite4KError("rewrite prompt omitted SOURCE TEXT")
    return prompt.split(marker, 1)[1].rsplit("\n\nRETURN:", 1)[0].strip()


def _embed_texts(texts: list[str], config: dict[str, Any], device: str = "cuda") -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    representation = config["representation"]
    tokenizer = AutoTokenizer.from_pretrained(
        representation["embedder"],
        revision=representation["revision"],
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        representation["embedder"],
        revision=representation["revision"],
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=int(representation["max_length"]),
                return_tensors="pt",
            ).to(device)
            output = model(**batch).last_hidden_state[:, 0, :]
            output = torch.nn.functional.normalize(output, dim=-1)
            chunks.append(output.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def _combined_residuals(
    source_texts: list[str], target_texts: list[str], config: dict[str, Any]
) -> np.ndarray:
    embeddings = _embed_texts(source_texts + target_texts, config)
    count = len(source_texts)
    semantic = embeddings[count:] - embeddings[:count]
    surface = np.stack(
        [surface_features(target) - surface_features(source) for source, target in zip(source_texts, target_texts)]
    )
    return np.concatenate([semantic, surface], axis=1)


def _attach_lora(model, config: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    lora = config["lora"]
    return get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias=lora["bias"],
            task_type=lora["task_type"],
        ),
    )


def _witness_stage2_weights(model, tokenizer, rows, schedule, config, root):
    import torch

    eligible = [
        index
        for index, row in enumerate(rows)
        if row["task_mode"] == "rewrite" and row["origin"] != "already_human_noop"
    ]
    subset = sorted(
        eligible,
        key=lambda index: hashlib.sha256(str(rows[index]["fingerprint"]).encode()).hexdigest(),
    )[:512]
    outputs = []
    model.eval()
    with torch.inference_mode():
        for index in subset:
            rendered = _rendered_prompt(tokenizer, str(rows[index]["prompt"]))
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config["generation"]["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs.append(
                tokenizer.decode(
                    generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
            )
    stage2 = schedule[2048:]
    weighted_indices = [
        index
        for index in stage2
        if rows[index]["task_mode"] == "rewrite"
        and rows[index]["origin"] != "already_human_noop"
    ]
    all_sources = [_source_text(rows[index]) for index in weighted_indices]
    all_targets = [str(rows[index]["completion"]) for index in weighted_indices]
    subset_sources = [_source_text(rows[index]) for index in subset]
    subset_targets = [str(rows[index]["completion"]) for index in subset]
    all_human = _combined_residuals(all_sources, all_targets, config)
    subset_human = _combined_residuals(subset_sources, subset_targets, config)
    subset_policy = _combined_residuals(subset_sources, outputs, config)
    weights, state = witness_weights(all_human, subset_human, subset_policy)
    mapping = {index: float(weight) for index, weight in zip(weighted_indices, weights)}
    write_json(
        root / "witness.json",
        {
            "artifact_schema": "humanwrite.m3.witness_weights.v1",
            "subset_fingerprints": [rows[index]["fingerprint"] for index in subset],
            "weights": {str(rows[index]["fingerprint"]): mapping[index] for index in weighted_indices},
            "gap": state["gap"].tolist(),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
        },
    )
    model.train()
    return mapping


def run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    validate_config(config)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["training"]["cublas_workspace_config"]
    set_seed(config["run"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    corpus_path = Path(config["data"]["corpus_path"])
    if _file_sha256(corpus_path) != config["data"]["corpus_sha256"]:
        raise M3Rewrite4KError("4K corpus hash mismatch")
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 4096 or len({row["fingerprint"] for row in rows}) != 4096:
        raise M3Rewrite4KError("4K corpus cardinality mismatch")
    root = Path(os.environ.get("DFTR_CHECKPOINT_DIR", f"/checkpoints/runs/{run_id}"))
    root.mkdir(parents=True, exist_ok=True)
    unexpected_outputs = [path for path in root.iterdir() if path.name != "worker.log"]
    if unexpected_outputs:
        raise M3Rewrite4KError("4K training output directory is not empty")
    write_json(root / "config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    schedule = deterministic_schedule(4096, config["run"]["seed"])
    selected_ids, target_frequency = select_frequent_tokens(
        (tokenizer.encode(str(row["completion"]), add_special_tokens=False) for row in rows),
        special_ids=set(tokenizer.all_special_ids),
        count=256,
    )
    write_json(
        root / "moment_target.json",
        {"selected_ids": selected_ids, "target_frequency": target_frequency.tolist()},
    )
    policy = _attach_lora(base, config)
    policy.train()
    moment_coefficient = 0.0
    calibration = []
    if config["objectives"]["moment_enabled"]:
        ce_norms, moment_norms = [], []
        parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        for index in schedule[:32]:
            batch = prepare_batch(tokenizer, [rows[index]], config)
            device = next(policy.parameters()).device
            output = policy(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                return_dict=True,
            )
            labels = batch["labels"].to(device)
            ce = per_example_cross_entropy(output.logits, labels).mean()
            moment = token_moment_loss(
                output.logits[:, :-1, :], labels[:, 1:], selected_ids, target_frequency
            )
            policy.zero_grad(set_to_none=True)
            ce.backward(retain_graph=True)
            ce_norm = _gradient_norm(parameters)
            policy.zero_grad(set_to_none=True)
            moment.backward()
            moment_norm = _gradient_norm(parameters)
            policy.zero_grad(set_to_none=True)
            ce_norms.append(ce_norm)
            moment_norms.append(moment_norm)
            calibration.append({"index": index, "ce_gradient_norm": ce_norm, "moment_gradient_norm": moment_norm})
        moment_coefficient = calibrate_moment_coefficient(ce_norms, moment_norms)
        write_json(
            root / "moment_calibration.json",
            {"chosen_coefficient": moment_coefficient, "rows": calibration},
        )
    logs = []
    total_completion_tokens = 0
    global_step = 0
    witness_mapping: dict[int, float] = {}
    for stage in (1, 2):
        if stage == 2:
            policy.save_pretrained(root / "stage-1-adapter", safe_serialization=True)
            base = policy.merge_and_unload()
            if config["objectives"]["witness_enabled"]:
                witness_mapping = _witness_stage2_weights(
                    base, tokenizer, rows, schedule, config, root
                )
            policy = _attach_lora(base, config)
            policy.train()
        parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        optimizer.zero_grad(set_to_none=True)
        stage_indices = schedule[(stage - 1) * 2048 : stage * 2048]
        microbatch = config["training"]["microbatch_size"]
        accumulation = config["training"]["gradient_accumulation_steps"]
        for batch_number, start in enumerate(range(0, len(stage_indices), microbatch), start=1):
            indices = stage_indices[start : start + microbatch]
            batch = prepare_batch(tokenizer, [rows[index] for index in indices], config)
            device = next(policy.parameters()).device
            labels = batch["labels"].to(device)
            output = policy(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                return_dict=True,
            )
            weights = torch.tensor(
                [witness_mapping.get(index, 1.0) for index in indices],
                dtype=output.logits.dtype,
                device=device,
            )
            ce_values = per_example_cross_entropy(output.logits, labels)
            ce_loss = (ce_values * weights).mean()
            moment_loss = output.logits.new_zeros(())
            if config["objectives"]["moment_enabled"]:
                moment_values = torch.stack(
                    [
                        token_moment_loss(
                            output.logits[item : item + 1, :-1, :],
                            labels[item : item + 1, 1:],
                            selected_ids,
                            target_frequency,
                        )
                        for item in range(len(indices))
                    ]
                )
                moment_loss = (moment_values * weights).mean()
            loss = ce_loss + moment_coefficient * moment_loss
            if not bool(torch.isfinite(loss).item()):
                raise M3Rewrite4KError("non-finite 4K training loss")
            (loss / accumulation).backward()
            total_completion_tokens += sum(batch["completion_tokens"])
            if batch_number % accumulation:
                continue
            global_step += 1
            preclip = torch.nn.utils.clip_grad_norm_(
                parameters, config["training"]["gradient_clip_norm"]
            )
            if not bool(torch.isfinite(preclip).item()):
                raise M3Rewrite4KError("non-finite 4K training gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            logs.append(
                {
                    "stage": stage,
                    "optimizer_step": global_step,
                    "ce_loss": float(ce_loss.detach()),
                    "moment_loss": float(moment_loss.detach()),
                    "moment_coefficient": moment_coefficient,
                    "preclip_gradient_norm": float(preclip),
                    "completion_tokens_total": total_completion_tokens,
                    "mean_witness_weight": float(weights.float().mean()),
                }
            )
            if global_step % config["training"]["checkpoint_every"] == 0:
                checkpoint = root / "checkpoints" / f"step-{global_step}"
                checkpoint.mkdir(parents=True, exist_ok=False)
                policy.save_pretrained(checkpoint, safe_serialization=True)
                torch.save(
                    {
                        "stage": stage,
                        "optimizer_step": global_step,
                        "optimizer_state": optimizer.state_dict(),
                        "schedule_sha256": canonical_hash(schedule),
                        "cpu_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    },
                    checkpoint / "training_state.pt",
                )
    if global_step != 512:
        raise M3Rewrite4KError("4K training did not complete exact optimizer exposure")
    policy.save_pretrained(root / "stage-2-adapter", safe_serialization=True)
    tokenizer.save_pretrained(root / "tokenizer")
    write_jsonl(root / "training_steps.jsonl", logs)
    manifest = {
        "artifact_schema": "humanwrite.m3.rewrite_train_14b_4k_result.v1",
        "run_id": run_id,
        "arm": config["run"]["arm"],
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "corpus_sha256": config["data"]["corpus_sha256"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "schedule_sha256": canonical_hash(schedule),
        "optimizer_steps": global_step,
        "target_records_seen": 4096,
        "completion_tokens": total_completion_tokens,
        "moment_coefficient": moment_coefficient,
        "stage_1_adapter": str(root / "stage-1-adapter"),
        "stage_2_adapter": str(root / "stage-2-adapter"),
    }
    write_json(root / "run_manifest.json", manifest)
    return manifest


__all__ = [
    "ARMS",
    "CORPUS_PATH",
    "EMBEDDER",
    "EMBEDDER_REVISION",
    "M3Rewrite4KError",
    "SCHEMA",
    "STEP",
    "build_config",
    "deterministic_schedule",
    "per_example_cross_entropy",
    "prepare_batch",
    "run",
    "validate_config",
]
