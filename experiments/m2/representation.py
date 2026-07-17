"""Shared frozen-base hidden-state representation for DFT preparation and training."""
from __future__ import annotations

import hashlib
import json
from typing import Any


TRAINING_BANDWIDTH_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
TRAINING_BANDWIDTH_DERIVATION = {
    "algorithm": "median_positive_unordered_human_pairwise_squared_distance.v1",
    "scales": TRAINING_BANDWIDTH_SCALES,
    "pair_scope": "all_unordered_training_human_pairs",
    "zero_distance_policy": "fail",
    "degenerate_policy": "fail",
    "distance_dtype": "float64_cpu",
}
TRAINING_BANDWIDTH_PARAMETERIZATION = "squared_distance_over_2_sigma_squared"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def representation_execution_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_schema": "dftr.m2.representation_execution_contract.v1",
        "model": config["model"],
        "representation": config["representation"],
        "runtime": config["runtime"],
        "gpu": config["compute"]["gpu"],
        "tokenizer_file_manifest_sha256": config["initial_adapter"][
            "file_manifest_sha256"
        ],
        "model_loader": "AutoModelForCausalLM.from_pretrained.local_files_only.v1",
        "tokenizer_loader": "initial_adapter.AutoTokenizer.local_files_only.v1",
        "adapter_state": "peft_source_adapter_loaded_then_disable_adapter_context.v1",
        "hidden_output_dtype": "float32",
        "distance_execution": "normalized_embeddings_cpu_float64",
    }


def load_source_peft_and_tokenizer(
    config: dict[str, Any], policy_adapter_path: str | None = None
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["base"],
        revision=config["model"]["revision"],
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["initial_adapter"]["path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(
        base,
        policy_adapter_path or config["initial_adapter"]["path"],
        local_files_only=True,
        is_trainable=True,
    )
    return model, tokenizer


def masked_hidden_embeddings(
    model: Any, tokenizer: Any, texts: list[str], config: dict[str, Any]
) -> Any:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    rows = []
    batch_size = int(config["representation"]["batch_size"])
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=int(config["representation"]["max_tokens"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states[int(config["representation"]["layer"])].float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        rows.append(
            F.normalize(pooled, dim=-1)
            if config["representation"]["normalize"]
            else pooled
        )
    return torch.cat(rows, dim=0)


def frozen_base_embeddings(
    model: Any, tokenizer: Any, texts: list[str], config: dict[str, Any]
) -> Any:
    was_training = bool(model.training)
    model.eval()
    with model.disable_adapter():
        embeddings = masked_hidden_embeddings(model, tokenizer, texts, config)
    if was_training:
        model.train()
    return embeddings
