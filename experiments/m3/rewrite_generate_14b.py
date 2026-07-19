"""Generate the fresh M3 rewrite panel from BASE, SFT14, or HUMANWRITE14."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from data.m3_eval_panel import EVAL_PANEL_PROTOCOL
from data.m3_scientific_corpus import BASE_MODEL, BASE_REVISION
from experiments.m1.contracts import write_json, write_jsonl
from experiments.m2.representation import canonical_hash


SCHEMA = "humanwrite.m3.rewrite_generate_14b.v1"
STEP = "generate_m3_rewrite_candidates"
ARMS = {"BASE", "SFT14", "HUMANWRITE14"}
PANEL_PATH = "/checkpoints/data/m3-rewriting-14b-v1/fresh-rewrite-eval-panel-256-v1.jsonl"
METHOD_KEYS = (
    "artifact_schema",
    "run",
    "compute",
    "model",
    "data",
    "checkpoint",
    "generation",
)


class M3RewriteGenerationError(ValueError):
    pass


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise M3RewriteGenerationError(f"{field} must be lowercase SHA-256")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(
    arm: str,
    panel_sha256: str,
    *,
    training_manifest_path: str | None = None,
    training_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise M3RewriteGenerationError("unknown rewrite generation arm")
    _sha(panel_sha256, "panel_sha256")
    if arm == "BASE":
        if training_manifest_path is not None or training_manifest_sha256 is not None:
            raise M3RewriteGenerationError("BASE generation cannot consume a training manifest")
    else:
        path = Path(str(training_manifest_path or ""))
        if not path.is_absolute() or not str(path).startswith("/checkpoints/"):
            raise M3RewriteGenerationError("trained generation requires a checkpoint manifest")
        _sha(training_manifest_sha256, "training_manifest_sha256")
    value = {
        "artifact_schema": SCHEMA,
        "run": {
            "comparison_id": "M3-rewriting-14b-4096-scientific-screen-v1",
            "arm": f"{arm}-fresh-rewrite-generation",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 4101,
        },
        "compute": {"gpu": "H100", "gpus": 1, "timeout_min": 120},
        "model": {
            "base": BASE_MODEL,
            "revision": BASE_REVISION,
            "torch_dtype": "bfloat16",
        },
        "data": {"panel_path": PANEL_PATH, "panel_sha256": panel_sha256, "records": 256},
        "checkpoint": {
            "arm": arm,
            "training_manifest_path": training_manifest_path,
            "training_manifest_sha256": training_manifest_sha256,
        },
        "generation": {
            "temperature": 0.7,
            "top_p": 0.90,
            "top_k": 50,
            "do_sample": True,
            "max_new_tokens": 383,
            "enable_thinking": False,
            "normal_eos_stopping": True,
            "checkpoint_every": 32,
            "paired_prompt_seed_rule": "sha256(seed:fingerprint)-uint63.v1",
        },
    }
    workflow_payload = {key: value[key] for key in METHOD_KEYS} | {
        "protocol_version": SCHEMA,
        "step": STEP,
    }
    value["workflow"] = {
        "protocol_version": SCHEMA,
        "step": STEP,
        "method_contract_sha256": canonical_hash(workflow_payload),
    }
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != set(METHOD_KEYS) | {"workflow"} or config.get("artifact_schema") != SCHEMA:
        raise M3RewriteGenerationError("rewrite generation exact schema mismatch")
    checkpoint = config.get("checkpoint") or {}
    data = config.get("data") or {}
    expected = build_config(
        str(checkpoint.get("arm") or ""),
        _sha(data.get("panel_sha256"), "data.panel_sha256"),
        training_manifest_path=checkpoint.get("training_manifest_path"),
        training_manifest_sha256=checkpoint.get("training_manifest_sha256"),
    )
    if config != expected:
        raise M3RewriteGenerationError("rewrite generation frozen contract mismatch")
    return config


def prompt_seed(seed: int, fingerprint: str) -> int:
    if type(seed) is not int or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise M3RewriteGenerationError("paired prompt seed inputs are invalid")
    digest = hashlib.sha256(f"{seed}:{fingerprint}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _render_prompt(tokenizer: Any, prompt: str) -> str:
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


def _load_policy(config: dict[str, Any]):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    checkpoint = config["checkpoint"]
    arm = checkpoint["arm"]
    if arm == "BASE":
        model.eval()
        return model, None
    manifest_path = Path(checkpoint["training_manifest_path"])
    if not manifest_path.is_file() or _file_sha256(manifest_path) != checkpoint["training_manifest_sha256"]:
        raise M3RewriteGenerationError("training manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("arm") != arm or manifest.get("optimizer_steps") != 512:
        raise M3RewriteGenerationError("training manifest arm or exposure mismatch")
    stage_1 = Path(str(manifest.get("stage_1_adapter") or ""))
    stage_2 = Path(str(manifest.get("stage_2_adapter") or ""))
    for stage, sha_field in (
        (stage_1, "stage_1_adapter_sha256"),
        (stage_2, "stage_2_adapter_sha256"),
    ):
        weights = stage / "adapter_model.safetensors"
        if not weights.is_file() or _file_sha256(weights) != manifest.get(sha_field):
            raise M3RewriteGenerationError("training adapter hash mismatch")
    stage_1_policy = PeftModel.from_pretrained(model, stage_1, is_trainable=False)
    merged = stage_1_policy.merge_and_unload()
    policy = PeftModel.from_pretrained(merged, stage_2, is_trainable=False)
    policy.eval()
    return policy, manifest


def run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    validate_config(config)
    panel_path = Path(config["data"]["panel_path"])
    if not panel_path.is_file() or _file_sha256(panel_path) != config["data"]["panel_sha256"]:
        raise M3RewriteGenerationError("fresh evaluation panel hash mismatch")
    panel = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (
        len(panel) != 256
        or len({row.get("fingerprint") for row in panel}) != 256
        or any(row.get("artifact_schema") != EVAL_PANEL_PROTOCOL for row in panel)
    ):
        raise M3RewriteGenerationError("fresh evaluation panel invariants failed")
    root = Path(os.environ.get("DFTR_CHECKPOINT_DIR", f"/checkpoints/runs/{run_id}"))
    root.mkdir(parents=True, exist_ok=True)
    if [path for path in root.iterdir() if path.name != "worker.log"]:
        raise M3RewriteGenerationError("rewrite generation output directory is not empty")
    write_json(root / "config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy, training_manifest = _load_policy(config)
    outputs = []
    generation = config["generation"]
    with torch.inference_mode():
        for index, row in enumerate(panel):
            fingerprint = str(row["fingerprint"])
            rendered = _render_prompt(tokenizer, str(row["prompt"]))
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
            seed = prompt_seed(config["run"]["seed"], fingerprint)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            sampled = policy.generate(
                **encoded,
                do_sample=True,
                temperature=generation["temperature"],
                top_p=generation["top_p"],
                top_k=generation["top_k"],
                max_new_tokens=generation["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                generator=generator,
            )
            token_ids = sampled[0, encoded["input_ids"].shape[1] :]
            text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if not text or "�" in text:
                raise M3RewriteGenerationError("empty or malformed rewrite candidate")
            outputs.append(
                {
                    "artifact_schema": "humanwrite.m3.rewrite_candidate.v1",
                    "arm": config["checkpoint"]["arm"],
                    "fingerprint": fingerprint,
                    "category": row["category"],
                    "prompt_seed": seed,
                    "output": text,
                    "output_tokens": int(token_ids.numel()),
                    "eos_reached": bool((token_ids == tokenizer.eos_token_id).any().item()),
                }
            )
            if (index + 1) % generation["checkpoint_every"] == 0:
                write_jsonl(root / "outputs.partial.jsonl", outputs)
    write_jsonl(root / "outputs.jsonl", outputs)
    output_sha = _file_sha256(root / "outputs.jsonl")
    manifest = {
        "artifact_schema": "humanwrite.m3.rewrite_generation_manifest.v1",
        "run_id": run_id,
        "arm": config["checkpoint"]["arm"],
        "panel_sha256": config["data"]["panel_sha256"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
        "training_manifest_sha256": config["checkpoint"]["training_manifest_sha256"],
        "records": len(outputs),
        "output_path": str(root / "outputs.jsonl"),
        "output_sha256": output_sha,
        "output_tokens": sum(int(row["output_tokens"]) for row in outputs),
        "eos_rate": sum(bool(row["eos_reached"]) for row in outputs) / len(outputs),
    }
    if training_manifest is not None:
        manifest["training_corpus_sha256"] = training_manifest["corpus_sha256"]
    if not math.isfinite(float(manifest["eos_rate"])):
        raise M3RewriteGenerationError("generation summary is invalid")
    write_json(root / "run_manifest.json", manifest)
    return manifest


__all__ = [
    "ARMS",
    "M3RewriteGenerationError",
    "PANEL_PATH",
    "SCHEMA",
    "STEP",
    "build_config",
    "prompt_seed",
    "run",
    "validate_config",
]
