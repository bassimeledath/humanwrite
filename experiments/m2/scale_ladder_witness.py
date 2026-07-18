"""Generate the frozen SFT baseline witness for the 4K data-scale screen."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from experiments.m2.lower_variance_train import (
    BASE_MODEL,
    BASE_REVISION,
    CONFIRMATION_GENERATION_CONTRACT,
    FULL_BRIEF_SCHEMA,
    FULL_BRIEF_SERIALIZER_SHA256,
    _render_lower_variance_prompt,
    canonical_hash,
)
from experiments.m2.representation import load_source_peft_and_tokenizer


SCALE_LADDER_WITNESS_SCHEMA = "dftr.m2.scale_ladder_baseline_witness.v1"
SCALE_LADDER_WITNESS_STEP = "generate_scale_ladder_witness"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ScaleLadderWitnessError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def witness_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "artifact_schema",
            "run",
            "compute",
            "model",
            "initial_adapter",
            "data",
            "generation",
            "runtime",
        )
    } | {
        "protocol_version": (config.get("workflow") or {}).get("protocol_version"),
        "step": (config.get("workflow") or {}).get("step"),
    }


def validate_scale_ladder_witness_config(config: dict[str, Any]) -> dict[str, Any]:
    expected_top = {
        "artifact_schema", "run", "compute", "model", "initial_adapter",
        "data", "generation", "runtime", "workflow",
    }
    if not isinstance(config, dict) or set(config) != expected_top:
        raise ScaleLadderWitnessError("scale-ladder witness config has unexpected keys")
    if config.get("artifact_schema") != SCALE_LADDER_WITNESS_SCHEMA:
        raise ScaleLadderWitnessError("unexpected scale-ladder witness schema")
    run = config.get("run") or {}
    if (
        set(run) != {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"}
        or not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") != "SFT-baseline-witness-4096"
        or run.get("budget_class") != "screen"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 41001
    ):
        raise ScaleLadderWitnessError("scale-ladder witness run contract is invalid")
    compute = config.get("compute") or {}
    if (
        set(compute) != {"gpu", "gpus", "timeout_min"}
        or compute.get("gpu") != "L40S"
        or compute.get("gpus") != 1
        or not isinstance(compute.get("timeout_min"), int)
        or not 30 <= compute["timeout_min"] <= 120
    ):
        raise ScaleLadderWitnessError("scale-ladder witness requires one bounded L40S")
    if config.get("model") != {
        "base": BASE_MODEL,
        "revision": BASE_REVISION,
        "torch_dtype": "bfloat16",
    }:
        raise ScaleLadderWitnessError("scale-ladder witness model is not frozen")
    adapter = config.get("initial_adapter") or {}
    if set(adapter) != {
        "path", "adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"
    } or not str(adapter.get("path") or "").startswith("/checkpoints/"):
        raise ScaleLadderWitnessError("scale-ladder witness adapter binding is invalid")
    data = config.get("data") or {}
    if set(data) != {
        "briefs_path", "briefs_sha256", "expected_documents", "output_dir",
        "prompt_format", "prompt_schema_version", "prompt_serializer_sha256",
    }:
        raise ScaleLadderWitnessError("scale-ladder witness data contract is invalid")
    if (
        not str(data.get("briefs_path") or "").startswith("/checkpoints/")
        or not str(data.get("output_dir") or "").startswith("/checkpoints/")
        or data.get("expected_documents") != 4096
        or data.get("prompt_schema_version") != FULL_BRIEF_SCHEMA
        or data.get("prompt_serializer_sha256") != FULL_BRIEF_SERIALIZER_SHA256
        or data.get("prompt_format") != "USER:\n{brief}\nASSISTANT:"
    ):
        raise ScaleLadderWitnessError("scale-ladder witness corpus binding is invalid")
    for field in ("briefs_sha256",):
        value = str(data.get(field) or "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ScaleLadderWitnessError(f"data.{field} must be a lowercase SHA-256")
    if config.get("generation") != CONFIRMATION_GENERATION_CONTRACT:
        raise ScaleLadderWitnessError("scale-ladder witness generation contract drifted")
    runtime = config.get("runtime") or {}
    if (
        set(runtime) != {
            "torch_version", "transformers_version", "peft_version",
            "deterministic_algorithms", "cublas_workspace_config",
        }
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
    ):
        raise ScaleLadderWitnessError("scale-ladder witness runtime is invalid")
    workflow = config.get("workflow") or {}
    if (
        set(workflow) != {"protocol_version", "step", "witness_contract_sha256"}
        or workflow.get("protocol_version") != SCALE_LADDER_WITNESS_SCHEMA
        or workflow.get("step") != SCALE_LADDER_WITNESS_STEP
        or workflow.get("witness_contract_sha256")
        != canonical_hash(witness_contract_payload(config))
    ):
        raise ScaleLadderWitnessError("scale-ladder witness contract hash mismatch")
    return config


def run_scale_ladder_witness(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    validate_scale_ladder_witness_config(config)
    if not SAFE_ID_RE.fullmatch(str(run_id)):
        raise ScaleLadderWitnessError("run_id is not a safe artifact identifier")
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["runtime"]["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(int(config["run"]["seed"]))
    torch.cuda.manual_seed_all(int(config["run"]["seed"]))

    data = config["data"]
    briefs_path = Path(data["briefs_path"])
    output_dir = Path(data["output_dir"])
    if not briefs_path.is_file() or _sha(briefs_path) != data["briefs_sha256"]:
        raise ScaleLadderWitnessError("scale-ladder witness brief hash mismatch")
    if output_dir.exists():
        raise ScaleLadderWitnessError("scale-ladder witness output already exists")
    for filename, expected in (
        ("adapter_model.safetensors", config["initial_adapter"]["adapter_model_sha256"]),
        ("adapter_config.json", config["initial_adapter"]["adapter_config_sha256"]),
    ):
        path = Path(config["initial_adapter"]["path"]) / filename
        if not path.is_file() or _sha(path) != expected:
            raise ScaleLadderWitnessError(f"initial adapter hash mismatch: {filename}")
    rows = [json.loads(line) for line in briefs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != data["expected_documents"]:
        raise ScaleLadderWitnessError("scale-ladder witness brief cardinality mismatch")
    prompt_ids = [str(row.get("source_fingerprint") or row.get("fingerprint") or "") for row in rows]
    if any(not value for value in prompt_ids) or len(set(prompt_ids)) != len(rows):
        raise ScaleLadderWitnessError("scale-ladder witness requires unique prompt IDs")

    model, tokenizer = load_source_peft_and_tokenizer(config)
    model.eval()
    batch_size = 8
    generated: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [_render_lower_variance_prompt(row, config) for row in batch],
                padding=True, truncation=True, max_length=1024, return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            sequences = model.generate(
                **encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=128, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
            continuation = sequences[:, encoded["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for offset, text in enumerate(texts):
                text = text.strip()
                if not text:
                    raise ScaleLadderWitnessError(f"empty generation at row {start + offset}")
                generated.append({
                    "prompt_id": prompt_ids[start + offset],
                    "source_fingerprint": prompt_ids[start + offset],
                    "generated_completion": text,
                    "sampling_seed": int(config["run"]["seed"]),
                    "batch_index": start // batch_size,
                })
            if start == 0 or (start + len(batch)) % 256 == 0:
                print(f"witness generated={start + len(batch)}/{len(rows)}", flush=True)

    output_dir.mkdir(parents=True)
    output_path = output_dir / "baseline-generated-4096.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in generated)
    output_path.write_text(payload, encoding="utf-8")
    manifest = {
        "artifact_schema": "dftr.m2.lower_variance_baseline_witness.v2",
        "scientific_role": "training_only_not_evaluation",
        "run_id": run_id,
        "status": "completed",
        "documents": len(generated),
        "briefs_path": str(briefs_path),
        "briefs_sha256": data["briefs_sha256"],
        "generation_contract": config["generation"],
        "generation_contract_sha256": canonical_hash(config["generation"]),
        "sampling_seed": int(config["run"]["seed"]),
        "output_path": str(output_path),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "config_sha256": canonical_hash(config),
    }
    (output_dir / "baseline-generated-4096.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = Path(os.environ["DFTR_CHECKPOINT_DIR"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "SCALE_LADDER_WITNESS_SCHEMA", "SCALE_LADDER_WITNESS_STEP",
    "run_scale_ladder_witness", "validate_scale_ladder_witness_config",
    "witness_contract_payload",
]
