"""Generate the frozen base-Qwen3-14B draft stratum for the M3 scientific corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from data.m3_scientific_corpus import BASE_MODEL, BASE_REVISION, scientific_manifest
from data.m3_training_tasks import render_generation_prompt
from experiments.m1.contracts import write_json
from experiments.m2.representation import canonical_hash


SCHEMA = "humanwrite.m3.baseline_drafts_14b.v1"
STEP = "generate_m3_baseline_drafts"
OUTPUT_SCHEMA = "humanwrite.m3.baseline_draft_candidate.v1"
METHOD_KEYS = ("artifact_schema", "run", "compute", "model", "data", "generation")


class M3BaselineDraftError(ValueError):
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
        raise M3BaselineDraftError(f"{field} must be lowercase SHA-256")
    return text


def method_payload(config: dict[str, Any]) -> dict[str, Any]:
    workflow = config.get("workflow") or {}
    return {key: config.get(key) for key in METHOD_KEYS} | {
        "protocol_version": workflow.get("protocol_version"),
        "step": workflow.get("step"),
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != set(METHOD_KEYS) | {"workflow"} or config.get("artifact_schema") != SCHEMA:
        raise M3BaselineDraftError("baseline-draft exact config schema mismatch")
    run = config.get("run") or {}
    if run != {
        "comparison_id": "M3-rewriting-14b-4096-scientific-screen-v1",
        "arm": "base-draft-construction",
        "budget_class": "screen",
        "task_kind": "experiment",
        "command": ["python", "-m", "experiments.runner"],
        "seed": 2701,
    }:
        raise M3BaselineDraftError("baseline-draft run contract mismatch")
    if config.get("compute") != {"gpu": "H100", "gpus": 1, "timeout_min": 120}:
        raise M3BaselineDraftError("baseline-draft compute contract mismatch")
    if config.get("model") != {
        "base": BASE_MODEL,
        "revision": BASE_REVISION,
        "torch_dtype": "bfloat16",
    }:
        raise M3BaselineDraftError("baseline-draft model contract mismatch")
    data = config.get("data") or {}
    if set(data) != {
        "source_path",
        "source_sha256",
        "source_records",
        "target_records",
        "output_path",
    }:
        raise M3BaselineDraftError("baseline-draft data schema mismatch")
    source_path, output_path = Path(str(data.get("source_path"))), Path(str(data.get("output_path")))
    if (
        not source_path.is_absolute()
        or not output_path.is_absolute()
        or not str(source_path).startswith("/checkpoints/")
        or not str(output_path).startswith("/checkpoints/")
        or source_path == output_path
        or data.get("source_records") != 4096
        or data.get("target_records") != 819
    ):
        raise M3BaselineDraftError("baseline-draft data contract mismatch")
    _sha(data.get("source_sha256"), "data.source_sha256")
    if config.get("generation") != {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "min_new_tokens": 32,
        "max_new_tokens": 383,
        "do_sample": True,
        "enable_thinking": False,
        "checkpoint_every": 32,
        "candidates_per_record": 4,
    }:
        raise M3BaselineDraftError("baseline-draft generation contract mismatch")
    workflow = config.get("workflow") or {}
    if (
        set(workflow) != {"protocol_version", "step", "method_contract_sha256"}
        or workflow.get("protocol_version") != SCHEMA
        or workflow.get("step") != STEP
        or canonical_hash(method_payload(config))
        != _sha(workflow.get("method_contract_sha256"), "workflow.method_contract_sha256")
    ):
        raise M3BaselineDraftError("baseline-draft workflow contract mismatch")
    return config


def baseline_prompt(source: dict[str, Any]) -> str:
    return (
        render_generation_prompt(source)
        + "\n\nThis is a first draft for a later editing pass. Follow the supplied outline and "
        "include every listed fact, name, number, date, and quotation."
    )


def _seed_for(base_seed: int, fingerprint: str) -> int:
    return int(hashlib.sha256(f"{base_seed}:{fingerprint}".encode()).hexdigest()[:16], 16) % (2**31)


def run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    validate_config(config)
    source_path = Path(config["data"]["source_path"])
    output_path = Path(config["data"]["output_path"])
    if _file_sha256(source_path) != config["data"]["source_sha256"]:
        raise M3BaselineDraftError("baseline-draft source hash mismatch")
    sources = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:4096]
    if len(sources) != 4096:
        raise M3BaselineDraftError("baseline-draft source cardinality mismatch")
    manifest = scientific_manifest(sources)
    records = [
        source for source, assignment in zip(sources, manifest)
        if assignment["origin"] == "baseline_model_draft"
    ]
    if len(records) != 819:
        raise M3BaselineDraftError("baseline-draft manifest cardinality mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("artifact_schema") != OUTPUT_SCHEMA:
                raise M3BaselineDraftError("existing baseline candidate schema mismatch")
            completed.add(str(row["fingerprint"]))
    emitted = 0
    generation = config["generation"]
    with output_path.open("a", encoding="utf-8") as sink, torch.inference_mode():
        for source in records:
            fingerprint = str(source["fingerprint"])
            if fingerprint in completed:
                continue
            prompt = baseline_prompt(source)
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
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
            target = int(source["target_length"])
            max_new = min(int(generation["max_new_tokens"]), max(64, math.ceil(target * 1.25)))
            candidates = []
            for attempt in range(1, int(generation["candidates_per_record"]) + 1):
                seed = _seed_for(int(config["run"]["seed"]) + attempt - 1, fingerprint)
                set_seed(seed)
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(generation["temperature"]),
                    top_p=float(generation["top_p"]),
                    top_k=int(generation["top_k"]),
                    min_new_tokens=int(generation["min_new_tokens"]),
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(
                    generated[0, encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                if not text or "�" in text:
                    raise M3BaselineDraftError(
                        f"invalid baseline generation for {fingerprint} attempt {attempt}"
                    )
                candidates.append(
                    {
                        "candidate_attempt": attempt,
                        "input_text": text,
                        "generation_seed": seed,
                    }
                )
            row = {
                "artifact_schema": OUTPUT_SCHEMA,
                "fingerprint": fingerprint,
                "source_fingerprint": str(source.get("source_fingerprint") or fingerprint),
                "completion": str(source["completion"]).strip(),
                "candidates": candidates,
                "rewrite_instruction": (
                    "Rewrite this draft so it reads naturally and distinctly human while preserving "
                    "every fact, name, number, date, quotation, URL, and intent."
                ),
                "generator_model": f"{BASE_MODEL}@{BASE_REVISION}",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            completed.add(fingerprint)
            emitted += 1
            if emitted % int(generation["checkpoint_every"]) == 0:
                # The wrapper commits the shared volume every 15 seconds; this
                # sidecar makes partial progress independently auditable.
                write_json(
                    output_path.with_suffix(".progress.json"),
                    {"run_id": run_id, "records_completed": len(completed)},
                )
    if len(completed) != 819:
        raise M3BaselineDraftError("baseline-draft generation did not reach 819 records")
    manifest_path = output_path.with_suffix(".manifest.json")
    result = {
        "artifact_schema": "humanwrite.m3.baseline_draft_result.v1",
        "run_id": run_id,
        "records": len(completed),
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "source_sha256": config["data"]["source_sha256"],
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
    }
    write_json(manifest_path, result)
    return result


__all__ = ["M3BaselineDraftError", "OUTPUT_SCHEMA", "SCHEMA", "STEP", "baseline_prompt", "run", "validate_config"]
