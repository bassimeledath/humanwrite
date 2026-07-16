from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .contracts import (
    M1ConfigError,
    build_run_paths,
    canonical_hash,
    count_output_tokens,
    ensure_fixed_hash,
    file_sha256,
    git_sha,
    is_placeholder,
    load_fixed_split_hashes,
    load_jsonl,
    load_resolved_model_manifest,
    read_structured,
    require_resolved_revision,
    resolve_repo_path,
    write_json,
    write_jsonl,
)


REALDATA_PROTOCOLS = {"m1.realdata-pilot.v1", "m1.realdata-adherence.v1"}
FULL_BRIEF_SCHEMA = "dft.full-brief.v1"
REALDATA_ARTIFACT_SCHEMAS = {
    "dftr.realdata_pilot_fixed_inputs.v1",
    "dftr.realdata_adherence_fixed_inputs.v1",
}


def run_m1(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    workflow = config.get("workflow") or {}
    step = str(workflow.get("step", "")).casefold()
    if step == "resolve_revision":
        return _resolve_revision(config, run_id)
    if step == "train_sft":
        return _train_sft(config, run_id)
    if step == "sample_sweep":
        return _sample_sweep(config, run_id)
    if step == "merge_adapter":
        return _merge_adapter(config, run_id)
    raise M1ConfigError(f"unsupported M1 workflow.step: {workflow.get('step')!r}")


def _merge_adapter(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Materialize an immutable full-model artifact for evaluators without PEFT."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.get("model") or {}
    revision = require_resolved_revision(config, context="adapter merge")
    source = Path(str((config.get("artifact") or {}).get("source_adapter_path") or ""))
    checkpoint_root = Path("/checkpoints/runs").resolve()
    try:
        source.resolve().relative_to(checkpoint_root)
    except (ValueError, OSError) as exc:
        raise M1ConfigError("adapter merge source must be under /checkpoints/runs") from exc
    if not source.is_dir() or not (source / "adapter_model.safetensors").is_file():
        raise M1ConfigError("adapter merge source is incomplete")
    expected_adapter_sha = str((config.get("artifact") or {}).get("adapter_sha256") or "")
    if file_sha256(source / "adapter_model.safetensors") != expected_adapter_sha:
        raise M1ConfigError("adapter merge source SHA-256 mismatch")

    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    merged_dir = checkpoint_dir / "merged-model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(model_config.get("base") or ""),
        revision=revision,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(source), local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    tokenizer.save_pretrained(merged_dir)
    sealed_metadata = config.get("sealed_metadata")
    if not isinstance(sealed_metadata, dict):
        raise M1ConfigError("adapter merge requires sealed_metadata")
    write_json(merged_dir / "train_config.json", sealed_metadata)

    files = {
        item.name: file_sha256(item)
        for item in sorted(merged_dir.iterdir())
        if item.is_file()
    }
    manifest = {
        "artifact_schema": "dftr.merged_adapter.v1",
        "run_id": run_id,
        "status": "completed",
        "model_base": str(model_config.get("base") or ""),
        "model_revision": revision,
        "source_adapter_path": str(source),
        "source_adapter_sha256": expected_adapter_sha,
        "merged_model_dir": str(merged_dir),
        "merged_files": files,
        "token_accounting": {"total_tokens": 0},
    }
    write_json(checkpoint_dir / "run_manifest.json", manifest)
    write_json(output_dir / "merged_adapter_manifest.json", manifest)
    return manifest


def _load_fixed_manifest(config: dict[str, Any]) -> dict[str, Any]:
    workflow = config.get("workflow") or {}
    manifest_path = workflow.get("fixed_manifest")
    if not manifest_path:
        raise M1ConfigError("workflow.fixed_manifest is required")
    resolved_manifest = resolve_repo_path(str(manifest_path))
    manifest = read_structured(resolved_manifest)
    protocol_version = str(workflow.get("protocol_version") or "")
    if protocol_version in REALDATA_PROTOCOLS:
        expected_manifest_sha = str(workflow.get("fixed_manifest_sha256") or "")
        if file_sha256(resolved_manifest) != expected_manifest_sha:
            raise M1ConfigError("real-data fixed manifest SHA-256 mismatch")
        expected_schema = (
            "dftr.realdata_adherence_fixed_inputs.v1"
            if protocol_version == "m1.realdata-adherence.v1"
            else "dftr.realdata_pilot_fixed_inputs.v1"
        )
        if manifest.get("artifact_schema") != expected_schema:
            raise M1ConfigError("unexpected real-data fixed manifest schema")
        if int(manifest.get("train_count", 0)) != 256 or int(manifest.get("dev_count", 0)) != 64:
            raise M1ConfigError("real-data pilot fixed manifest must contain train=256/dev=64")
        fixed_hashes = {
            "train": str(manifest.get("train_split_hash") or ""),
            "dev": str(manifest.get("dev_split_hash") or ""),
        }
        if not all(fixed_hashes.values()):
            raise M1ConfigError("real-data fixed manifest split hashes are missing")
    else:
        fixed_hashes = load_fixed_split_hashes()
    ensure_fixed_hash(
        manifest.get("train_split_hash"),
        expected=fixed_hashes["train"],
        field_name="fixed manifest train_split_hash",
    )
    ensure_fixed_hash(
        manifest.get("dev_split_hash"),
        expected=fixed_hashes["dev"],
        field_name="fixed manifest dev_split_hash",
    )
    data_config = config.get("data") or {}
    ensure_fixed_hash(
        data_config.get("train_split_hash"),
        expected=fixed_hashes["train"],
        field_name="config data.train_split_hash",
    )
    ensure_fixed_hash(
        data_config.get("dev_split_hash"),
        expected=fixed_hashes["dev"],
        field_name="config data.dev_split_hash",
    )
    if protocol_version in REALDATA_PROTOCOLS:
        for field in ("train_path", "dev_path"):
            if str(data_config.get(field) or "") != str(manifest.get(field) or ""):
                raise M1ConfigError(f"real-data config/manifest {field} mismatch")
    return manifest


def _record_common_manifest(
    *,
    config: dict[str, Any],
    run_id: str,
    status: str,
    mode: str,
    output_dir: Path,
    checkpoint_dir: Path,
    fixed_manifest: dict[str, Any],
    artifacts: dict[str, Any],
    token_accounting: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "comparison_id": str((config.get("run") or {}).get("comparison_id", "unknown-comparison")),
        "arm": str((config.get("run") or {}).get("arm", "SFT")),
        "status": status,
        "mode": mode,
        "config_hash": canonical_hash(config),
        "git_sha": git_sha(),
        "train_split_hash": str(fixed_manifest["train_split_hash"]),
        "dev_split_hash": str(fixed_manifest["dev_split_hash"]),
        "artifacts": artifacts,
        "token_accounting": token_accounting,
        "provenance": {
            "visible_fixture_counts": {
                "train": int(fixed_manifest.get("train_count", 0)),
                "dev": int(fixed_manifest.get("dev_count", 0)),
            },
            "provenance_note": str(fixed_manifest.get("provenance_note", "")),
        },
    }
    if extra:
        manifest.update(extra)
    for base in (output_dir, checkpoint_dir):
        write_json(base / "run_manifest.json", manifest)
        write_json(base / "config.json", config)
    if checkpoint_dir != output_dir:
        write_json(output_dir / "checkpoint_manifest.json", manifest)
    return manifest


def _resolve_revision(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    fixed_manifest = _load_fixed_manifest(config)
    resolved = load_resolved_model_manifest()
    model_config = config.get("model") or {}
    requested_base = str(model_config.get("base", ""))
    if requested_base != str(resolved["base_model"]):
        raise M1ConfigError(
            f"resolved model base mismatch: config={requested_base} wrapper={resolved['base_model']}"
        )
    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    resolved_artifact = {
        "base_model": requested_base,
        "requested_revision": str(model_config.get("requested_revision") or "default"),
        "resolved_revision": str(resolved["resolved_revision"]),
        "snapshot_path": str(resolved["snapshot_path"]),
        "wrapper_recorded_at": resolved.get("resolved_at"),
        "snapshot_path_sha256": file_sha256(resolved["snapshot_path"])
        if Path(str(resolved["snapshot_path"])).is_file()
        else None,
    }
    write_json(checkpoint_dir / "resolved_revision.json", resolved_artifact)
    return _record_common_manifest(
        config=config,
        run_id=run_id,
        status="completed",
        mode="m1_resolve_revision",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        fixed_manifest=fixed_manifest,
        artifacts={
            "resolved_revision": str((checkpoint_dir / "resolved_revision.json").resolve()),
        },
        token_accounting={"train_tokens": 0, "generated_tokens": 0, "total_tokens": 0},
        extra={"resolved_model": resolved_artifact},
    )


def _render_full_brief(record: dict[str, Any]) -> str:
    required = (
        "user_prompt", "use_case", "style_kind", "style", "detail_mode",
        "target_length", "em_dashes_allowed", "outline",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise M1ConfigError(f"canonical full brief is missing fields: {', '.join(missing)}")
    user_prompt = str(record["user_prompt"]).strip()
    if not user_prompt:
        raise M1ConfigError("canonical record is missing user_prompt")
    outline = record["outline"]
    if not isinstance(outline, list):
        raise M1ConfigError("canonical full brief outline must be a list")
    target_length = int(record["target_length"])
    if target_length <= 0:
        raise M1ConfigError("canonical full brief target_length must be positive")
    return "\n".join((
        f"Writing request: {user_prompt}",
        f"Use case: {str(record['use_case']).strip()}",
        f"Style category: {str(record['style_kind']).strip()}",
        f"Style: {str(record['style']).strip()}",
        f"Detail mode: {str(record['detail_mode']).strip()}",
        f"Target length: about {target_length} words",
        f"Em dashes allowed: {'yes' if bool(record['em_dashes_allowed']) else 'no'}",
        "Grounding outline (use only these supported facts when non-empty): "
        + json.dumps(outline, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ))


def _render_prompt(
    record: dict[str, Any], prompt_format: str, prompt_schema_version: str = "user_prompt.v1"
) -> str:
    user_prompt = str(record.get("user_prompt", "")).strip()
    if not user_prompt:
        raise M1ConfigError("canonical record is missing user_prompt")
    if prompt_schema_version == FULL_BRIEF_SCHEMA:
        if "{brief}" not in prompt_format:
            raise M1ConfigError("full-brief prompt_format must contain {brief}")
        return prompt_format.format(brief=_render_full_brief(record))
    if prompt_schema_version != "user_prompt.v1":
        raise M1ConfigError(f"unsupported prompt schema version: {prompt_schema_version}")
    if "{user_prompt}" not in prompt_format:
        raise M1ConfigError("legacy prompt_format must contain {user_prompt}")
    return prompt_format.format(user_prompt=user_prompt)


def _load_training_records(config: dict[str, Any], fixed_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_config = config.get("data") or {}
    train_path = resolve_repo_path(str(data_config.get("train_path") or fixed_manifest.get("train_path")))
    dev_path = resolve_repo_path(str(data_config.get("dev_path") or fixed_manifest.get("dev_path")))
    if not train_path.is_file() or not dev_path.is_file():
        raise M1ConfigError("canonical M0 train/dev JSONL paths are required")
    train_records, dev_records = load_jsonl(train_path), load_jsonl(dev_path)
    if fixed_manifest.get("artifact_schema") in REALDATA_ARTIFACT_SCHEMAS:
        for split, path, records in (
            ("train", train_path, train_records),
            ("dev", dev_path, dev_records),
        ):
            ensure_fixed_hash(
                file_sha256(path),
                expected=str(fixed_manifest[f"{split}_briefs_sha256"]),
                field_name=f"real-data {split} briefs SHA-256",
            )
            if len(records) != int(fixed_manifest[f"{split}_count"]):
                raise M1ConfigError(f"real-data {split} record count mismatch")
            if any(record.get("split") != split for record in records):
                raise M1ConfigError(f"real-data {split} brief split label mismatch")
            fingerprints = sorted(str(record.get("fingerprint") or "") for record in records)
            if any(not value for value in fingerprints) or len(fingerprints) != len(set(fingerprints)):
                raise M1ConfigError(f"real-data {split} brief fingerprints are invalid")
            observed_split_hash = hashlib.sha256("\n".join(fingerprints).encode("utf-8")).hexdigest()
            ensure_fixed_hash(
                observed_split_hash,
                expected=str(fixed_manifest[f"{split}_split_hash"]),
                field_name=f"real-data {split} brief split hash",
            )
    return train_records, dev_records


def _prepare_supervised_examples(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    prompt_format: str,
    prompt_schema_version: str = "user_prompt.v1",
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    examples = []
    total_tokens = 0
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise M1ConfigError("tokenizer.eos_token_id is required for SFT training")
    for record in records:
        prompt = _render_prompt(record, prompt_format, prompt_schema_version)
        completion = str(record.get("completion", "")).strip()
        if not completion:
            raise M1ConfigError("canonical record is missing completion text")
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_tokens,
        )["input_ids"]
        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
            truncation=True,
            max_length=max_new_tokens,
        )["input_ids"]
        input_ids = prompt_ids + completion_ids + [eos_token_id]
        labels = ([-100] * len(prompt_ids)) + completion_ids + [eos_token_id]
        attention_mask = [1] * len(input_ids)
        total_tokens += sum(token_id != -100 for token_id in labels)
        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )
    return examples, total_tokens


def _build_collator(tokenizer: Any) -> Any:
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        if tokenizer.eos_token_id is None:
            raise M1ConfigError("tokenizer must define pad_token_id or eos_token_id")
        pad_id = tokenizer.eos_token_id

    class _Collator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
            attention = [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features]
            labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
            return {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    input_ids, batch_first=True, padding_value=pad_id
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    attention, batch_first=True, padding_value=0
                ),
                "labels": torch.nn.utils.rnn.pad_sequence(
                    labels, batch_first=True, padding_value=-100
                ),
            }

    return _Collator()


def _load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model_config = config.get("model") or {}
    base_model = str(model_config.get("base", ""))
    revision = require_resolved_revision(config, context="M1 tokenizer load")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_local_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.get("model") or {}
    base_model = str(model_config.get("base", ""))
    revision = require_resolved_revision(config, context="M1 model load")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def _train_sft(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    fixed_manifest = _load_fixed_manifest(config)
    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    tokenizer = _load_tokenizer(config)
    train_records, dev_records = _load_training_records(config, fixed_manifest)
    train_config = config.get("training") or {}
    prompt_format = str(
        (config.get("data") or {}).get("prompt_format") or fixed_manifest.get("prompt_format")
    )
    prompt_schema_version = str(
        (config.get("data") or {}).get("prompt_schema_version")
        or fixed_manifest.get("prompt_schema_version")
        or "user_prompt.v1"
    )
    protocol_version = str((config.get("workflow") or {}).get("protocol_version") or "")
    if protocol_version == "m1.realdata-adherence.v1" and prompt_schema_version != FULL_BRIEF_SCHEMA:
        raise M1ConfigError("adherence protocol requires dft.full-brief.v1 conditioning")
    max_input_tokens = int(
        (config.get("data") or {}).get("max_input_tokens") or fixed_manifest.get("max_input_tokens")
    )
    max_new_tokens = int(
        (config.get("data") or {}).get("max_new_tokens") or fixed_manifest.get("max_new_tokens")
    )
    seeds = list((config.get("run") or {}).get("seeds") or fixed_manifest.get("training_seeds") or [])
    allowed_seeds = ([11], [11, 29, 47]) if protocol_version == "m1.realdata-adherence.v1" else ([11, 29, 47],)
    if seeds not in allowed_seeds:
        raise M1ConfigError(f"M1 training seeds are invalid for {protocol_version}: {seeds}")
    allowed_models = (
        {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"}
        if protocol_version == "m1.realdata-adherence.v1"
        else {"Qwen/Qwen3-1.7B"}
    )
    if str((config.get("model") or {}).get("base")) not in allowed_models:
        raise M1ConfigError(f"M1 evidentiary SFT model is invalid for {protocol_version}")

    examples, tokens_per_epoch = _prepare_supervised_examples(
        train_records,
        tokenizer=tokenizer,
        prompt_format=prompt_format,
        prompt_schema_version=prompt_schema_version,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    lora = (config.get("model") or {}).get("lora") or {}
    lora_config = LoraConfig(
        r=int(lora.get("rank", 64)),
        lora_alpha=int(lora.get("alpha", 128)),
        lora_dropout=float(lora.get("dropout", 0.0)),
        target_modules=list(
            lora.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    collator = _build_collator(tokenizer)
    seed_manifests = []
    total_train_tokens = 0
    for seed in seeds:
        model, _ = _load_local_model_and_tokenizer(config)
        set_seed(int(seed))
        seed_dir = checkpoint_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_seed_model = get_peft_model(model, lora_config)
        training_args = TrainingArguments(
            output_dir=str(seed_dir / "trainer_state"),
            per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
            num_train_epochs=float(train_config.get("num_train_epochs", 1.0)),
            learning_rate=float(train_config.get("learning_rate", 2.0e-4)),
            logging_steps=int(train_config.get("logging_steps", 1)),
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            seed=int(seed),
            dataloader_pin_memory=False,
            bf16=bool(train_config.get("bf16", False)),
            fp16=bool(train_config.get("fp16", False)),
        )
        trainer = Trainer(
            model=per_seed_model,
            args=training_args,
            train_dataset=examples,
            data_collator=collator,
            tokenizer=tokenizer,
        )
        train_result = trainer.train()
        per_seed_model.save_pretrained(seed_dir)
        tokenizer.save_pretrained(seed_dir)
        metrics = dict(train_result.metrics)
        metrics["train_records"] = len(train_records)
        metrics["dev_records"] = len(dev_records)
        metrics["seed"] = int(seed)
        train_tokens = int(math.ceil(tokens_per_epoch * float(train_config.get("num_train_epochs", 1.0))))
        total_train_tokens += train_tokens
        metrics["train_tokens"] = train_tokens
        write_json(seed_dir / "training_metrics.json", metrics)
        seed_manifest = {
            "seed": int(seed),
            "checkpoint_dir": str(seed_dir.resolve()),
            "checkpoint_files": {
                file.name: file_sha256(file)
                for file in sorted(seed_dir.iterdir())
                if file.is_file()
            },
            "training_metrics": str((seed_dir / "training_metrics.json").resolve()),
            "train_tokens": train_tokens,
        }
        write_json(seed_dir / "provenance.json", seed_manifest)
        seed_manifests.append(seed_manifest)

    checkpoints_manifest_path = checkpoint_dir / "checkpoints_manifest.json"
    write_json(
        checkpoints_manifest_path,
        {
            "protocol_version": "m1.checkpoints.v1",
            "model_base": str((config.get("model") or {}).get("base")),
            "model_revision": str((config.get("model") or {}).get("revision")),
            "checkpoints": seed_manifests,
        },
    )
    return _record_common_manifest(
        config=config,
        run_id=run_id,
        status="completed",
        mode="m1_train_sft",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        fixed_manifest=fixed_manifest,
        artifacts={
            "checkpoints_manifest": str(checkpoints_manifest_path.resolve()),
        },
        token_accounting={
            "train_tokens": total_train_tokens,
            "generated_tokens": 0,
            "total_tokens": total_train_tokens,
        },
        extra={
            "training_seeds": seeds,
            "checkpoint_count": len(seed_manifests),
        },
    )


def _load_checkpoint_index(config: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = config.get("sampling") or {}
    manifest_path = sampling.get("checkpoints_manifest")
    if not manifest_path or is_placeholder(manifest_path):
        raise M1ConfigError("sampling.checkpoints_manifest must point to a resolved checkpoint manifest")
    resolved_manifest = resolve_repo_path(str(manifest_path))
    manifest = read_structured(resolved_manifest)
    protocol_version = str((config.get("workflow") or {}).get("protocol_version"))
    realdata = protocol_version in REALDATA_PROTOCOLS
    if realdata:
        ensure_fixed_hash(
            file_sha256(resolved_manifest),
            expected=str(sampling.get("checkpoints_manifest_sha256") or ""),
            field_name="real-data checkpoint manifest SHA-256",
        )
    model_config = config.get("model") or {}
    expected_base = str(model_config.get("base", ""))
    expected_revision = require_resolved_revision(config, context="M1 checkpoint manifest")
    manifest_base = str(manifest.get("model_base", ""))
    manifest_revision = str(manifest.get("model_revision", ""))
    if manifest_base != expected_base:
        raise M1ConfigError(
            f"checkpoint manifest model_base mismatch: expected {expected_base} but found {manifest_base}"
        )
    if manifest_revision != expected_revision:
        raise M1ConfigError(
            "checkpoint manifest model_revision mismatch: "
            f"expected {expected_revision} but found {manifest_revision}"
        )
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise M1ConfigError("checkpoint manifest must contain a non-empty checkpoints list")
    if realdata:
        observed_seeds = [int(row.get("seed", -1)) for row in checkpoints]
        allowed_checkpoint_seeds = (
            ([11], [11, 29, 47])
            if protocol_version == "m1.realdata-adherence.v1"
            else ([11, 29, 47],)
        )
        if observed_seeds not in allowed_checkpoint_seeds:
            raise M1ConfigError("real-data checkpoint seeds do not match the protocol stage")
        adapter_hashes = [
            str((row.get("checkpoint_files") or {}).get("adapter_model.safetensors") or "")
            for row in checkpoints
        ]
        if any(len(value) != 64 for value in adapter_hashes) or len(set(adapter_hashes)) != len(checkpoints):
            raise M1ConfigError("real-data checkpoint adapter hashes must be present and distinct")
        if any(int(row.get("train_tokens", 0)) <= 0 for row in checkpoints):
            raise M1ConfigError("real-data checkpoints require positive train_tokens")
    return checkpoints


def _load_sampler_grid(config: dict[str, Any]) -> dict[str, Any]:
    sampling = config.get("sampling") or {}
    grid_path = sampling.get("sampler_grid")
    if not grid_path:
        raise M1ConfigError("sampling.sampler_grid is required")
    manifest = read_structured(resolve_repo_path(str(grid_path)))
    points = manifest.get("points")
    workflow = config.get("workflow") or {}
    sampling_stage = str(sampling.get("stage") or "")
    directional_pilot = (
        workflow.get("protocol_version") in REALDATA_PROTOCOLS
        and sampling_stage in {"directional_default", "adherence_directional"}
    )
    expected_points = 1 if directional_pilot else 5
    if not isinstance(points, list) or len(points) != expected_points:
        raise M1ConfigError(f"sampler grid must contain exactly {expected_points} points")
    default_points = [point for point in points if float(point.get("temperature", -1)) == 1.0 and float(point.get("top_p", -1)) == 1.0]
    if len(default_points) != 1:
        raise M1ConfigError("sampler grid must contain exactly one default point at temperature=1.0, top_p=1.0")
    return manifest


def _generate_outputs(
    *,
    checkpoint_dir: Path,
    base_model: str,
    base_revision: str,
    records: list[dict[str, Any]],
    prompt_format: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sampling_seed: int,
    do_sample: bool,
    prompt_schema_version: str = "user_prompt.v1",
) -> list[str]:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = str(checkpoint_dir)
    tokenizer_source = model_source
    tokenizer_kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": True}
    if (checkpoint_dir / "adapter_config.json").is_file():
        peft_config = PeftConfig.from_pretrained(model_source, local_files_only=True)
        adapter_base = str(peft_config.base_model_name_or_path)
        if adapter_base != base_model:
            raise M1ConfigError(
                f"adapter base model mismatch: expected {base_model} but found {adapter_base}"
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=base_revision,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_source, local_files_only=True)
        if not _has_tokenizer_files(checkpoint_dir):
            tokenizer_source = base_model
            tokenizer_kwargs["revision"] = base_revision
    else:
        model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    torch.manual_seed(int(sampling_seed))
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sampling_seed))
        model = model.cuda()
    outputs = []
    prompts = [
        _render_prompt(record, prompt_format, prompt_schema_version) for record in records
    ]
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_args.update(temperature=float(temperature), top_p=float(top_p))
        with torch.inference_mode():
            sequence = model.generate(**encoded, **generation_args)[0]
        input_width = encoded["input_ids"].shape[1]
        outputs.append(tokenizer.decode(sequence[input_width:], skip_special_tokens=True))
    return outputs


def _has_tokenizer_files(checkpoint_dir: Path) -> bool:
    tokenizer_files = {
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    return any((checkpoint_dir / name).is_file() for name in tokenizer_files)


def _directional_dev_subset(
    config: dict[str, Any], dev_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    requested_fingerprints = list(
        (config.get("sampling") or {}).get("dev_subset_fingerprints") or []
    )
    if len(requested_fingerprints) != 16 or len(set(requested_fingerprints)) != 16:
        raise M1ConfigError("directional pilot requires exactly 16 unique dev fingerprints")
    dev_index = {str(record.get("fingerprint") or ""): record for record in dev_records}
    if any(fingerprint not in dev_index for fingerprint in requested_fingerprints):
        raise M1ConfigError("directional pilot fingerprint is absent from fixed dev briefs")
    observed_subset_hash = hashlib.sha256(
        "\n".join(sorted(requested_fingerprints)).encode("utf-8")
    ).hexdigest()
    ensure_fixed_hash(
        observed_subset_hash,
        expected=str((config.get("sampling") or {}).get("dev_subset_hash") or ""),
        field_name="directional pilot dev subset hash",
    )
    requested = set(requested_fingerprints)
    return [
        record for record in dev_records if str(record.get("fingerprint") or "") in requested
    ]


def _sample_sweep(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    fixed_manifest = _load_fixed_manifest(config)
    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    model_config = config.get("model") or {}
    base_model = str(model_config.get("base", ""))
    if not base_model:
        raise M1ConfigError("model.base is required for M1 sampler sweep")
    base_revision = require_resolved_revision(config, context="M1 sampler sweep base load")
    checkpoints = _load_checkpoint_index(config)
    sampler_grid = _load_sampler_grid(config)
    prompt_format = str(
        (config.get("data") or {}).get("prompt_format") or fixed_manifest.get("prompt_format")
    )
    prompt_schema_version = str(
        (config.get("data") or {}).get("prompt_schema_version")
        or fixed_manifest.get("prompt_schema_version")
        or "user_prompt.v1"
    )
    protocol_version = str((config.get("workflow") or {}).get("protocol_version") or "")
    if protocol_version == "m1.realdata-adherence.v1" and prompt_schema_version != FULL_BRIEF_SCHEMA:
        raise M1ConfigError("adherence protocol requires dft.full-brief.v1 conditioning")
    max_input_tokens = int(
        (config.get("data") or {}).get("max_input_tokens") or fixed_manifest.get("max_input_tokens")
    )
    max_new_tokens = int(
        (config.get("data") or {}).get("max_new_tokens") or fixed_manifest.get("max_new_tokens")
    )
    sampling_seeds = list((config.get("sampling") or {}).get("seeds") or fixed_manifest.get("sampling_seeds") or [])
    if sampling_seeds != [101, 202, 303]:
        raise M1ConfigError(f"M1 sampling seeds must be [101, 202, 303], found {sampling_seeds}")
    _, dev_records = _load_training_records(config, fixed_manifest)
    if (
        protocol_version in REALDATA_PROTOCOLS
        and str((config.get("sampling") or {}).get("stage"))
        in {"directional_default", "adherence_directional"}
    ):
        dev_records = _directional_dev_subset(config, dev_records)
    index_rows = []
    total_generated_tokens = 0
    for checkpoint in checkpoints:
        checkpoint_seed = int(checkpoint["seed"])
        checkpoint_dir_path = resolve_repo_path(str(checkpoint["checkpoint_dir"]))
        for point in sampler_grid["points"]:
            sampler_id = str(point["id"])
            for sampling_seed in sampling_seeds:
                outputs = _generate_outputs(
                    checkpoint_dir=checkpoint_dir_path,
                    base_model=base_model,
                    base_revision=base_revision,
                    records=dev_records,
                    prompt_format=prompt_format,
                    prompt_schema_version=prompt_schema_version,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=float(point["temperature"]),
                    top_p=float(point["top_p"]),
                    sampling_seed=int(sampling_seed),
                    do_sample=bool(point.get("do_sample", True)),
                )
                rows = []
                for record, output in zip(dev_records, outputs):
                    rows.append(
                        {
                            "fineweb_id": record["fineweb_id"],
                            "user_prompt": record["user_prompt"],
                            "outline": record["outline"],
                            "reference_completion": record["completion"],
                            "generated_completion": output,
                            "output": output,
                            "checkpoint_seed": checkpoint_seed,
                            "sampling_seed": int(sampling_seed),
                            "sampler_id": sampler_id,
                            "temperature": float(point["temperature"]),
                            "top_p": float(point["top_p"]),
                        }
                    )
                sample_path = checkpoint_dir / "samples" / f"checkpoint-seed-{checkpoint_seed}" / sampler_id / f"sampling-seed-{sampling_seed}.jsonl"
                write_jsonl(sample_path, rows)
                generated_tokens = count_output_tokens(rows)
                total_generated_tokens += generated_tokens
                index_rows.append(
                    {
                        "checkpoint_seed": checkpoint_seed,
                        "checkpoint_dir": str(checkpoint_dir_path.resolve()),
                        "sampling_seed": int(sampling_seed),
                        "sampler_id": sampler_id,
                        "temperature": float(point["temperature"]),
                        "top_p": float(point["top_p"]),
                        "do_sample": bool(point.get("do_sample", True)),
                        "samples_path": str(sample_path.resolve()),
                        "generated_tokens": generated_tokens,
                        "report_path": None,
                        "kl_drift": checkpoint.get("kl_drift"),
                    }
                )
    eval_index = {
        "protocol_version": "m1.tier1_eval_index.v1",
        "entries": index_rows,
    }
    eval_index_path = checkpoint_dir / "tier1_eval_index.template.json"
    write_json(eval_index_path, eval_index)
    return _record_common_manifest(
        config=config,
        run_id=run_id,
        status="completed",
        mode="m1_sample_sweep",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        fixed_manifest=fixed_manifest,
        artifacts={
            "tier1_eval_index_template": str(eval_index_path.resolve()),
        },
        token_accounting={
            "train_tokens": 0,
            "generated_tokens": total_generated_tokens,
            "total_tokens": total_generated_tokens,
        },
        extra={
            "sampler_grid": sampler_grid,
            "sampling_seeds": sampling_seeds,
            "sample_count": len(index_rows),
        },
    )
