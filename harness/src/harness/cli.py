"""DFT-R Tier 1 development evaluator command line interface."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
import yaml

from harness import __version__
from harness.metrics import distribution, quality, validity


HARNESS_DIR = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = HARNESS_DIR / "calibration.json"
BASELINE_PATH = HARNESS_DIR / "baseline_stats.json"
DEPLOYMENT_SAMPLER_PATH = HARNESS_DIR / "deployment_sampler.json"
DEV_EMBEDDER_ID = "BAAI/bge-small-en-v1.5"
WEIGHTS = (0.50, 0.25, 0.25)
NON_INFERIORITY_MARGIN = 0.05
CALIBRATION_CONFIDENCE_LEVEL = 0.95
CALIBRATION_Z_95 = 1.959963984540054
CONTINUOUS_INTERVAL_METHOD = "deterministic central order-statistic interval"
REPETITION_INTERVAL_METHOD = "Wilson score interval for binomial proportion"
_CONFIG_NAMES = ("train_config.yaml", "train_config.yml", "train_config.json", "config.yaml", "config.yml", "config.json")
_TEXT_KEYS = ("generated_completion", "generated_text", "output", "text", "completion")


@dataclass
class EvalReport:
    checkpoint_id: str
    harness_version: str
    S: float
    semantic_mmd: float
    semantic_mmd_delta_vs_human_floor: float
    lexical_l2: float
    structural_dist: float
    gate_outline_fact_recall: bool
    gate_unsupported_claim_rate: bool
    gate_language_integrity: bool
    gate_no_collapse: bool
    quality_pref_winrate: float
    jmq: float
    authorship_auc: float
    authorship_auc_ci: tuple[float, float]
    diversity_self_bleu: float
    repetition_rate: float
    human_reference_bank_id: str
    calibration_sha256: str
    baseline_sha256: str
    notes: list[str]


def _read_structured(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _find_config(checkpoint_dir: str | Path) -> tuple[Path | None, dict]:
    directory = Path(checkpoint_dir)
    for name in _CONFIG_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate, _read_structured(candidate)
    return None, {}


def _walk_items(value: Any, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_items(child, path + (str(key).casefold(),))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child, path)
    else:
        yield path, value


def _first_config_value(config, exact_keys, contains=()):
    exact_keys = {key.casefold() for key in exact_keys}
    for path, value in _walk_items(config):
        key = path[-1] if path else ""
        if key in exact_keys or any(fragment in key for fragment in contains):
            if value is not None and not isinstance(value, (dict, list)):
                return value
    return None


def _training_representation_value(config, representation):
    """Find generic ids only when their config path clearly belongs to training."""
    markers = {"train", "training", "objective", "reward", "mmd", "gail"}
    keys = {f"{representation}_id", representation}
    for path, value in _walk_items(config):
        if path and path[-1] in keys and markers.intersection(path[:-1]):
            if value is not None and not isinstance(value, (dict, list)):
                return value
    return None


def _guard_representation(checkpoint_dir: str) -> None:
    """Reject reward/evaluation representation collisions from train config."""
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise ValueError("representation guard requires a checkpoint directory")
    config_path, config = _find_config(directory)
    if config_path is None:
        raise ValueError("checkpoint is missing train_config/config metadata")

    train_embedder = _first_config_value(
        config,
        {"train_embedder_id", "reward_embedder_id", "mmd_embedder_id"},
        ("reward_embedder", "mmd_embedder"),
    ) or _training_representation_value(config, "embedder")
    if train_embedder and str(train_embedder).casefold() == DEV_EMBEDDER_ID.casefold():
        raise ValueError(
            "cross-representation violation: training reward and dev evaluator use the same embedder"
        )

    train_disc = _first_config_value(
        config,
        {"train_discriminator_id", "gail_discriminator_id", "discriminator_id", "disc"},
        ("train_discriminator", "gail_discriminator"),
    ) or _training_representation_value(config, "discriminator")
    eval_disc = _first_config_value(
        config,
        {"eval_discriminator_id", "evaluator_discriminator_id", "authorship_probe_id"},
        ("eval_discriminator", "evaluator_discriminator"),
    )
    if train_disc and eval_disc and str(train_disc).casefold() == str(eval_disc).casefold():
        raise ValueError(
            "cross-representation violation: a GAIL training discriminator cannot evaluate its checkpoint"
        )


def _read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def _record_text(record):
    for key in _TEXT_KEYS:
        if key in record and isinstance(record[key], str):
            return record[key]
    raise ValueError("record has no text field (expected canonical completion or generated output)")


def _sample_path(target: Path):
    if target.is_file():
        return target
    for name in ("samples.jsonl", "eval_samples.jsonl", "generations.jsonl"):
        candidate = target / name
        if candidate.is_file():
            return candidate
    return None


def _load_deployment_sampler():
    if not DEPLOYMENT_SAMPLER_PATH.is_file():
        raise ValueError("frozen deployment sampler config is missing")
    sampler = _read_structured(DEPLOYMENT_SAMPLER_PATH)
    required = {
        "prompt_bank",
        "prompt_format",
        "seed",
        "batch_size",
        "max_input_tokens",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
    }
    if sampler.get("frozen") is not True:
        raise ValueError("deployment sampler is not frozen")
    missing = sorted(key for key in required if key not in sampler or sampler[key] is None)
    if missing:
        raise ValueError("deployment sampler has null/unset fields: " + ", ".join(missing))
    if not isinstance(sampler["prompt_bank"], str) or not sampler["prompt_bank"].strip():
        raise ValueError("deployment sampler prompt_bank must be a non-empty path")
    if not isinstance(sampler["prompt_format"], str):
        raise ValueError("deployment sampler prompt_format must be text")
    if not isinstance(sampler["do_sample"], bool):
        raise ValueError("deployment sampler do_sample must be boolean")
    integer_fields = ("seed", "batch_size", "max_input_tokens", "max_new_tokens")
    if any(isinstance(sampler[key], bool) or not isinstance(sampler[key], int) for key in integer_fields):
        raise ValueError("deployment sampler seed/batch/token fields must be integers")
    if int(sampler["batch_size"]) < 1 or int(sampler["max_input_tokens"]) < 1:
        raise ValueError("deployment sampler batch_size/max_input_tokens must be positive")
    if int(sampler["max_new_tokens"]) < 1 or int(sampler["seed"]) < 0:
        raise ValueError("deployment sampler seed/token limits are invalid")
    if not 0 < float(sampler["top_p"]) <= 1 or float(sampler["temperature"]) <= 0:
        raise ValueError("deployment sampler temperature/top_p are invalid")
    fields = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(sampler["prompt_format"])
        if field_name is not None
    }
    if fields != {"user_prompt"}:
        raise ValueError("deployment sampler prompt_format must use only {user_prompt}")
    return sampler


def _prompt_bank_records(sampler):
    path = Path(str(sampler["prompt_bank"]))
    if not path.is_absolute():
        path = DEPLOYMENT_SAMPLER_PATH.parent / path
    records = _read_jsonl(path)
    for index, record in enumerate(records, 1):
        if not isinstance(record.get("user_prompt"), str) or not record["user_prompt"].strip():
            raise ValueError(f"prompt bank record {index} lacks canonical user_prompt")
    return records


def _default_checkpoint_generator(checkpoint_dir, records, sampler):
    """Local transformers/PEFT generator for the frozen deployment sampler."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("checkpoint generation requires transformers and torch") from error

    checkpoint = Path(checkpoint_dir)
    model_source = str(checkpoint)
    tokenizer_source = model_source
    if (checkpoint / "adapter_config.json").is_file():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as error:
            raise RuntimeError("PEFT adapter generation requires peft") from error
        peft_config = PeftConfig.from_pretrained(model_source, local_files_only=True)
        base_source = peft_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_source, local_files_only=True)
        model = PeftModel.from_pretrained(model, model_source, local_files_only=True)
        if not (checkpoint / "tokenizer_config.json").is_file():
            tokenizer_source = base_source
    else:
        model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    torch.manual_seed(int(sampler["seed"]))
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sampler["seed"]))
    prompts = [
        str(sampler["prompt_format"]).format(user_prompt=record["user_prompt"])
        for record in records
    ]
    outputs = []
    batch_size = int(sampler["batch_size"])
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(sampler["max_input_tokens"]),
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_args = {
            "max_new_tokens": int(sampler["max_new_tokens"]),
            "do_sample": bool(sampler["do_sample"]),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if sampler["do_sample"]:
            generation_args.update(
                temperature=float(sampler["temperature"]), top_p=float(sampler["top_p"])
            )
        with torch.inference_mode():
            sequences = model.generate(**encoded, **generation_args)
        input_width = encoded["input_ids"].shape[1]
        outputs.extend(
            tokenizer.decode(sequence[input_width:], skip_special_tokens=True)
            for sequence in sequences
        )
    return outputs


def _generate_checkpoint_records(checkpoint_dir, generator=None):
    sampler = _load_deployment_sampler()
    prompt_records = _prompt_bank_records(sampler)
    active_generator = generator or _default_checkpoint_generator
    if hasattr(active_generator, "generate") and not callable(active_generator):
        outputs = active_generator.generate(
            checkpoint_dir=str(checkpoint_dir), records=prompt_records, sampler=dict(sampler)
        )
    else:
        try:
            outputs = active_generator(
                checkpoint_dir=str(checkpoint_dir), records=prompt_records, sampler=dict(sampler)
            )
        except TypeError:
            outputs = active_generator(str(checkpoint_dir), prompt_records, dict(sampler))
    outputs = list(outputs)
    if len(outputs) != len(prompt_records):
        raise ValueError("checkpoint generator returned the wrong number of outputs")
    generated_records = []
    for index, (record, output) in enumerate(zip(prompt_records, outputs), 1):
        if not isinstance(output, str):
            raise ValueError(f"checkpoint generator output {index} is not text")
        generated = dict(record)
        if isinstance(record.get("completion"), str):
            generated["reference_completion"] = record["completion"]
        generated["generated_completion"] = output
        generated_records.append(generated)
    return generated_records


def _environment_judge():
    url, token = os.environ.get("HARNESS_JUDGE_URL"), os.environ.get("HARNESS_JUDGE_TOKEN")
    if not url or not token:
        return None

    def judge(*, prompt, candidate_a, candidate_b):
        try:
            response = requests.post(
                url,
                json={
                    "prompt": prompt,
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError):
            # Suppress the client exception context because some HTTP clients
            # include prepared authorization headers in their exception repr.
            raise RuntimeError("Tier 1 quality judge request failed") from None
        if not isinstance(result, dict) or set(result) != {"winner"}:
            raise RuntimeError("Tier 1 quality judge returned an invalid aggregate response")
        return result

    return judge


def _file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _identity_digest(records, *, require_fingerprints):
    fingerprints, text_hashes = [], []
    for index, record in enumerate(records, 1):
        text = _record_text(record)
        text_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
        fingerprint = str(record.get("fingerprint") or "").strip()
        if require_fingerprints and not fingerprint:
            raise ValueError(f"human reference record {index} is missing fingerprint")
        if fingerprint:
            fingerprints.append(fingerprint)
    if len(set(text_hashes)) != len(text_hashes):
        raise ValueError("human reference bank contains duplicate document text")
    if fingerprints and len(set(fingerprints)) != len(fingerprints):
        raise ValueError("human reference bank contains duplicate fingerprints")
    identities = fingerprints if len(fingerprints) == len(records) else text_hashes
    digest = hashlib.sha256(("\n".join(sorted(identities)) + "\n").encode("utf-8")).hexdigest()
    return digest, set(fingerprints)


def _external_human_records(reference_path, manifest_path, sample_records):
    reference_path, manifest_path = Path(reference_path), Path(manifest_path)
    if not reference_path.is_file() or not manifest_path.is_file():
        raise ValueError("frozen human reference bank and manifest must both exist")
    records = _read_jsonl(reference_path)
    manifest = _read_structured(manifest_path)
    required = {
        "artifact_schema",
        "bank_path",
        "bank_sha256",
        "config_path",
        "config_sha256",
        "counts",
        "domains",
        "fingerprints",
        "policy",
        "selection",
        "source",
    }
    if (
        set(manifest) != required
        or manifest.get("artifact_schema") != "dftr.tier1_human_bank.manifest.v1"
    ):
        raise ValueError("human reference manifest violates dftr.tier1_human_bank.manifest.v1")
    if _file_sha256(reference_path) != str(manifest["bank_sha256"]):
        raise ValueError("human reference bank file hash does not match its frozen manifest")
    config_path = Path(str(manifest["config_path"]))
    if not config_path.is_absolute():
        config_path = HARNESS_DIR.parent / config_path
    if not config_path.is_file() or _file_sha256(config_path) != str(manifest["config_sha256"]):
        raise ValueError("human reference selection config is missing or hash-mismatched")
    selection_config = _read_structured(config_path)
    if selection_config.get("artifact_schema") != "dftr.tier1_human_bank.config.v1":
        raise ValueError("human reference selection config schema is invalid")
    counts = manifest.get("counts") or {}
    if len(records) != int(counts.get("bank_size", 0)):
        raise ValueError("human reference bank count does not match its frozen manifest")
    fingerprint_digest, fingerprints = _identity_digest(records, require_fingerprints=True)
    del fingerprint_digest
    ordered_fingerprints = [str(record["fingerprint"]) for record in records]
    ordered_domains = [str(record.get("domain") or "") for record in records]
    if ordered_fingerprints != [str(value) for value in manifest.get("fingerprints") or []]:
        raise ValueError("human reference fingerprints do not match the frozen manifest")
    if ordered_domains != [str(value) for value in manifest.get("domains") or []]:
        raise ValueError("human reference domains do not match the frozen manifest")
    if len(set(ordered_domains)) != len(records) or int(counts.get("unique_domain_count", 0)) != len(records):
        raise ValueError("human reference bank does not preserve distinct-domain selection")
    policy, source, selection = (
        manifest.get("policy") or {},
        manifest.get("source") or {},
        manifest.get("selection") or {},
    )
    if (
        selection_config.get("policy") != policy
        or selection_config.get("source") != source
        or selection_config.get("selection") != selection
    ):
        raise ValueError("human reference manifest does not match its frozen selection config")
    if (
        policy.get("agent_visible") is not True
        or policy.get("hidden_test_materialized") is not False
        or "never training data" not in str(policy.get("purpose", "")).casefold()
    ):
        raise ValueError("human reference manifest violates visible/non-training policy")
    if int(selection.get("bank_size", 0)) != len(records) or not selection.get("seed_label"):
        raise ValueError("human reference manifest lacks frozen selection provenance")
    if not source.get("dataset_id") or not source.get("revision") or not source.get("dataset_config"):
        raise ValueError("human reference manifest lacks pinned source provenance")
    if any(
        str(record.get("source_revision")) != str(source["revision"])
        or str(record.get("source_config")) != str(source["dataset_config"])
        for record in records
    ):
        raise ValueError("human reference rows do not match manifest source provenance")
    excluded_fingerprints = set()
    for raw_path in selection_config.get("exclude_manifests") or []:
        excluded_path = Path(str(raw_path))
        if not excluded_path.is_absolute():
            excluded_path = HARNESS_DIR.parent / excluded_path
        if not excluded_path.is_file():
            raise ValueError("human reference exclusion manifest is missing")
        excluded_manifest = _read_structured(excluded_path)
        values = excluded_manifest.get("fingerprints")
        if not isinstance(values, list):
            raise ValueError("human reference exclusion manifest lacks fingerprints")
        excluded_fingerprints.update(str(value) for value in values)
    if not excluded_fingerprints or fingerprints & excluded_fingerprints:
        raise ValueError("human reference bank lacks proven train/dev fingerprint exclusion")
    sampled_fingerprints = {
        str(record.get("fingerprint")) for record in sample_records if record.get("fingerprint")
    }
    overlap = fingerprints & sampled_fingerprints
    if overlap:
        raise ValueError("human reference bank overlaps sampled prompt/reference fingerprints")
    bank_ids = {str(record.get("fineweb_id")) for record in records if record.get("fineweb_id")}
    sampled_ids = {
        str(record.get("fineweb_id")) for record in sample_records if record.get("fineweb_id")
    }
    if bank_ids & sampled_ids:
        raise ValueError("human reference bank overlaps sampled prompt/reference ids")
    if any(str(record.get("split", "")) != "tier1_visible_human" for record in records):
        raise ValueError("human reference bank rows must use tier1_visible_human split")
    return records, _file_sha256(manifest_path)


def _human_records(sample_records):
    reference = os.environ.get("HARNESS_HUMAN_REFERENCE")
    if reference:
        manifest = os.environ.get("HARNESS_HUMAN_REFERENCE_MANIFEST")
        if not manifest:
            manifest = str(Path(reference).with_suffix(".manifest.json"))
        records, bank_id = _external_human_records(reference, manifest, sample_records)
        if len(records) < 4:
            raise ValueError("Tier 1 requires at least 4 unique held-out human documents")
        return records, bank_id

    inline = []
    for record in sample_records:
        value = record.get("reference_completion", record.get("human_completion"))
        if not isinstance(value, str):
            inline = []
            break
        inline.append({"completion": value, "fingerprint": record.get("fingerprint")})
    if inline:
        bank_id, _ = _identity_digest(inline, require_fingerprints=False)
        if len(inline) < 4:
            raise ValueError(
                "Tier 1 requires at least 4 unique held-out human documents; "
                "supply a separately frozen HARNESS_HUMAN_REFERENCE bank"
            )
        return inline, "inline:" + bank_id
    default = HARNESS_DIR / "dev_human.jsonl"
    if not default.is_file():
        raise ValueError(
            "human references are required: pair reference_completion inline or set HARNESS_HUMAN_REFERENCE"
        )
    manifest = str(default.with_suffix(".manifest.json"))
    records, bank_id = _external_human_records(default, manifest, sample_records)
    if len(records) < 4:
        raise ValueError("Tier 1 requires at least 4 unique held-out human documents")
    return records, bank_id


def _load_json(path, default):
    if not Path(path).is_file():
        return default
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _active_calibration_path():
    configured = os.environ.get("HARNESS_CALIBRATION_PATH")
    return Path(configured).resolve() if configured else Path(CALIBRATION_PATH)


def _baseline_value(baseline, name, field, default):
    value = baseline.get(name, default)
    if isinstance(value, dict):
        value = value.get(field, default)
    return default if value is None else float(value)


def _standardize(value, baseline, name, notes):
    section = baseline.get(name)
    if not isinstance(section, dict) or section.get("mean") is None or section.get("std") is None:
        notes.append(f"{name} baseline mean/std unavailable; S uses the raw component")
        return value
    std = float(section["std"])
    if std <= 0:
        raise ValueError(f"baseline std for {name} must be positive")
    return (value - float(section["mean"])) / std


def _baseline_ready(baseline):
    if baseline.get("frozen") is not True:
        return False
    required = {
        "semantic_mmd": ("mean", "std"),
        "lexical_l2": ("mean", "std"),
        "structural_dist": ("mean", "std"),
        "outline_fact_recall": ("mean",),
        "unsupported_claim_rate": ("mean",),
    }
    for name, fields in required.items():
        section = baseline.get(name)
        if not isinstance(section, dict) or any(section.get(field) is None for field in fields):
            return False
        if "std" in fields and float(section["std"]) <= 0:
            return False
    return True


def _calibration_ready(calibration):
    schema = calibration.get("artifact_schema")
    if (
        schema not in {"harness.calibration.v3", "harness.calibration.v4"}
        or calibration.get("frozen") is not True
    ):
        return False
    methods = calibration.get("interval_methods") or {}
    if any(
        (methods.get(name) or {}).get("method") != CONTINUOUS_INTERVAL_METHOD
        for name in (
            "self_bleu",
            "non_target_script_char_rate",
            "paragraph_len_tokens",
            "sentence_len_tokens",
        )
    ):
        return False
    if (
        (methods.get("repeated_sentence_start_rate") or {}).get("method")
        != REPETITION_INTERVAL_METHOD
    ):
        return False
    if (
        (methods.get("repeated_sentence_start_rate") or {}).get("z")
        != CALIBRATION_Z_95
        or calibration.get("confidence_level") != CALIBRATION_CONFIDENCE_LEVEL
    ):
        return False
    for name in ("self_bleu", "repeated_sentence_start_rate", "non_target_script_char_rate"):
        section = calibration.get(name)
        if not isinstance(section, dict) or section.get("low") is None or section.get("high") is None:
            return False
    repetition = calibration.get("repeated_sentence_start_rate") or {}
    if schema == "harness.calibration.v4" and (
        repetition.get("bound_mode") != "upper_only"
        or calibration.get("policy_version") != "repetition-upper-only.v1"
        or len(str(calibration.get("source_calibration_sha256") or "")) != 64
    ):
        return False
    return True


def evaluate(
    target: str,
    report_path: Optional[str] = None,
    *,
    embedder=None,
    judge=None,
    probe=None,
    generator=None,
) -> EvalReport:
    """Evaluate pre-generated output records against an independent human bank.

    Checkpoint targets are guarded. Existing pre-generated samples are used as
    is; otherwise the frozen deployment sampler generates from the dev prompt
    bank. Injectable arguments are for wrapper-owned clients and offline tests.
    """
    target_path = Path(target)
    if not target_path.exists():
        raise FileNotFoundError(target)
    if target_path.is_dir():
        _guard_representation(str(target_path))
    sample_path = _sample_path(target_path)
    if sample_path is not None:
        records = _read_jsonl(sample_path)
    elif target_path.is_dir():
        records = _generate_checkpoint_records(target_path, generator)
    else:
        raise ValueError("evaluation target is not a JSONL file")
    human_records, human_bank_id = _human_records(records)
    generated = [_record_text(record) for record in records]
    humans = [_record_text(record) for record in human_records]
    if len(generated) < 2 or len(humans) < 4:
        raise ValueError("evaluation requires at least 2 generated and 4 human documents")
    paired_humans = [humans[index % len(humans)] for index in range(len(generated))]
    prompts = [str(record.get("user_prompt", "")) for record in records]
    outlines = [record.get("outline", []) for record in records]

    notes = []
    active_embedder = distribution._resolve_embedder(
        embedder if embedder is not None else DEV_EMBEDDER_ID
    )
    semantic = distribution.semantic_mmd(generated, humans, active_embedder)
    floor, floor_ci = distribution.human_floor_mmd(humans, active_embedder)
    notes.append(f"human-floor 95% CI: [{floor_ci[0]:.6g}, {floor_ci[1]:.6g}]")
    lexical = distribution.lexical_l2(generated, humans, {"hash_dim": 4096, "ngram_range": (1, 3)})
    structural = distribution.structural_distance(generated, humans)

    calibration_path = _active_calibration_path()
    calibration = _load_json(calibration_path, {})
    baseline = _load_json(BASELINE_PATH, {})
    recall = validity.outline_fact_recall(generated, outlines)
    unsupported = validity.unsupported_claim_rate(generated, outlines)
    collapse = validity.collapse_flags(generated, calibration)
    margin = NON_INFERIORITY_MARGIN
    validity_baselines_available = _baseline_ready(baseline)
    calibration_available = _calibration_ready(calibration)
    recall_baseline = _baseline_value(baseline, "outline_fact_recall", "mean", 0.0)
    unsupported_baseline = _baseline_value(baseline, "unsupported_claim_rate", "mean", 1.0)
    if not validity_baselines_available:
        notes.append("SFT validity baselines unavailable; non-inferiority gates fail closed")
    if not calibration_available:
        notes.append("human calibration is unavailable/unfrozen; calibrated gates fail closed")

    active_judge = judge if judge is not None else _environment_judge()
    if active_judge is None:
        preference, jmq_value = 0.5, 1.0
        notes.append("quality judge not supplied; neutral placeholders reported")
    else:
        preference = quality.quality_preference(generated, paired_humans, prompts, active_judge)
        jmq_value = 2.0 * preference
        notes.append("quality preference is secondary and may exhibit judge self-preference")
    if probe is None:
        auc, auc_low, auc_high = quality.fresh_authorship_auc(generated, humans)
        notes.append("authorship AUC uses a fresh deterministic character n-gram out-of-fold probe")
    else:
        auc, auc_low, auc_high = quality.authorship_auc(generated, humans, probe)

    z_semantic = _standardize(semantic, baseline, "semantic_mmd", notes)
    z_lexical = _standardize(lexical, baseline, "lexical_l2", notes)
    z_structural = _standardize(structural, baseline, "structural_dist", notes)
    score = WEIGHTS[0] * z_semantic + WEIGHTS[1] * z_lexical + WEIGHTS[2] * z_structural
    report = EvalReport(
        checkpoint_id=_ckpt_hash(str(target_path)),
        harness_version=__version__,
        S=float(score),
        semantic_mmd=float(semantic),
        semantic_mmd_delta_vs_human_floor=float(semantic - floor),
        lexical_l2=float(lexical),
        structural_dist=float(structural),
        gate_outline_fact_recall=bool(
            validity_baselines_available and recall >= recall_baseline - margin
        ),
        gate_unsupported_claim_rate=bool(
            validity_baselines_available and unsupported <= unsupported_baseline + margin
        ),
        gate_language_integrity=bool(
            calibration_available and validity.language_integrity(generated, calibration)
        ),
        gate_no_collapse=bool(calibration_available and collapse["pass"]),
        quality_pref_winrate=float(preference),
        jmq=float(jmq_value),
        authorship_auc=float(auc),
        authorship_auc_ci=(float(auc_low), float(auc_high)),
        diversity_self_bleu=float(collapse["self_bleu"]),
        repetition_rate=float(collapse["repetition_rate"]),
        human_reference_bank_id=human_bank_id,
        calibration_sha256=_file_sha256(calibration_path) if calibration_path.is_file() else "missing",
        baseline_sha256=_file_sha256(BASELINE_PATH) if Path(BASELINE_PATH).is_file() else "missing",
        notes=notes,
    )
    if report_path:
        Path(report_path).write_text(json.dumps(asdict(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def calibrate(human_split_jsonl: str) -> dict:
    """Compute and persist human-calibrated distribution intervals."""
    records = _read_jsonl(human_split_jsonl)
    texts = [_record_text(record) for record in records]
    if len(texts) < 2:
        raise ValueError("calibration requires at least two human documents")
    script_rates = [validity.non_target_script_char_rate(text) for text in texts]
    repetition_rates = [validity.repeated_sentence_start_rate([text]) for text in texts]
    individual_bleu = []
    for index, text in enumerate(texts):
        others = [other for other_index, other in enumerate(texts) if other_index != index]
        individual_bleu.append(validity.sentence_bleu(text, others))
    paragraph_lengths, sentence_lengths = [], []
    for text in texts:
        paragraph_lengths.extend(
            len(validity._tokens(part)) for part in re.split(r"\n\s*\n", text) if part.strip()
        )
        sentence_lengths.extend(len(validity._tokens(sentence)) for sentence in validity._sentences(text))

    def interval(values):
        ordered = sorted(float(value) for value in values)
        low_q = (1.0 - CALIBRATION_CONFIDENCE_LEVEL) / 2.0
        high_q = 1.0 - low_q
        low_index = min(len(ordered) - 1, max(0, int(low_q * (len(ordered) - 1))))
        high_index = min(len(ordered) - 1, max(0, int(high_q * (len(ordered) - 1))))
        return {"low": ordered[low_index], "high": ordered[high_index]}

    def length_interval(values):
        if not values:
            return {"low": 0.0, "high": 0.0}
        return interval(values)

    result = {
        "artifact_schema": "harness.calibration.v3",
        "frozen": True,
        "_comment": "Human-calibrated ranges from the supplied canonical human split.",
        "self_bleu": interval(individual_bleu),
        "repeated_sentence_start_rate": _wilson_interval(
            int(sum(repetition_rates)), len(repetition_rates), CALIBRATION_CONFIDENCE_LEVEL
        ),
        "non_target_script_char_rate": interval(script_rates),
        "paragraph_len_tokens": length_interval(paragraph_lengths),
        "sentence_len_tokens": length_interval(sentence_lengths),
        "metric_counts": {
            "repeated_sentence_start_rate": {
                "successes": int(sum(repetition_rates)),
                "trials": len(repetition_rates),
            }
        },
        "interval_methods": _calibration_interval_methods(CALIBRATION_CONFIDENCE_LEVEL),
        "confidence_level": CALIBRATION_CONFIDENCE_LEVEL,
    }
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _validated_interval(value, field_name):
    if not isinstance(value, dict) or value.get("low") is None or value.get("high") is None:
        raise ValueError(f"{field_name} must contain non-null low/high")
    low, high = float(value["low"]), float(value["high"])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{field_name} contains an invalid interval")
    return {"low": low, "high": high}


def _wilson_interval(successes: int, trials: int, confidence_level: float) -> dict[str, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials")
    if confidence_level != CALIBRATION_CONFIDENCE_LEVEL:
        raise ValueError("Wilson interval confidence level must be the frozen 0.95")
    z = CALIBRATION_Z_95
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (z / denominator) * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return {"low": max(0.0, center - half_width), "high": min(1.0, center + half_width)}


def _calibration_interval_methods(confidence_level):
    if confidence_level != CALIBRATION_CONFIDENCE_LEVEL:
        raise ValueError("calibration interval methods require the frozen 0.95 confidence")
    return {
        "self_bleu": {"method": CONTINUOUS_INTERVAL_METHOD},
        "repeated_sentence_start_rate": {
            "method": REPETITION_INTERVAL_METHOD,
            "z": CALIBRATION_Z_95,
        },
        "non_target_script_char_rate": {"method": CONTINUOUS_INTERVAL_METHOD},
        "paragraph_len_tokens": {"method": CONTINUOUS_INTERVAL_METHOD},
        "sentence_len_tokens": {"method": CONTINUOUS_INTERVAL_METHOD},
    }


def prepare_calibration_transfer(proposal_path: str, expected_sha256: str) -> dict:
    """Validate a reviewed M1 proposal and emit exact immutable harness bytes."""
    path = Path(proposal_path)
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError("calibration proposal SHA-256 does not match operator-reviewed bytes")
    proposal = _read_structured(path)
    if proposal.get("artifact_schema") != "m1.calibration_proposal.review.v3":
        raise ValueError("unsupported calibration proposal schema")
    required = {
        "artifact_schema",
        "human_split_sha256",
        "sample_count",
        "point_estimates",
        "interval_methods",
        "metric_counts",
        "confidence_level",
        "intervals",
        "split_hashes",
        "resampling_seeds",
        "review_limitations",
    }
    if not required.issubset(proposal):
        raise ValueError("calibration proposal is missing transfer-contract fields")
    if int(proposal["sample_count"]) < 2:
        raise ValueError("calibration proposal requires at least two unique human documents")
    if float(proposal["confidence_level"]) != CALIBRATION_CONFIDENCE_LEVEL:
        raise ValueError("calibration proposal confidence level must be 0.95")
    if not proposal["resampling_seeds"] or not proposal["review_limitations"]:
        raise ValueError("calibration proposal lacks reproducibility/limitation evidence")
    intervals = proposal["intervals"]
    expected_methods = _calibration_interval_methods(CALIBRATION_CONFIDENCE_LEVEL)
    if proposal["interval_methods"] != expected_methods:
        raise ValueError("calibration proposal metric-specific interval methods are invalid")
    repetition_counts = (proposal.get("metric_counts") or {}).get(
        "repeated_sentence_start_rate"
    ) or {}
    successes, trials = repetition_counts.get("successes"), repetition_counts.get("trials")
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("calibration proposal repetition counts must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise ValueError("calibration proposal repetition counts must be integers")
    if trials != int(proposal["sample_count"]):
        raise ValueError("calibration proposal repetition trial count must equal sample_count")
    point = float((proposal.get("point_estimates") or {}).get("repeated_sentence_start_rate"))
    if not math.isclose(point, successes / trials, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("calibration proposal repetition point estimate disagrees with counts")
    expected_repetition_interval = _wilson_interval(
        successes, trials, CALIBRATION_CONFIDENCE_LEVEL
    )
    actual_repetition_interval = _validated_interval(
        intervals.get("repeated_sentence_start_rate"),
        "intervals.repeated_sentence_start_rate",
    )
    if any(
        not math.isclose(actual_repetition_interval[key], expected_repetition_interval[key], rel_tol=0.0, abs_tol=1e-15)
        for key in ("low", "high")
    ):
        raise ValueError("calibration proposal repetition interval is not the Wilson interval")
    names = (
        "self_bleu",
        "repeated_sentence_start_rate",
        "non_target_script_char_rate",
        "paragraph_len_tokens",
        "sentence_len_tokens",
    )
    target = {
        "artifact_schema": "harness.calibration.v3",
        "frozen": True,
        "source_proposal_sha256": actual_sha256,
        "source_human_split_sha256": str(proposal["human_split_sha256"]),
        "source_sample_count": int(proposal["sample_count"]),
        "interval_methods": expected_methods,
        "metric_counts": proposal["metric_counts"],
        "confidence_level": float(proposal["confidence_level"]),
        "split_hashes": proposal["split_hashes"],
        "resampling_seeds": [int(seed) for seed in proposal["resampling_seeds"]],
        "operator_review_required": True,
    }
    target.update({name: _validated_interval(intervals.get(name), f"intervals.{name}") for name in names})
    return target


def prepare_baseline_transfer(proposal_path: str, expected_sha256: str) -> dict:
    """Validate a default-sampler baseline proposal for operator transfer."""
    path = Path(proposal_path)
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError("baseline proposal SHA-256 does not match operator-reviewed bytes")
    proposal = _read_structured(path)
    if proposal.get("artifact_schema") != "m1.baseline_stats.review.v1":
        raise ValueError("unsupported baseline proposal schema")
    if proposal.get("baseline_sampler_id") != "default_t1.0_p1.0":
        raise ValueError("bootstrap baseline must use preregistered default sampler")
    if int(proposal.get("sample_count", 0)) < 2:
        raise ValueError("baseline proposal requires at least two independent reports")
    target = {
        "artifact_schema": "harness.baseline_stats.v1",
        "frozen": True,
        "source_proposal_sha256": actual_sha256,
        "baseline_sampler_id": proposal["baseline_sampler_id"],
        "sample_count": int(proposal["sample_count"]),
        "train_split_hash": str(proposal["train_split_hash"]),
        "dev_split_hash": str(proposal["dev_split_hash"]),
        "human_reference_bank_id": str(proposal["human_reference_bank_id"]),
        "calibration_sha256": str(proposal["calibration_sha256"]),
        "operator_review_required": True,
    }
    for name in ("semantic_mmd", "lexical_l2", "structural_dist"):
        section = proposal.get(name)
        if not isinstance(section, dict):
            raise ValueError(f"baseline proposal is missing {name}")
        mean, std = float(section.get("mean")), float(section.get("std"))
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"baseline proposal {name} mean/std is invalid")
        target[name] = {"mean": mean, "std": std}
    for name in ("outline_fact_recall", "unsupported_claim_rate"):
        section = proposal.get(name)
        mean = float(section.get("mean")) if isinstance(section, dict) else math.nan
        if not math.isfinite(mean):
            raise ValueError(f"baseline proposal {name} mean is invalid")
        target[name] = {"mean": mean}
    return target


def _sealed_metadata(checkpoint_dir):
    _, config = _find_config(checkpoint_dir)
    arm = _first_config_value(config, {"arm"})
    comparison_id = _first_config_value(config, {"comparison_id"})
    train_embedder = _first_config_value(
        config,
        {"train_embedder_id", "reward_embedder_id", "mmd_embedder_id"},
        ("train_embedder", "reward_embedder", "mmd_embedder"),
    )
    artifact_uri = os.environ.get("SEALED_ARTIFACT_URI") or _first_config_value(config, {"artifact_uri"})
    artifact_uri = artifact_uri or str(Path(checkpoint_dir).resolve())
    missing = [name for name, value in (("arm", arm), ("comparison_id", comparison_id), ("train_embedder_id", train_embedder)) if not value]
    if missing:
        raise ValueError("checkpoint config missing sealed-submit metadata: " + ", ".join(missing))
    if str(arm) not in {"A", "B1", "B2", "C", "D", "E", "SFT"}:
        raise ValueError(f"invalid arm for sealed submission: {arm}")
    return str(artifact_uri), str(arm), str(train_embedder), str(comparison_id)


def sealed_submit(checkpoint_dir: str) -> dict:
    """Submit only contract-approved aggregate metadata to the Tier 2 service."""
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise ValueError("sealed-submit requires a checkpoint directory")
    url, token = os.environ.get("SEALED_EVAL_URL"), os.environ.get("SEALED_EVAL_TOKEN")
    if not url or not token:
        raise RuntimeError("SEALED_EVAL_URL and SEALED_EVAL_TOKEN must be set")
    artifact_uri, arm, train_embedder, comparison_id = _sealed_metadata(directory)
    endpoint = url.rstrip("/") if url.rstrip("/").endswith("/submit") else url.rstrip("/") + "/submit"
    payload = {
        "checkpoint_hash": _ckpt_hash(str(directory)),
        "artifact_uri": artifact_uri,
        "arm": arm,
        "train_embedder_id": train_embedder,
        "comparison_id": comparison_id,
    }
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.RequestException as error:
        raise RuntimeError("sealed evaluator request failed") from error
    if response.status_code == 429:
        raise RuntimeError("sealed evaluator weekly quota exhausted")
    if response.status_code == 409:
        raise RuntimeError("sealed evaluator rejected an embedder independence violation")
    try:
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError("sealed evaluator returned an invalid response") from error
    required = {"window_id", "quota_remaining", "primary", "authorship_auc", "authorship_auc_ci", "gates", "verdict", "aggregate_only"}
    if (
        not isinstance(result, dict)
        or set(result) != required
        or result.get("aggregate_only") is not True
        or not isinstance(result.get("primary"), dict)
        or set(result["primary"])
        != {"semantic_mmd", "semantic_mmd_delta_vs_floor", "S"}
    ):
        raise RuntimeError("sealed evaluator response violated the aggregate-only API contract")
    if result.get("verdict") not in {"confirm", "reject", "inconclusive"}:
        raise RuntimeError("sealed evaluator returned an invalid verdict")
    return result


def _hash_piece(hasher, kind: bytes, name: str, payload: bytes = b""):
    encoded = name.encode("utf-8", "surrogateescape")
    hasher.update(kind)
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _ckpt_hash(path: str) -> str:
    """Stable, path-aware content hash that never follows symlinks."""
    root = Path(path)
    if not root.exists() and not root.is_symlink():
        raise FileNotFoundError(path)
    hasher = hashlib.sha256()
    if root.is_symlink():
        _hash_piece(hasher, b"L", root.name, os.readlink(root).encode())
    elif root.is_file():
        _hash_piece(hasher, b"F", root.name)
        with root.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
    elif root.is_dir():
        _hash_piece(hasher, b"D", ".")
        for item in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
            relative = item.relative_to(root).as_posix()
            if item.is_symlink():
                _hash_piece(hasher, b"L", relative, os.readlink(item).encode())
            elif item.is_dir():
                _hash_piece(hasher, b"D", relative)
            elif item.is_file():
                stat = item.stat()
                _hash_piece(hasher, b"F", relative, stat.st_size.to_bytes(8, "big"))
                with item.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        hasher.update(chunk)
            else:
                raise ValueError(f"unsupported special file in checkpoint: {relative}")
    else:
        raise ValueError(f"unsupported checkpoint path: {path}")
    return hasher.hexdigest()[:16]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    evaluate_parser = sub.add_parser("eval", help="Tier 1 dev evaluation")
    evaluate_parser.add_argument("target", help="checkpoint dir or samples.jsonl")
    evaluate_parser.add_argument("--report", default=None)
    sealed_parser = sub.add_parser("sealed-submit", help="Tier 2 promotion check (quota-limited)")
    sealed_parser.add_argument("checkpoint_dir")
    calibration_parser = sub.add_parser("calibrate", help="M1: build calibration.json from human splits")
    calibration_parser.add_argument("human_split_jsonl")
    calibration_transfer = sub.add_parser(
        "prepare-calibration-transfer", help="validate proposal and emit operator-reviewed calibration"
    )
    calibration_transfer.add_argument("proposal")
    calibration_transfer.add_argument("--expected-sha256", required=True)
    baseline_transfer = sub.add_parser(
        "prepare-baseline-transfer", help="validate proposal and emit operator-reviewed baseline"
    )
    baseline_transfer.add_argument("proposal")
    baseline_transfer.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "eval":
            output = asdict(evaluate(args.target, args.report))
        elif args.cmd == "sealed-submit":
            output = sealed_submit(args.checkpoint_dir)
        elif args.cmd == "calibrate":
            output = calibrate(args.human_split_jsonl)
        elif args.cmd == "prepare-calibration-transfer":
            output = prepare_calibration_transfer(args.proposal, args.expected_sha256)
        else:
            output = prepare_baseline_transfer(args.proposal, args.expected_sha256)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"harness: {error}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
