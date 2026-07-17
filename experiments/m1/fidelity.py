from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Callable

from .contracts import (
    M1ConfigError,
    build_run_paths,
    canonical_hash,
    file_sha256,
    git_sha,
    load_jsonl,
    read_structured,
    resolve_repo_path,
    write_json,
    write_jsonl,
)


REPLAY_SCHEMA = "dftr.adapter_merge_replay.v1"
CONTRACT_SCHEMA = "dftr.canonical_generation.v1"
EXACT_SAMPLING_SEEDS = [101, 202, 303]
TOKENIZER_FILES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
PAID_OR_HIDDEN_KEYS = {
    "api",
    "api_key",
    "judge",
    "provider",
    "sealed",
    "sealed_eval_url",
    "hidden",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _hash_piece(hasher: Any, kind: bytes, name: str, payload: bytes = b"") -> None:
    encoded = name.encode("utf-8", "surrogateescape")
    hasher.update(kind)
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def canonical_directory_hash(path: str | Path) -> str:
    """Match the public sealed client's stable, path-aware 16-hex directory hash."""
    root = Path(path)
    if not root.exists() and not root.is_symlink():
        raise M1ConfigError(f"artifact path does not exist: {root}")
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
                _hash_piece(hasher, b"F", relative, item.stat().st_size.to_bytes(8, "big"))
                with item.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        hasher.update(chunk)
            else:
                raise M1ConfigError(f"unsupported special file in artifact: {relative}")
    else:
        raise M1ConfigError(f"unsupported artifact path: {root}")
    return hasher.hexdigest()[:16]


def hash_token_ids(values: list[int]) -> str:
    payload = b"".join(struct.pack(">q", int(value)) for value in values)
    return _sha256_bytes(payload)


def derive_record_seed(global_seed: int, fingerprint: str) -> int:
    if global_seed < 0 or not fingerprint:
        raise M1ConfigError("record seed derivation requires a nonnegative seed and fingerprint")
    payload = int(global_seed).to_bytes(8, "big") + b"\0" + fingerprint.encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key).casefold())
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def assert_public_only_config(config: dict[str, Any]) -> None:
    forbidden = sorted(set(_walk_keys(config)) & PAID_OR_HIDDEN_KEYS)
    if forbidden:
        raise M1ConfigError("replay config contains paid or hidden surfaces: " + ", ".join(forbidden))
    run = config.get("run") or {}
    if str(run.get("task_kind", "experiment")) != "experiment":
        raise M1ConfigError("replay must use the credential-free experiment task kind")
    command = run.get("command", ["python", "-m", "experiments.runner"])
    if command[:3] != ["python", "-m", "experiments.runner"]:
        raise M1ConfigError("replay command is outside the public runner allowlist")


def _require_sha(value: Any, field: str, *, length: int = 64) -> str:
    text = str(value or "")
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise M1ConfigError(f"{field} must be an exact lowercase {length}-hex digest")
    return text


def load_generation_contract(config: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    workflow = config.get("workflow") or {}
    path = resolve_repo_path(str(workflow.get("generation_contract") or ""))
    if not path.is_file():
        raise M1ConfigError("workflow.generation_contract is required")
    expected = _require_sha(workflow.get("generation_contract_sha256"), "generation contract SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise M1ConfigError("generation contract SHA-256 mismatch")
    contract = read_structured(path)
    if contract.get("contract_schema") != CONTRACT_SCHEMA:
        raise M1ConfigError("unexpected generation contract schema")
    expected_shape = {
        "dtype": "bfloat16",
        "prompt_schema": "dft.full-brief.v1",
        "prompt_format": "USER:\n{brief}\nASSISTANT:",
        "add_special_tokens": True,
        "padding_side": "left",
        "truncation_side": "right",
        "max_input_tokens": 1024,
        "max_new_tokens": 384,
        "rng_derivation": "sha256_u64be(global_seed_nul_fingerprint)_mod_2^63",
    }
    observed_shape = {
        "dtype": contract.get("dtype"),
        "prompt_schema": (contract.get("prompt") or {}).get("schema"),
        "prompt_format": (contract.get("prompt") or {}).get("format"),
        "add_special_tokens": (contract.get("tokenization") or {}).get("add_special_tokens"),
        "padding_side": (contract.get("tokenization") or {}).get("padding_side"),
        "truncation_side": (contract.get("tokenization") or {}).get("truncation_side"),
        "max_input_tokens": (contract.get("tokenization") or {}).get("max_input_tokens"),
        "max_new_tokens": (contract.get("prospective_generation") or {}).get("max_new_tokens"),
        "rng_derivation": (contract.get("prospective_generation") or {}).get("rng_derivation"),
    }
    if observed_shape != expected_shape:
        raise M1ConfigError("generation contract does not match the canonical replay interface")
    return contract, path, observed


def validate_replay_spec(config: dict[str, Any]) -> tuple[list[str], list[int]]:
    assert_public_only_config(config)
    workflow = config.get("workflow") or {}
    if workflow.get("step") != "replay_equivalence" or workflow.get("protocol_version") != REPLAY_SCHEMA:
        raise M1ConfigError("replay requires the exact replay workflow schema and step")
    fingerprints = list((config.get("sampling") or {}).get("dev_subset_fingerprints") or [])
    if len(fingerprints) != 16 or len(set(fingerprints)) != 16:
        raise M1ConfigError("replay requires exactly 16 unique fingerprints")
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in fingerprints):
        raise M1ConfigError("replay fingerprints must be lowercase SHA-256 values")
    subset_hash = _sha256_bytes("\n".join(sorted(fingerprints)).encode("utf-8"))
    if subset_hash != _require_sha((config.get("sampling") or {}).get("dev_subset_hash"), "subset hash"):
        raise M1ConfigError("replay subset hash mismatch")
    seeds = list((config.get("sampling") or {}).get("seeds") or [])
    if seeds != EXACT_SAMPLING_SEEDS:
        raise M1ConfigError("replay sampling seeds must be exactly [101, 202, 303]")
    _require_sha(workflow.get("fixed_manifest_sha256"), "fixed manifest SHA-256")
    _require_sha((config.get("artifacts") or {}).get("adapter_manifest_sha256"), "adapter manifest SHA-256")
    _require_sha((config.get("artifacts") or {}).get("adapter_sha256"), "adapter SHA-256")
    _require_sha((config.get("artifacts") or {}).get("merged_content_hash"), "merged content hash", length=16)
    return fingerprints, seeds


def _verify_file_map(root: Path, expected: dict[str, Any], label: str) -> dict[str, str]:
    if not isinstance(expected, dict) or not expected:
        raise M1ConfigError(f"{label} expected file map is required")
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        expected_sha = _require_sha(digest, f"{label} {relative} SHA-256")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise M1ConfigError(f"{label} file is missing or symlinked: {relative}")
        observed[str(relative)] = file_sha256(path)
        if observed[str(relative)] != expected_sha:
            raise M1ConfigError(f"{label} file SHA-256 mismatch: {relative}")
    return observed


def _tokenizer_file_map(root: Path) -> dict[str, str]:
    return {
        name: file_sha256(root / name)
        for name in sorted(TOKENIZER_FILES)
        if (root / name).is_file() and not (root / name).is_symlink()
    }


def verify_artifact_identities(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config.get("artifacts") or {}
    adapter_dir = Path(str(artifacts.get("adapter_path") or ""))
    merged_dir = Path(str(artifacts.get("merged_path") or ""))
    checkpoint_root = Path("/checkpoints/runs").resolve()
    for label, path in (("adapter", adapter_dir), ("merged", merged_dir)):
        try:
            path.resolve().relative_to(checkpoint_root)
        except (ValueError, OSError) as exc:
            raise M1ConfigError(f"{label} artifact must be under /checkpoints/runs") from exc
        if not path.is_dir():
            raise M1ConfigError(f"{label} artifact directory is missing")

    manifest_path = Path(str(artifacts.get("adapter_manifest") or ""))
    if file_sha256(manifest_path) != str(artifacts.get("adapter_manifest_sha256")):
        raise M1ConfigError("adapter checkpoint manifest SHA-256 mismatch")
    manifest = read_structured(manifest_path)
    seed = int(artifacts.get("adapter_seed", -1))
    rows = [row for row in manifest.get("checkpoints", []) if int(row.get("seed", -1)) == seed]
    if len(rows) != 1 or Path(str(rows[0].get("checkpoint_dir"))).resolve() != adapter_dir.resolve():
        raise M1ConfigError("adapter manifest does not bind the exact seed and directory")
    adapter_files = dict(rows[0].get("checkpoint_files") or {})
    if adapter_files.get("adapter_model.safetensors") != artifacts.get("adapter_sha256"):
        raise M1ConfigError("adapter manifest/model SHA-256 binding mismatch")
    observed_adapter_files = _verify_file_map(adapter_dir, adapter_files, "adapter")
    merged_files = _verify_file_map(
        merged_dir, dict(artifacts.get("merged_weight_files") or {}), "merged"
    )
    merged_content_hash = canonical_directory_hash(merged_dir)
    if merged_content_hash != str(artifacts.get("merged_content_hash")):
        raise M1ConfigError("merged directory content hash mismatch")
    adapter_tokenizer = _tokenizer_file_map(adapter_dir)
    merged_tokenizer = _tokenizer_file_map(merged_dir)
    if not adapter_tokenizer or adapter_tokenizer != merged_tokenizer:
        raise M1ConfigError("adapter and merged tokenizer file identities differ")
    return {
        "adapter": {
            "path": str(adapter_dir.resolve()),
            "files": observed_adapter_files,
            "tensor_identity_sha256": _canonical_json_sha256(
                {"adapter_model.safetensors": observed_adapter_files["adapter_model.safetensors"]}
            ),
        },
        "merged": {
            "path": str(merged_dir.resolve()),
            "content_hash": merged_content_hash,
            "weight_files": merged_files,
            "tensor_identity_sha256": _canonical_json_sha256(merged_files),
        },
        "tokenizer": {
            "files": adapter_tokenizer,
            "identity_sha256": _canonical_json_sha256(adapter_tokenizer),
        },
    }


def _library_versions() -> dict[str, str]:
    import peft
    import torch
    import transformers

    return {
        "peft": str(peft.__version__),
        "python": os.sys.version,
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
    }


def _load_models(config: dict[str, Any], contract: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    artifacts = config["artifacts"]
    model_config = config["model"]
    adapter_path = str(artifacts["adapter_path"])
    merged_path = str(artifacts["merged_path"])
    common = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    adapter_base = AutoModelForCausalLM.from_pretrained(
        str(model_config["base"]), revision=str(model_config["revision"]), **common
    )
    adapter_model = PeftModel.from_pretrained(adapter_base, adapter_path, local_files_only=True)
    merged_model = AutoModelForCausalLM.from_pretrained(merged_path, **common)
    adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True, trust_remote_code=True)
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_path, local_files_only=True, trust_remote_code=True)
    for tokenizer in (adapter_tokenizer, merged_tokenizer):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = str(contract["tokenization"]["padding_side"])
        tokenizer.truncation_side = str(contract["tokenization"]["truncation_side"])
    adapter_model.eval()
    merged_model.eval()
    return adapter_model, merged_model, adapter_tokenizer, merged_tokenizer


def _tokenize_prompts(
    records: list[dict[str, Any]],
    tokenizer: Any,
    contract: dict[str, Any],
    render_prompt: Callable[[dict[str, Any], str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenization = contract["tokenization"]
    prompt_contract = contract["prompt"]
    rows: list[dict[str, Any]] = []
    encoded_rows: list[dict[str, Any]] = []
    for record in records:
        prompt = render_prompt(record, prompt_contract["format"], prompt_contract["schema"])
        kwargs = {
            "add_special_tokens": bool(tokenization["add_special_tokens"]),
            "return_attention_mask": True,
        }
        full = tokenizer(prompt, truncation=False, **kwargs)
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=int(tokenization["max_input_tokens"]),
            **kwargs,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        mask = [int(value) for value in encoded["attention_mask"]]
        rows.append({
            "fingerprint": str(record["fingerprint"]),
            "prompt": prompt,
            "prompt_utf8_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "input_ids": ids,
            "input_ids_sha256": hash_token_ids(ids),
            "attention_mask": mask,
            "attention_mask_sha256": hash_token_ids(mask),
            "untruncated_token_count": len(full["input_ids"]),
            "truncated_token_count": len(ids),
            "tokens_removed_by_right_truncation": max(0, len(full["input_ids"]) - len(ids)),
        })
        encoded_rows.append({"input_ids": ids, "attention_mask": mask})
    return rows, encoded_rows


def _generate_one(
    model: Any,
    tokenizer: Any,
    encoded: dict[str, Any],
    *,
    generation: dict[str, Any],
    generator: Any | None = None,
) -> tuple[list[int], str]:
    import torch

    device = next(model.parameters()).device
    tensors = {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in encoded.items()
    }
    args = dict(generation)
    if generator is not None:
        args["generator"] = generator
    with torch.inference_mode():
        sequence = model.generate(**tensors, pad_token_id=tokenizer.pad_token_id, **args)[0]
    output_ids = [int(value) for value in sequence[tensors["input_ids"].shape[1]:].tolist()]
    return output_ids, tokenizer.decode(output_ids, skip_special_tokens=True)


def _run_diagnostics(
    adapter_model: Any,
    merged_model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    encoded_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    import torch

    diagnostics = contract["diagnostics"]
    greedy_rows = []
    total_abs = 0.0
    max_abs = 0.0
    compared = 0
    top1_equal = 0
    for record, encoded in zip(records, encoded_rows):
        generation = {
            "max_new_tokens": int(diagnostics["greedy_max_new_tokens"]),
            "do_sample": False,
        }
        adapter_ids, _ = _generate_one(adapter_model, tokenizer, encoded, generation=generation)
        merged_ids, _ = _generate_one(merged_model, tokenizer, encoded, generation=generation)
        greedy_rows.append({
            "fingerprint": record["fingerprint"],
            "adapter_token_ids": adapter_ids,
            "merged_token_ids": merged_ids,
            "exact": adapter_ids == merged_ids,
        })

        reference_ids = tokenizer(
            str(record["completion"]),
            add_special_tokens=False,
            truncation=True,
            max_length=int(diagnostics["teacher_forced_max_tokens"]),
        )["input_ids"]
        if not reference_ids:
            raise M1ConfigError("teacher-forced reference completion tokenized to zero tokens")
        joined = encoded["input_ids"] + [int(value) for value in reference_ids]
        mask = encoded["attention_mask"] + [1] * len(reference_ids)
        device = next(adapter_model.parameters()).device
        input_tensor = torch.tensor([joined], dtype=torch.long, device=device)
        mask_tensor = torch.tensor([mask], dtype=torch.long, device=device)
        with torch.inference_mode():
            adapter_logits = adapter_model(input_ids=input_tensor, attention_mask=mask_tensor).logits
            merged_logits = merged_model(input_ids=input_tensor, attention_mask=mask_tensor).logits
        start = len(encoded["input_ids"]) - 1
        stop = start + len(reference_ids)
        adapter_slice = adapter_logits[:, start:stop, :].float()
        merged_slice = merged_logits[:, start:stop, :].float()
        difference = (adapter_slice - merged_slice).abs()
        total_abs += float(difference.sum().item())
        compared += int(difference.numel())
        max_abs = max(max_abs, float(difference.max().item()))
        top1_equal += int((adapter_slice.argmax(-1) == merged_slice.argmax(-1)).sum().item())
    teacher_tokens = sum(
        min(
            len(tokenizer(str(record["completion"]), add_special_tokens=False)["input_ids"]),
            int(diagnostics["teacher_forced_max_tokens"]),
        )
        for record in records
    )
    result = {
        "greedy": greedy_rows,
        "greedy_exact_count": sum(row["exact"] for row in greedy_rows),
        "greedy_total": len(greedy_rows),
        "teacher_forced": {
            "mean_absolute_logit_difference": total_abs / compared,
            "maximum_absolute_logit_difference": max_abs,
            "top1_agreement": top1_equal / teacher_tokens,
            "reference_token_count": teacher_tokens,
            "logit_element_count": compared,
        },
    }
    thresholds = diagnostics["thresholds"]
    result["gate_passed"] = bool(
        result["greedy_exact_count"] == result["greedy_total"]
        and result["teacher_forced"]["mean_absolute_logit_difference"]
        <= float(thresholds["mean_absolute_logit_difference_max"])
        and result["teacher_forced"]["maximum_absolute_logit_difference"]
        <= float(thresholds["maximum_absolute_logit_difference_max"])
        and result["teacher_forced"]["top1_agreement"] >= float(thresholds["top1_agreement_min"])
    )
    return result


def _historical_adapter_replay(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    encoded_rows: list[dict[str, Any]],
    config: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    import torch

    archive = config["archive"]
    rows: list[dict[str, Any]] = []
    all_exact = True
    generation = dict(contract["prospective_generation"])
    generation = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": True,
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
    }
    record_by_fingerprint = {str(record["fingerprint"]): record for record in records}
    encoded_by_fingerprint = {
        str(record["fingerprint"]): encoded
        for record, encoded in zip(records, encoded_rows)
    }
    fingerprint_by_fineweb_id = {
        str(record["fineweb_id"]): str(record["fingerprint"])
        for record in records
    }
    for global_seed in EXACT_SAMPLING_SEEDS:
        torch.manual_seed(global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(global_seed)
        path = Path(str(archive["sample_paths"][str(global_seed)]))
        expected_file_sha = str(archive["sample_sha256"][str(global_seed)])
        if file_sha256(path) != expected_file_sha:
            raise M1ConfigError(f"archived sample SHA-256 mismatch for seed {global_seed}")
        archived = load_jsonl(path)
        if len(archived) != len(records):
            raise M1ConfigError("archived adapter sample cardinality mismatch")
        archived_fingerprints = [
            str(old.get("fingerprint") or fingerprint_by_fineweb_id.get(str(old.get("fineweb_id")), ""))
            for old in archived
        ]
        if set(archived_fingerprints) != set(record_by_fingerprint) or len(set(archived_fingerprints)) != len(records):
            raise M1ConfigError("archived adapter sample does not bind the exact frozen records")
        for fingerprint, old in zip(archived_fingerprints, archived):
            record = record_by_fingerprint[fingerprint]
            encoded = encoded_by_fingerprint[fingerprint]
            output_ids, output = _generate_one(model, tokenizer, encoded, generation=generation)
            archived_output = str(old.get("generated_completion", old.get("output", "")))
            exact = output.encode("utf-8") == archived_output.encode("utf-8")
            all_exact = all_exact and exact
            rows.append({
                "global_seed": global_seed,
                "fingerprint": record["fingerprint"],
                "output": output,
                "output_token_ids": output_ids,
                "output_utf8_sha256": _sha256_bytes(output.encode("utf-8")),
                "archived_output_utf8_sha256": _sha256_bytes(archived_output.encode("utf-8")),
                "exact_archive_match": exact,
            })
    return rows, all_exact


def _prospective_pairs(
    adapter_model: Any,
    merged_model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    encoded_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    import torch

    prospective = contract["prospective_generation"]
    generation = {
        "max_new_tokens": int(prospective["max_new_tokens"]),
        "do_sample": True,
        "temperature": float(prospective["temperature"]),
        "top_p": float(prospective["top_p"]),
    }
    rows: list[dict[str, Any]] = []
    all_exact = True
    device = next(adapter_model.parameters()).device
    generator_device = device.type if hasattr(device, "type") else str(device)
    for global_seed in EXACT_SAMPLING_SEEDS:
        for record, encoded in zip(records, encoded_rows):
            fingerprint = str(record["fingerprint"])
            record_seed = derive_record_seed(global_seed, fingerprint)
            adapter_generator = torch.Generator(device=generator_device).manual_seed(record_seed)
            merged_generator = torch.Generator(device=generator_device).manual_seed(record_seed)
            adapter_ids, adapter_output = _generate_one(
                adapter_model, tokenizer, encoded, generation=generation, generator=adapter_generator
            )
            merged_ids, merged_output = _generate_one(
                merged_model, tokenizer, encoded, generation=generation, generator=merged_generator
            )
            exact = adapter_ids == merged_ids and adapter_output.encode("utf-8") == merged_output.encode("utf-8")
            all_exact = all_exact and exact
            rows.append({
                "global_seed": global_seed,
                "record_seed": record_seed,
                "fingerprint": fingerprint,
                "adapter_output": adapter_output,
                "merged_output": merged_output,
                "adapter_output_token_ids": adapter_ids,
                "merged_output_token_ids": merged_ids,
                "adapter_output_utf8_sha256": _sha256_bytes(adapter_output.encode("utf-8")),
                "merged_output_utf8_sha256": _sha256_bytes(merged_output.encode("utf-8")),
                "exact": exact,
            })
    return rows, all_exact


def replay_equivalence(
    config: dict[str, Any],
    run_id: str,
    *,
    render_prompt: Callable[[dict[str, Any], str, str], str],
) -> dict[str, Any]:
    fingerprints, seeds = validate_replay_spec(config)
    contract, contract_path, contract_sha = load_generation_contract(config)
    workflow = config["workflow"]
    historical_config_path = resolve_repo_path(str(workflow["historical_sampling_config"]))
    if file_sha256(historical_config_path) != str(workflow["historical_sampling_config_sha256"]):
        raise M1ConfigError("historical sampling config SHA-256 mismatch")
    fixed_path = resolve_repo_path(str(workflow["fixed_manifest"]))
    if file_sha256(fixed_path) != str(workflow["fixed_manifest_sha256"]):
        raise M1ConfigError("fixed manifest SHA-256 mismatch")
    fixed = read_structured(fixed_path)
    data = config.get("data") or {}
    dev_path = Path(str(data.get("dev_path") or ""))
    if file_sha256(dev_path) != str(data.get("dev_briefs_sha256")):
        raise M1ConfigError("dev briefs SHA-256 mismatch")
    if str(data.get("dev_briefs_sha256")) != str(fixed.get("dev_briefs_sha256")):
        raise M1ConfigError("dev briefs/fixed manifest identity mismatch")
    dev_records = load_jsonl(dev_path)
    index = {str(record.get("fingerprint") or ""): record for record in dev_records}
    if any(fingerprint not in index for fingerprint in fingerprints):
        raise M1ConfigError("a frozen replay fingerprint is absent from the dev split")
    records = [index[fingerprint] for fingerprint in fingerprints]
    archive_index = resolve_repo_path(str(config["archive"]["index_path"]))
    if file_sha256(archive_index) != str(config["archive"]["index_sha256"]):
        raise M1ConfigError("archived Tier-1 index SHA-256 mismatch")
    identities = verify_artifact_identities(config)
    serializer_path = resolve_repo_path(str(workflow.get("serializer_source") or ""))
    if file_sha256(serializer_path) != str(workflow.get("serializer_source_sha256")):
        raise M1ConfigError("serializer source SHA-256 mismatch")

    import torch

    torch.use_deterministic_algorithms(True)
    output_dir, checkpoint_dir = build_run_paths(config, run_id)
    adapter_model, merged_model, adapter_tokenizer, merged_tokenizer = _load_models(config, contract)
    adapter_chat_template = str(adapter_tokenizer.chat_template or "")
    merged_chat_template = str(merged_tokenizer.chat_template or "")
    if adapter_chat_template != merged_chat_template:
        raise M1ConfigError("adapter and merged tokenizer chat templates differ")
    identities["tokenizer"]["chat_template_utf8_sha256"] = _sha256_bytes(
        adapter_chat_template.encode("utf-8")
    )
    prompt_rows, encoded_rows = _tokenize_prompts(records, adapter_tokenizer, contract, render_prompt)
    merged_prompt_rows, merged_encoded_rows = _tokenize_prompts(
        records, merged_tokenizer, contract, render_prompt
    )
    if encoded_rows != merged_encoded_rows or prompt_rows != merged_prompt_rows:
        raise M1ConfigError("adapter and merged prompt/tokenization attestations differ")
    write_jsonl(checkpoint_dir / "prompt_attestations.jsonl", prompt_rows)

    diagnostics = _run_diagnostics(
        adapter_model, merged_model, adapter_tokenizer, records, encoded_rows, contract
    )
    write_json(checkpoint_dir / "deterministic_diagnostics.json", diagnostics)
    historical_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    archive_passed = False
    pair_passed = False
    if diagnostics["gate_passed"]:
        historical_rows, archive_passed = _historical_adapter_replay(
            adapter_model, adapter_tokenizer, records, encoded_rows, config, contract
        )
        write_jsonl(checkpoint_dir / "historical_adapter_replay.jsonl", historical_rows)
        if archive_passed:
            pair_rows, pair_passed = _prospective_pairs(
                adapter_model,
                merged_model,
                adapter_tokenizer,
                records,
                encoded_rows,
                contract,
            )
            write_jsonl(checkpoint_dir / "prospective_adapter_merged_pairs.jsonl", pair_rows)
    passed = bool(diagnostics["gate_passed"] and archive_passed and pair_passed)
    result = {
        "artifact_schema": "dftr.adapter_merge_replay_result.v1",
        "run_id": run_id,
        "comparison_id": str(config["run"]["comparison_id"]),
        "status": "completed",
        "verdict": "pass" if passed else "fail",
        "interpretation_allowed": bool(archive_passed),
        "config_hash": canonical_hash(config),
        "git_sha": git_sha(),
        "model": {
            "base": config["model"]["base"],
            "revision": config["model"]["revision"],
            "dtype": contract["dtype"],
            "adapter_parameter_dtype": str(next(adapter_model.parameters()).dtype),
            "merged_parameter_dtype": str(next(merged_model.parameters()).dtype),
        },
        "identities": identities,
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
            "generation_config_sha256": _canonical_json_sha256(
                contract["prospective_generation"]
            ),
            "serializer_source": str(serializer_path),
            "serializer_source_sha256": file_sha256(serializer_path),
        },
        "libraries": _library_versions(),
        "frozen_inputs": {
            "count": len(fingerprints),
            "fingerprints": fingerprints,
            "sampling_seeds": seeds,
            "dev_briefs_sha256": file_sha256(dev_path),
            "subset_hash": str((config.get("sampling") or {})["dev_subset_hash"]),
        },
        "checks": {
            "deterministic_gate_passed": diagnostics["gate_passed"],
            "archive_exact_count": sum(row.get("exact_archive_match", False) for row in historical_rows),
            "archive_total": len(historical_rows),
            "archive_reproduction_passed": archive_passed,
            "paired_exact_count": sum(row.get("exact", False) for row in pair_rows),
            "paired_total": len(pair_rows),
            "paired_stochastic_passed": pair_passed,
        },
        "artifacts": {
            "prompt_attestations": str((checkpoint_dir / "prompt_attestations.jsonl").resolve()),
            "deterministic_diagnostics": str((checkpoint_dir / "deterministic_diagnostics.json").resolve()),
            "historical_adapter_replay": str((checkpoint_dir / "historical_adapter_replay.jsonl").resolve()) if historical_rows else None,
            "prospective_pairs": str((checkpoint_dir / "prospective_adapter_merged_pairs.jsonl").resolve()) if pair_rows else None,
        },
        "token_accounting": {"total_tokens": 0},
    }
    for base in (output_dir, checkpoint_dir):
        write_json(base / "run_manifest.json", result)
        write_json(base / "config.json", config)
    return result
