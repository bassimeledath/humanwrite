from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
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
REPLAY_SCHEMA_V2 = "dftr.adapter_merge_replay.v2"
REPLAY_SCHEMAS = {REPLAY_SCHEMA, REPLAY_SCHEMA_V2}
REPLAY_COMPARISON_V1 = "M2-adapter-merge-fidelity-replay-v1"
REPLAY_COMPARISON_V2 = "M2-adapter-merge-fidelity-replay-v2"
CANONICAL_REPLAY_V1_CONFIG_PATH = "configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml"
CANONICAL_REPLAY_V1_CONFIG_SHA256 = (
    "8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c"
)
CANONICAL_REPLAY_V1_CONFIG_HASH = (
    "859798f2ce66b81a2db32665b7f8fda5a76f5d9e82c64789e7e1f797c4587b9f"
)
CONTRACT_SCHEMA = "dftr.canonical_generation.v1"
REPLAY_TRANSFORMERS_VERSION = "4.57.6"
CANONICAL_GENERATION_CONTRACT_PATH = "configs/m2/canonical_full_brief_generation_v1.json"
CANONICAL_GENERATION_CONTRACT_SHA256 = (
    "db7c970440c451ffd21e634b53df3fa3d556b139e87257dfff7521442fe8f219"
)
CANONICAL_HISTORICAL_CONFIG_PATH = (
    "configs/m1/m1_realdata_adherence_directional_qwen3_4b_three_seed_v1.yaml"
)
CANONICAL_HISTORICAL_CONFIG_SHA256 = (
    "a02d893eda4c5e457864e1145e5cb4a4d238ab04037bc74e269a1ab20e52a72c"
)
FROZEN_SUBSET_HASH = "18a8031ed63cf72636523974000af05e1d8bdd16f351b4f5a13fbe3dcfefe9e3"
SNAPSHOT_IDENTITY_MANIFEST_PATH = (
    "configs/m2/manifests/m2_adapter_merge_snapshot_identity_v2.json"
)
SNAPSHOT_IDENTITY_MANIFEST_SHA256 = (
    "602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c"
)
ORIGINAL_MERGE_CONTENT_HASH_V2 = "7f095c31e83f8b03"
SUBMITTED_SNAPSHOT_CONTENT_HASH_V2 = "0f437f62bc1cca0c"
SNAPSHOT_METADATA_DIFFERENCE_FILES = ["generation_config.json", "train_config.json"]
EXACT_SAMPLING_SEEDS = [101, 202, 303]
TOKENIZER_FILES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
SENSITIVE_REPLAY_KEY_WORDS = {
    "api", "judge", "provider", "sealed", "hidden", "private",
    "credential", "credentials", "secret", "secrets",
    "auth", "authentication", "authorization", "key", "keys",
    "endpoint", "endpoints", "service", "services",
}
TOKEN_KEY_WORDS = {"token", "tokens"}
PUBLIC_TOKEN_METADATA_WORDS = {
    "add", "added", "additional", "all", "begin", "between", "bos", "cls",
    "count", "counts",
    "decoder", "eos", "exact", "forced", "generation", "greedy", "id", "ids",
    "image", "input", "map", "mask", "max", "min", "new", "output", "pad",
    "parity", "policy", "sep", "skip", "special", "split", "spaces", "start",
    "suppress", "teacher", "token", "tokens", "type", "types", "unk", "video",
    "vision", "extended", "extra", "healing",
}
PUBLIC_TOKEN_METADATA_MODIFIERS = PUBLIC_TOKEN_METADATA_WORDS - TOKEN_KEY_WORDS
KEY_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")


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


def _key_words(key: Any) -> tuple[str, ...]:
    words: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", str(key)):
        words.extend(match.casefold() for match in KEY_WORD_RE.findall(chunk))
    return tuple(words)


def _is_sensitive_replay_key(key: Any) -> bool:
    words = _key_words(key)
    word_set = set(words)
    if word_set & SENSITIVE_REPLAY_KEY_WORDS:
        return True
    # OAuth/OIDC uses ``id_token``; model metadata uses the inverse
    # ``*_token_id`` order (for example ``eos_token_id``).
    if words == ("id", "token"):
        return True
    if not word_set & TOKEN_KEY_WORDS:
        return False
    return not (
        word_set <= PUBLIC_TOKEN_METADATA_WORDS
        and bool(word_set & PUBLIC_TOKEN_METADATA_MODIFIERS)
    )


def _forbidden_surface_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_sensitive_replay_key(key):
                keys.append(str(key))
            keys.extend(_forbidden_surface_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_forbidden_surface_keys(child))
    return keys


def assert_public_only_config(config: dict[str, Any]) -> None:
    forbidden = sorted(set(_forbidden_surface_keys(config)))
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
    if str(workflow.get("generation_contract") or "") != CANONICAL_GENERATION_CONTRACT_PATH:
        raise M1ConfigError("replay requires the canonical generation contract path")
    if str(workflow.get("generation_contract_sha256") or "") != CANONICAL_GENERATION_CONTRACT_SHA256:
        raise M1ConfigError("replay requires the canonical generation contract SHA-256")
    path = resolve_repo_path(CANONICAL_GENERATION_CONTRACT_PATH)
    if not path.is_file():
        raise M1ConfigError("workflow.generation_contract is required")
    expected = CANONICAL_GENERATION_CONTRACT_SHA256
    observed = file_sha256(path)
    if observed != expected:
        raise M1ConfigError("generation contract SHA-256 mismatch")
    contract = read_structured(path)
    if contract.get("contract_schema") != CONTRACT_SCHEMA:
        raise M1ConfigError("unexpected generation contract schema")
    expected_shape = {
        "dtype": "bfloat16",
        "transformers_version": REPLAY_TRANSFORMERS_VERSION,
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
        "transformers_version": (contract.get("runtime") or {}).get(
            "transformers_version"
        ),
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


def load_snapshot_identity_audit(config: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    workflow = config.get("workflow") or {}
    if workflow.get("protocol_version") != REPLAY_SCHEMA_V2:
        raise M1ConfigError("snapshot identity audit is required only for replay protocol v2")
    audit_config = config.get("submitted_snapshot_audit") or {}
    if str(audit_config.get("identity_manifest") or "") != SNAPSHOT_IDENTITY_MANIFEST_PATH:
        raise M1ConfigError("replay v2 requires the canonical snapshot identity manifest path")
    if str(audit_config.get("identity_manifest_sha256") or "") != SNAPSHOT_IDENTITY_MANIFEST_SHA256:
        raise M1ConfigError("replay v2 requires the canonical snapshot identity manifest SHA-256")
    path = resolve_repo_path(SNAPSHOT_IDENTITY_MANIFEST_PATH)
    if not path.is_file() or file_sha256(path) != SNAPSHOT_IDENTITY_MANIFEST_SHA256:
        raise M1ConfigError("canonical snapshot identity manifest SHA-256 mismatch")
    manifest = read_structured(path)
    if manifest.get("artifact_schema") != "dftr.adapter_merge_snapshot_identity.v2":
        raise M1ConfigError("unexpected snapshot identity manifest schema")
    original = manifest.get("original_merge") or {}
    snapshot = manifest.get("submitted_snapshot") or {}
    relation = manifest.get("relation") or {}
    artifacts = config.get("artifacts") or {}
    if (
        original.get("path") != artifacts.get("merged_path")
        or original.get("canonical_directory_hash") != ORIGINAL_MERGE_CONTENT_HASH_V2
        or artifacts.get("merged_content_hash") != ORIGINAL_MERGE_CONTENT_HASH_V2
    ):
        raise M1ConfigError("replay v2 original merge identity mismatch")
    if (
        snapshot.get("canonical_directory_hash") != SUBMITTED_SNAPSHOT_CONTENT_HASH_V2
        or audit_config.get("canonical_directory_hash") != SUBMITTED_SNAPSHOT_CONTENT_HASH_V2
    ):
        raise M1ConfigError("replay v2 submitted snapshot identity mismatch")
    difference_files = list(relation.get("difference_files") or [])
    if difference_files != SNAPSHOT_METADATA_DIFFERENCE_FILES or list(
        audit_config.get("metadata_difference_files") or []
    ) != SNAPSHOT_METADATA_DIFFERENCE_FILES:
        raise M1ConfigError("replay v2 metadata difference file set mismatch")
    original_files = dict(original.get("file_sha256") or {})
    identical_files = dict(snapshot.get("identical_file_sha256") or {})
    metadata_differences = dict(snapshot.get("metadata_differences") or {})
    if set(metadata_differences) != set(SNAPSHOT_METADATA_DIFFERENCE_FILES):
        raise M1ConfigError("snapshot audit must contain exactly two metadata differences")
    if set(identical_files) != set(original_files) - set(SNAPSHOT_METADATA_DIFFERENCE_FILES):
        raise M1ConfigError("snapshot identical-file set does not cover every non-metadata file")
    if any(identical_files[name] != original_files[name] for name in identical_files):
        raise M1ConfigError("snapshot shared file identity differs from original merge")
    for name in SNAPSHOT_METADATA_DIFFERENCE_FILES:
        row = metadata_differences.get(name) or {}
        if row.get("original_sha256") != original_files.get(name):
            raise M1ConfigError(f"snapshot metadata original hash mismatch: {name}")
        _require_sha(row.get("snapshot_sha256"), f"snapshot {name} SHA-256")
        if row.get("snapshot_sha256") == row.get("original_sha256"):
            raise M1ConfigError(f"snapshot metadata difference is not a difference: {name}")
    expected_authority = CANONICAL_GENERATION_CONTRACT_PATH
    if (
        relation.get("generation_arguments_authority") != expected_authority
        or relation.get("generation_arguments_are_explicit") is not True
        or audit_config.get("generation_arguments_authority") != expected_authority
        or audit_config.get("weights_tokenizer_index_identity")
        != "exact_serialization_bytes"
        or relation.get("weights_tokenizer_index_identity")
        != "exact_serialization_bytes"
    ):
        raise M1ConfigError("snapshot audit does not preserve explicit generation authority")
    return manifest, path, SNAPSHOT_IDENTITY_MANIFEST_SHA256


def validate_replay_spec(config: dict[str, Any]) -> tuple[list[str], list[int]]:
    assert_public_only_config(config)
    workflow = config.get("workflow") or {}
    protocol_version = workflow.get("protocol_version")
    if workflow.get("step") != "replay_equivalence" or protocol_version not in REPLAY_SCHEMAS:
        raise M1ConfigError("replay requires the exact replay workflow schema and step")
    comparison_id = str((config.get("run") or {}).get("comparison_id") or "")
    if protocol_version == REPLAY_SCHEMA:
        if comparison_id != REPLAY_COMPARISON_V1:
            raise M1ConfigError(
                "replay protocol and comparison identity must match bidirectionally"
            )
    elif comparison_id != REPLAY_COMPARISON_V2:
        raise M1ConfigError("replay protocol and comparison identity must match bidirectionally")
    if str((config.get("runtime") or {}).get("transformers_version")) != REPLAY_TRANSFORMERS_VERSION:
        raise M1ConfigError(
            f"replay requires Transformers {REPLAY_TRANSFORMERS_VERSION} exactly"
        )
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
    if str(workflow.get("historical_sampling_config") or "") != CANONICAL_HISTORICAL_CONFIG_PATH:
        raise M1ConfigError("replay requires the canonical frozen historical config path")
    if str(workflow.get("historical_sampling_config_sha256") or "") != CANONICAL_HISTORICAL_CONFIG_SHA256:
        raise M1ConfigError("replay requires the canonical frozen historical config SHA-256")
    historical_path = resolve_repo_path(CANONICAL_HISTORICAL_CONFIG_PATH)
    if not historical_path.is_file():
        raise M1ConfigError("historical sampling config is required to bind fingerprint order")
    if file_sha256(historical_path) != CANONICAL_HISTORICAL_CONFIG_SHA256:
        raise M1ConfigError("canonical frozen historical config SHA-256 mismatch")
    historical = read_structured(historical_path)
    historical_sampling = historical.get("sampling") or {}
    if fingerprints != list(historical_sampling.get("dev_subset_fingerprints") or []):
        raise M1ConfigError("replay fingerprint order differs from the frozen historical config")
    if seeds != list(historical_sampling.get("seeds") or []):
        raise M1ConfigError("replay seed order differs from the frozen historical config")
    if str((config.get("sampling") or {}).get("dev_subset_hash")) != str(
        historical_sampling.get("dev_subset_hash")
    ) or str((config.get("sampling") or {}).get("dev_subset_hash")) != FROZEN_SUBSET_HASH:
        raise M1ConfigError("replay subset identity differs from the frozen historical config")
    _require_sha(workflow.get("fixed_manifest_sha256"), "fixed manifest SHA-256")
    _require_sha((config.get("artifacts") or {}).get("adapter_manifest_sha256"), "adapter manifest SHA-256")
    _require_sha((config.get("artifacts") or {}).get("adapter_sha256"), "adapter SHA-256")
    _require_sha((config.get("artifacts") or {}).get("merged_content_hash"), "merged content hash", length=16)
    if protocol_version == REPLAY_SCHEMA_V2:
        load_snapshot_identity_audit(config)
    else:
        canonical_path = resolve_repo_path(CANONICAL_REPLAY_V1_CONFIG_PATH)
        if (
            not canonical_path.is_file()
            or file_sha256(canonical_path) != CANONICAL_REPLAY_V1_CONFIG_SHA256
            or canonical_hash(config) != CANONICAL_REPLAY_V1_CONFIG_HASH
            or config != read_structured(canonical_path)
        ):
            raise M1ConfigError(
                "replay v1 is restricted to the exact canonical historical config identity"
            )
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
    snapshot_identity: dict[str, Any] | None = None
    if (config.get("workflow") or {}).get("protocol_version") == REPLAY_SCHEMA_V2:
        snapshot_identity, identity_path, identity_sha = load_snapshot_identity_audit(config)
        _verify_file_map(
            merged_dir,
            dict((snapshot_identity.get("original_merge") or {}).get("file_sha256") or {}),
            "original merged",
        )
    merged_content_hash = canonical_directory_hash(merged_dir)
    if merged_content_hash != str(artifacts.get("merged_content_hash")):
        raise M1ConfigError("merged directory content hash mismatch")
    adapter_tokenizer = _tokenizer_file_map(adapter_dir)
    merged_tokenizer = _tokenizer_file_map(merged_dir)
    if not adapter_tokenizer or adapter_tokenizer != merged_tokenizer:
        raise M1ConfigError("adapter and merged tokenizer file identities differ")
    result = {
        "adapter": {
            "path": str(adapter_dir.resolve()),
            "files": observed_adapter_files,
            "weight_serialization_file_map_identity_sha256": _canonical_json_sha256(
                {"adapter_model.safetensors": observed_adapter_files["adapter_model.safetensors"]}
            ),
        },
        "merged": {
            "path": str(merged_dir.resolve()),
            "content_hash": merged_content_hash,
            "weight_files": merged_files,
            "weight_shard_serialization_file_map_identity_sha256": _canonical_json_sha256(
                merged_files
            ),
        },
        "tokenizer": {
            "files": adapter_tokenizer,
            "identity_sha256": _canonical_json_sha256(adapter_tokenizer),
        },
    }
    if snapshot_identity is not None:
        result["submitted_snapshot_audit"] = {
            "manifest_path": str(identity_path),
            "manifest_sha256": identity_sha,
            "original_merge_content_hash": ORIGINAL_MERGE_CONTENT_HASH_V2,
            "submitted_snapshot_content_hash": SUBMITTED_SNAPSHOT_CONTENT_HASH_V2,
            "metadata_differences": snapshot_identity["submitted_snapshot"][
                "metadata_differences"
            ],
            "weights_tokenizer_index_identity": "exact_serialization_bytes",
            "generation_arguments_authority": CANONICAL_GENERATION_CONTRACT_PATH,
        }
    return result


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


def verify_runtime_version(config: dict[str, Any], contract: dict[str, Any]) -> str:
    import transformers

    configured = str((config.get("runtime") or {}).get("transformers_version"))
    contracted = str((contract.get("runtime") or {}).get("transformers_version"))
    installed = str(transformers.__version__)
    if configured != REPLAY_TRANSFORMERS_VERSION or contracted != REPLAY_TRANSFORMERS_VERSION:
        raise M1ConfigError("replay Transformers version does not match the frozen contract")
    if installed != REPLAY_TRANSFORMERS_VERSION:
        raise M1ConfigError(
            "installed Transformers version mismatch: "
            f"expected {REPLAY_TRANSFORMERS_VERSION} but found {installed}"
        )
    return installed


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
    rng_seed = int(generator.initial_seed()) if generator is not None else None
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices, enabled=rng_seed is not None):
        if rng_seed is not None:
            torch.manual_seed(rng_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(rng_seed)
        with torch.inference_mode():
            sequence = model.generate(
                **tensors, pad_token_id=tokenizer.pad_token_id, **args
            )[0]
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
    verify_runtime_version(config, contract)
    workflow = config["workflow"]
    historical_config_path = resolve_repo_path(CANONICAL_HISTORICAL_CONFIG_PATH)
    if file_sha256(historical_config_path) != CANONICAL_HISTORICAL_CONFIG_SHA256:
        raise M1ConfigError("canonical frozen historical config SHA-256 mismatch")
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
    protocol_version = str(workflow.get("protocol_version"))
    result = {
        "artifact_schema": (
            "dftr.adapter_merge_replay_result.v2"
            if protocol_version == REPLAY_SCHEMA_V2
            else "dftr.adapter_merge_replay_result.v1"
        ),
        "protocol_version": protocol_version,
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
