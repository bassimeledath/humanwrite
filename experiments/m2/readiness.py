"""Fail-closed evidence verification performed before an A64 model is loaded."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

try:
    from harness.measurement_v2 import (
        MeasurementV2Error,
        decode_trusted_public_key,
        validate_blind_qualification,
        validate_protocol,
    )
except ImportError:  # repository worker uses the source tree without installing the harness
    from harness.src.harness.measurement_v2 import (
        MeasurementV2Error,
        decode_trusted_public_key,
        validate_blind_qualification,
        validate_protocol,
    )


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReadinessError(ValueError):
    """Raised when the A0-to-A64 evidence chain is incomplete or forged."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if not SHA256_RE.fullmatch(text):
        raise ReadinessError(f"{label} must be a lowercase SHA-256")
    return text


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReadinessError(f"{label} schema mismatch")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load(path: Path, expected_sha: str, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ReadinessError(f"{label} must be an absolute regular file")
    if _sha256(path) != _require_sha(expected_sha, f"{label}.sha256"):
        raise ReadinessError(f"{label} byte hash mismatch")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ReadinessError(f"{label} is not valid unique-key JSON") from error
    if not isinstance(value, dict):
        raise ReadinessError(f"{label} must be a JSON object")
    return value


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved_root, resolved_path = root.resolve(), path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ReadinessError(f"{label} escapes its artifact root") from error
    return resolved_path


def _directory_file_map(root: Path) -> dict[str, str]:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ReadinessError("A0 checkpoint directory must be an absolute regular directory")
    file_map: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ReadinessError("A0 checkpoint directory cannot contain symlinks")
        if item.is_file() and item.name != "checkpoint_manifest.json":
            file_map[item.relative_to(root).as_posix()] = _sha256(item)
    if not file_map:
        raise ReadinessError("A0 checkpoint directory is empty")
    return file_map


def verify_a64_readiness(
    *,
    config: dict[str, Any],
    readiness_path: Path,
    readiness_sha256: str,
    base_model: str,
    base_revision: str,
    generated_tokens_per_rollout: int,
) -> str:
    readiness = _load(readiness_path, readiness_sha256, "A64 readiness manifest")
    _exact(
        readiness,
        {
            "artifact_schema", "status", "comparison_id", "method_contract_sha256",
            "a0_checkpoint_manifest", "a0_generation_manifest",
            "measurement_protocol", "blind_qualification", "trusted_public_keys",
        },
        "A64 readiness manifest",
    )
    method_sha = str((config.get("workflow") or {}).get("method_contract_sha256") or "")
    comparison_id = str((config.get("run") or {}).get("comparison_id") or "")
    if (
        readiness.get("artifact_schema") != "dftr.m2.a64_readiness.v1"
        or readiness.get("status") != "ready"
        or readiness.get("comparison_id") != comparison_id
        or readiness.get("method_contract_sha256") != method_sha
    ):
        raise ReadinessError("A64 readiness does not bind the frozen comparison")

    checkpoint_binding = _exact(
        readiness.get("a0_checkpoint_manifest"),
        {"path", "sha256", "adapter_model_sha256"},
        "A0 checkpoint binding",
    )
    checkpoint_path = Path(str(checkpoint_binding["path"]))
    checkpoint = _load(
        checkpoint_path, str(checkpoint_binding["sha256"]), "A0 checkpoint manifest"
    )
    checkpoint_dir = Path(str(checkpoint.get("checkpoint_dir") or ""))
    if checkpoint_path.resolve() != (checkpoint_dir / "checkpoint_manifest.json").resolve():
        raise ReadinessError("A0 checkpoint manifest is not inside its declared checkpoint")
    observed_map = _directory_file_map(checkpoint_dir)
    if checkpoint.get("file_map_excludes") != ["checkpoint_manifest.json"]:
        raise ReadinessError("A0 checkpoint file-map exclusion contract mismatch")
    if checkpoint.get("file_sha256") != observed_map:
        raise ReadinessError("A0 checkpoint file map does not match actual adapter bytes")
    adapter_sha = _require_sha(
        checkpoint_binding.get("adapter_model_sha256"),
        "A0 checkpoint adapter_model_sha256",
    )
    expected_tokens = (
        int(config["training"]["steps"])
        * int(config["training"]["rollout_batch_size"])
        * generated_tokens_per_rollout
    )
    a0_arm = next((arm for arm in config["arms"] if arm.get("id") == "A0"), None)
    if (
        checkpoint.get("artifact_schema") != "dftr.m2.adapter_native_checkpoint.v1"
        or checkpoint.get("status") != "completed"
        or checkpoint.get("adapter_native") is not True
        or checkpoint.get("arm") != "A0"
        or checkpoint.get("base_model") != base_model
        or checkpoint.get("base_revision") != base_revision
        or checkpoint.get("method_contract_sha256") != method_sha
        or checkpoint.get("source_adapter_manifest_sha256")
        != config["initial_adapter"]["file_manifest_sha256"]
        or checkpoint.get("source_adapter_model_sha256")
        != config["initial_adapter"]["adapter_model_sha256"]
        or checkpoint.get("source_adapter_config_sha256")
        != config["initial_adapter"]["adapter_config_sha256"]
        or checkpoint.get("steps") != int(config["training"]["steps"])
        or checkpoint.get("generated_tokens") != expected_tokens
        or not a0_arm
        or checkpoint.get("mmd_coefficient") != float(a0_arm["mmd_coefficient"])
        or observed_map.get("adapter_model.safetensors") != adapter_sha
    ):
        raise ReadinessError("A0 checkpoint completion or matched exposure binding mismatch")

    protocol_binding = _exact(
        readiness.get("measurement_protocol"),
        {"path", "sha256", "artifact_root"},
        "measurement protocol binding",
    )
    artifact_root = Path(str(protocol_binding["artifact_root"]))
    if not artifact_root.is_absolute() or not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ReadinessError("measurement artifact root must be an absolute regular directory")
    protocol_path = _inside(
        artifact_root, Path(str(protocol_binding["path"])), "measurement protocol"
    )
    protocol = _load(protocol_path, str(protocol_binding["sha256"]), "measurement protocol")

    trust_binding = _exact(
        readiness.get("trusted_public_keys"), {"path", "sha256"}, "trust-store binding"
    )
    trust_path = Path(str(trust_binding["path"]))
    trust_contract = config.get("readiness_trust") or {}
    if (
        trust_path.resolve()
        != Path(str(trust_contract.get("trusted_public_keys_path") or "")).resolve()
        or str(trust_binding.get("sha256"))
        != trust_contract.get("trusted_public_keys_sha256")
    ):
        raise ReadinessError("trust store is not bound by the frozen method contract")
    trusted = _load(trust_path, str(trust_binding["sha256"]), "trusted public keys")
    if len(trusted) < 2 or any(not isinstance(key, str) or not isinstance(value, str) for key, value in trusted.items()):
        raise ReadinessError("trusted public keys must contain distinct operator and blind keys")
    protocol_key = str(trust_contract.get("protocol_signer_key_id") or "")
    blind_key = str(trust_contract.get("blind_signer_key_id") or "")
    try:
        distinct_key_bytes = (
            protocol_key in trusted
            and blind_key in trusted
            and decode_trusted_public_key(trusted[protocol_key])
            != decode_trusted_public_key(trusted[blind_key])
        )
    except MeasurementV2Error as error:
        raise ReadinessError("frozen trust store contains an invalid public key") from error
    if (
        not distinct_key_bytes
        or (protocol.get("operator_signature") or {}).get("key_id") != protocol_key
    ):
        raise ReadinessError("frozen protocol/blind trust identities are invalid")
    try:
        validate_protocol(
            protocol, artifact_root=artifact_root, trusted_public_keys=trusted
        )
    except (MeasurementV2Error, OSError, KeyError, TypeError, ValueError) as error:
        raise ReadinessError(f"signed measurement protocol validation failed: {error}") from error

    generation_binding = _exact(
        readiness.get("a0_generation_manifest"),
        {"path", "sha256", "output_path", "output_sha256"},
        "A0 generation binding",
    )
    generation_path = Path(str(generation_binding["path"]))
    output_path = Path(str(generation_binding["output_path"]))
    baseline_binding = (protocol.get("artifact_bindings") or {}).get("matched_baseline") or {}
    outputs_binding = (protocol.get("artifact_bindings") or {}).get("matched_baseline_outputs") or {}
    expected_generation_path = _inside(
        artifact_root, artifact_root / str(baseline_binding.get("path") or ""),
        "matched baseline",
    )
    expected_output_path = _inside(
        artifact_root, artifact_root / str(outputs_binding.get("path") or ""),
        "matched baseline outputs",
    )
    if generation_path.resolve() != expected_generation_path or output_path.resolve() != expected_output_path:
        raise ReadinessError("A0 generation paths are not the protocol-bound control artifacts")
    generation = _load(
        generation_path, str(generation_binding["sha256"]), "A0 generation manifest"
    )
    output_sha = _require_sha(generation_binding.get("output_sha256"), "A0 output SHA-256")
    if _sha256(output_path) != output_sha:
        raise ReadinessError("A0 generation output byte hash mismatch")
    matched = protocol.get("matched_design") or {}
    hashes = protocol.get("hashes") or {}
    if (
        generation.get("artifact_schema") != "dftr.measurement.matched_sft_baseline.v2"
        or generation.get("status") != "materialized"
        or generation.get("frozen") is not True
        or generation.get("checkpoint_sha256") != adapter_sha
        or generation.get("output_manifest_sha256") != output_sha
        or str(generation_binding.get("sha256")) != str(baseline_binding.get("sha256"))
        or output_sha != str(outputs_binding.get("sha256"))
        or output_sha != matched.get("control_output_manifest_sha256")
        or output_sha != hashes.get("matched_baseline_outputs_sha256")
        or adapter_sha != matched.get("control_checkpoint_sha256")
        or generation.get("documents_per_cell") != protocol.get("design", {}).get("documents_per_cell")
    ):
        raise ReadinessError("A0 generation completion, output, or exposure binding mismatch")

    blind_binding = _exact(
        readiness.get("blind_qualification"),
        {"path", "sha256", "operator", "fixture_pack_path", "fixture_pack_sha256"},
        "blind qualification binding",
    )
    blind = _load(
        Path(str(blind_binding["path"])),
        str(blind_binding["sha256"]),
        "blind qualification",
    )
    fixture_path = Path(str(blind_binding["fixture_pack_path"]))
    fixture_sha = _require_sha(
        blind_binding.get("fixture_pack_sha256"), "blind fixture pack SHA-256"
    )
    if (
        not fixture_path.is_absolute()
        or not fixture_path.is_file()
        or fixture_path.is_symlink()
        or _sha256(fixture_path) != fixture_sha
        or blind.get("fixture_pack_sha256") != fixture_sha
        or (blind.get("operator_signature") or {}).get("key_id") != blind_key
    ):
        raise ReadinessError("blind fixture bytes or frozen signer identity mismatch")
    try:
        validate_blind_qualification(
            protocol=protocol,
            blind_test_manifest=blind,
            operator=str(blind_binding["operator"]),
            artifact_root=artifact_root,
            trusted_public_keys=trusted,
        )
    except (MeasurementV2Error, OSError, KeyError, TypeError, ValueError) as error:
        raise ReadinessError(f"signed blind qualification validation failed: {error}") from error
    return _require_sha(readiness_sha256, "A64 readiness manifest SHA-256")
