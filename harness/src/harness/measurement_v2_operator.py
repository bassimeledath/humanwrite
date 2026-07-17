"""Deterministic operator-side materialization and scoring for measurement v2.

The checked-in measurement-v2 package validates evidence but deliberately does
not manufacture it.  This module turns operator-supplied visible inputs into a
self-contained, signed artifact bundle and invokes those validators.  Missing,
placeholder, underpowered, or unsigned inputs fail closed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Sequence

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .measurement_v2 import (
    REQUIRED_BLIND_GROUPS,
    REQUIRED_HARD_GATE_SCHEMAS,
    build_attestation,
    validate_protocol,
    validate_report_v2,
    verify_historical_inventory,
)
from .metrics.distribution_v2 import (
    EmbeddingPanel,
    MeasurementV2Error,
    bandwidth_hash,
    common_kernel_report,
    human_only_bandwidths,
)
from .metrics.quality_v2 import grouped_authorship_auc
from .metrics.validity_v2 import repetition_noninferiority, same_n_self_bleu


N = 64
PANEL_NAMES = ("human_eval", "human_floor_a", "human_floor_b")
EMBEDDING_SCHEMA = "dftr.measurement.embedding_bundle.v2"
POWER_ASSUMPTIONS_SCHEMA = "dftr.measurement.power_assumptions.v2"
DECISION_SCHEMA = "dftr.measurement.decision_contract.v2"
KEY_SCHEMA = "dftr.measurement.operator_private_key.v1"
WRAPPER_RECEIPT_PUBLIC_KEY_BASE64 = "Gi5VBeV8wETwaDtonwGQ5XdwOHUKKo9o/kLCXnpzGbk="


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise MeasurementV2Error(f"{field} must be a lowercase SHA-256")
    return text


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeasurementV2Error(f"cannot load JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise MeasurementV2Error(f"JSON input must be an object: {path}")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MeasurementV2Error(f"cannot load JSONL input {path}: {error}") from error
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise MeasurementV2Error(
                f"invalid JSONL row {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise MeasurementV2Error(
                f"JSONL row must be an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _sha(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return _sha(path)


def _copy_file(source: str | Path, target: Path) -> str:
    source_path = Path(source)
    if not source_path.is_file() or source_path.is_symlink():
        raise MeasurementV2Error(f"bound input must be a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)
    return _sha(target)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def generate_operator_key(
    private_key_path: str | Path,
    trusted_keys_path: str | Path,
    *,
    key_id: str,
) -> dict[str, str]:
    private_path, trusted_path = Path(private_key_path), Path(trusted_keys_path)
    if not key_id.strip():
        raise MeasurementV2Error("operator key ID is required")
    if private_path.exists():
        raise MeasurementV2Error("refusing to overwrite an operator private key")
    key = Ed25519PrivateKey.generate()
    private_raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        private_path,
        {
            "artifact_schema": KEY_SCHEMA,
            "key_id": key_id,
            "private_key_base64": base64.b64encode(private_raw).decode("ascii"),
        },
    )
    try:
        private_path.chmod(0o600)
    except OSError:
        pass
    trusted = _load_json(trusted_path) if trusted_path.exists() else {}
    if key_id in trusted:
        raise MeasurementV2Error("trusted key ID already exists")
    trusted[key_id] = base64.b64encode(public_raw).decode("ascii")
    trusted_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(trusted_path, trusted)
    return {"key_id": key_id, "trusted_keys_sha256": _sha(trusted_path)}


def _load_private_key(path: str | Path) -> tuple[str, Ed25519PrivateKey, str]:
    value = _load_json(path)
    if value.get("artifact_schema") != KEY_SCHEMA:
        raise MeasurementV2Error("unexpected operator private-key schema")
    key_id = str(value.get("key_id") or "")
    try:
        raw = base64.b64decode(
            str(value.get("private_key_base64") or ""), validate=True
        )
        key = Ed25519PrivateKey.from_private_bytes(raw)
    except (ValueError, TypeError) as error:
        raise MeasurementV2Error("invalid Ed25519 operator private key") from error
    public_raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return key_id, key, base64.b64encode(public_raw).decode("ascii")


def _sign(document: dict[str, Any], private_key_path: str | Path) -> None:
    key_id, key, _ = _load_private_key(private_key_path)
    if "operator_signature" in document:
        raise MeasurementV2Error("refusing to replace an existing operator signature")
    payload = _canonical_bytes(document)
    document["operator_signature"] = {
        "algorithm": "ed25519",
        "key_id": key_id,
        "signed_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signature_base64": base64.b64encode(key.sign(payload)).decode("ascii"),
    }


def _directory_hash(root: Path) -> str:
    if not root.is_dir():
        raise MeasurementV2Error(f"embedder model directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise MeasurementV2Error("embedder model directory is empty")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_embedding_bundle(
    input_jsonl: str | Path,
    output_json: str | Path,
    *,
    model_path: str | Path,
    model_id: str,
    model_revision: str,
    id_field: str,
    text_field: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Embed a visible JSONL bundle using a caller-materialized local model."""
    if not model_id or not model_revision:
        raise MeasurementV2Error("embedder ID and immutable revision are required")
    rows = _load_jsonl(input_jsonl)
    identifiers, texts = [], []
    for row in rows:
        identifier, text = str(row.get(id_field) or ""), row.get(text_field)
        if (
            not identifier
            or identifier in identifiers
            or not isinstance(text, str)
            or not text
        ):
            raise MeasurementV2Error(
                "embedding input requires unique IDs and nonempty text"
            )
        identifiers.append(identifier)
        texts.append(text)
    if not identifiers:
        raise MeasurementV2Error("embedding input is empty")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        vectors = model.encode(
            texts,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as error:
        raise MeasurementV2Error(
            f"local independent embedder failed: {error}"
        ) from error
    array = np.asarray(vectors, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] != len(identifiers)
        or not np.isfinite(array).all()
    ):
        raise MeasurementV2Error("independent embedder returned invalid vectors")
    preprocessing = {
        "text": "utf8_input_verbatim",
        "normalize_embeddings": True,
        "batch_size": int(batch_size),
    }
    bundle = {
        "artifact_schema": EMBEDDING_SCHEMA,
        "status": "materialized",
        "embedder_id": model_id,
        "embedder_revision": model_revision,
        "embedder_sha256": _directory_hash(Path(model_path)),
        "preprocessing": preprocessing,
        "preprocessing_sha256": _canonical_sha(preprocessing),
        "rows": [
            {"document_id": identifier, "embedding": [float(item) for item in vector]}
            for identifier, vector in zip(identifiers, array)
        ],
    }
    _write_json(Path(output_json), bundle)
    return bundle


def _load_embeddings(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    value = _load_json(path)
    if (
        value.get("artifact_schema") != EMBEDDING_SCHEMA
        or value.get("status") != "materialized"
    ):
        raise MeasurementV2Error("embedding bundle is not materialized v2 evidence")
    _require_sha(value.get("embedder_sha256"), "embedder SHA-256")
    preprocessing = value.get("preprocessing")
    if not isinstance(preprocessing, dict) or _canonical_sha(
        preprocessing
    ) != _require_sha(value.get("preprocessing_sha256"), "preprocessing SHA-256"):
        raise MeasurementV2Error("embedding preprocessing contract is not hash-bound")
    result: dict[str, np.ndarray] = {}
    dimension = None
    for row in value.get("rows") or []:
        identifier = str((row or {}).get("document_id") or "")
        vector = np.asarray((row or {}).get("embedding"), dtype=np.float64)
        if (
            not identifier
            or identifier in result
            or vector.ndim != 1
            or vector.size < 1
            or not np.isfinite(vector).all()
        ):
            raise MeasurementV2Error("embedding bundle rows are invalid or duplicated")
        dimension = vector.size if dimension is None else dimension
        if vector.size != dimension:
            raise MeasurementV2Error("embedding bundle dimensions differ")
        result[identifier] = vector
    if not result:
        raise MeasurementV2Error("embedding bundle has no rows")
    return result, value


def _select_human_panels(
    rows: Sequence[dict[str, Any]], *, selection_seed: str
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    if len(rows) < 3 * N:
        raise MeasurementV2Error("visible human source requires at least 192 rows")
    by_id, content_hashes = {}, set()
    eligible_rows = []
    for source_row in rows:
        row = dict(source_row)
        document_id = str(row.get("document_id") or row.get("fingerprint") or "")
        text = row.get("text", row.get("completion"))
        source_shape = (
            bool(row.get("fingerprint"))
            and row.get("fingerprint")
            == hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
            and bool(row.get("source_revision"))
            and row.get("split") in {"train", "dev"}
        )
        basis = str(row.get("eligibility_basis") or "").strip()
        flags = row.get("exclusion_flags")
        if source_shape and not basis and flags is None:
            basis = (
                "fixed-code-visible-source:"
                + str(row.get("source_config") or "unknown")
                + ":"
                + str(row["source_revision"])
            )
            flags = []
            row["eligible"] = True
        if (
            not document_id
            or document_id in by_id
            or not isinstance(text, str)
            or not text.strip()
            or row.get("eligible") is not True
            or not basis
            or not isinstance(flags, list)
            or flags
        ):
            raise MeasurementV2Error(
                "human source rows must carry passing unique eligibility evidence"
            )
        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_sha in content_hashes:
            raise MeasurementV2Error("human source contains duplicate text bytes")
        content_hashes.add(content_sha)
        normalized = {
            **row,
            "document_id": document_id,
            "text": text,
            "eligible": True,
            "eligibility_basis": basis,
            "exclusion_flags": [],
        }
        by_id[document_id] = normalized
        eligible_rows.append(normalized)

    def rank(values: Sequence[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        return sorted(
            values,
            key=lambda row: hashlib.sha256(
                (
                    selection_seed
                    + ":"
                    + label
                    + "\0"
                    + str(row["document_id"])
                    + "\0"
                    + hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
                ).encode("utf-8")
            ).hexdigest(),
        )

    train_rows = [row for row in eligible_rows if row.get("split") == "train"]
    dev_rows = [row for row in eligible_rows if row.get("split") == "dev"]
    if len(train_rows) == 2 * N and len(dev_rows) == N:
        floor_ranked = rank(train_rows, "floor")
        panel_source = {
            "human_eval": sorted(dev_rows, key=lambda row: str(row["document_id"])),
            "human_floor_a": floor_ranked[:N],
            "human_floor_b": floor_ranked[N:],
        }
    else:
        ranked = rank(eligible_rows, "all")[: 3 * N]
        panel_source = {
            name: ranked[offset * N : (offset + 1) * N]
            for offset, name in enumerate(PANEL_NAMES)
        }
    selected_rows = [row for name in PANEL_NAMES for row in panel_source[name]]
    panels: dict[str, list[dict[str, str]]] = {}
    for name in PANEL_NAMES:
        panels[name] = [
            {
                "document_id": str(row["document_id"]),
                "content_sha256": hashlib.sha256(
                    str(row["text"]).encode("utf-8")
                ).hexdigest(),
            }
            for row in panel_source[name]
        ]
    return panels, selected_rows


def _validate_prompt_briefs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != N:
        raise MeasurementV2Error("prompt brief bundle must contain exactly 64 rows")
    seen_ids, seen_references = set(), set()
    result = []
    required_brief_fields = (
        "user_prompt",
        "use_case",
        "style_kind",
        "style",
        "detail_mode",
        "target_length",
        "em_dashes_allowed",
        "outline",
    )
    for source_row in rows:
        row = dict(source_row)
        prompt_id = str(row.get("prompt_id") or row.get("fingerprint") or "")
        reference = row.get("reference_text", row.get("completion"))
        fingerprint = str(
            row.get("reference_fingerprint") or row.get("fingerprint") or ""
        )
        brief = row.get("full_brief")
        if brief is None and all(field in row for field in required_brief_fields):
            if not isinstance(row["outline"], list) or int(row["target_length"]) <= 0:
                raise MeasurementV2Error("synthesized prompt brief fields are invalid")
            brief = "\n".join(
                (
                    f"Writing request: {str(row['user_prompt']).strip()}",
                    f"Use case: {str(row['use_case']).strip()}",
                    f"Style category: {str(row['style_kind']).strip()}",
                    f"Style: {str(row['style']).strip()}",
                    f"Detail mode: {str(row['detail_mode']).strip()}",
                    f"Target length: about {int(row['target_length'])} words",
                    f"Em dashes allowed: {'yes' if bool(row['em_dashes_allowed']) else 'no'}",
                    "Grounding outline (use only these supported facts when non-empty): "
                    + json.dumps(
                        row["outline"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
        if (
            not prompt_id
            or prompt_id in seen_ids
            or not isinstance(brief, str)
            or not brief
            or not isinstance(reference, str)
            or not reference
            or fingerprint != hashlib.sha256(reference.encode("utf-8")).hexdigest()
            or fingerprint in seen_references
            or row.get("split") not in {"quality_visible_human", "dev"}
        ):
            raise MeasurementV2Error(
                "prompt briefs require 64 unique prompts and hash-bound prompt-matched references"
            )
        seen_ids.add(prompt_id)
        seen_references.add(fingerprint)
        result.append(
            {
                **row,
                "prompt_id": prompt_id,
                "full_brief": brief,
                "reference_text": reference,
                "reference_fingerprint": fingerprint,
                "split": "quality_visible_human",
            }
        )
    return sorted(result, key=lambda row: str(row["prompt_id"]))


def _seed_grid(
    rows: Sequence[dict[str, Any]], prompt_ids: Sequence[str]
) -> list[dict[str, Any]]:
    training_seeds = sorted({row.get("training_seed") for row in rows})
    if len(training_seeds) != 1 or not isinstance(training_seeds[0], int):
        raise MeasurementV2Error(
            "the bounded operator pipeline requires exactly one training seed"
        )
    sampling_seeds = sorted({row.get("sampling_seed") for row in rows})
    if len(sampling_seeds) != 1 or not isinstance(sampling_seeds[0], int):
        raise MeasurementV2Error(
            "the bounded operator pipeline requires exactly one sampling seed"
        )
    expected = {
        (prompt_id, training_seeds[0], sampling_seeds[0]) for prompt_id in prompt_ids
    }
    observed = {
        (
            str(row.get("prompt_id") or ""),
            row.get("training_seed"),
            row.get("sampling_seed"),
        )
        for row in rows
    }
    if len(rows) != N or observed != expected:
        raise MeasurementV2Error(
            "output rows do not cover the exact 64-prompt seed grid"
        )
    if any(not isinstance(row.get("text"), str) or not row.get("text") for row in rows):
        raise MeasurementV2Error("output rows require nonempty text")
    return [{"training_seed": training_seeds[0], "sampling_seeds": sampling_seeds}]


def _validate_generation_provenance(
    rows: Sequence[dict[str, Any]],
    *,
    checkpoint_sha256: str,
    generation_contract_sha256: str,
    decoding_policy_sha256: str,
) -> None:
    expected = {
        "checkpoint_sha256": _require_sha(checkpoint_sha256, "generation checkpoint"),
        "generation_contract_sha256": _require_sha(
            generation_contract_sha256, "generation contract"
        ),
        "decoding_policy_sha256": _require_sha(
            decoding_policy_sha256, "decoding policy"
        ),
    }
    if any(
        any(row.get(field) != value for field, value in expected.items())
        for row in rows
    ):
        raise MeasurementV2Error(
            "raw generation rows do not carry the expected checkpoint and contract provenance"
        )


def _validate_generation_run_manifest(
    manifest_path: str | Path,
    outputs_path: str | Path,
    generation_config: str | Path,
    generation_ledger: str | Path,
    wrapper_receipt: str | Path,
    *,
    arm: str,
    checkpoint_sha256: str,
    generation_contract_sha256: str,
    decoding_policy_sha256: str,
) -> tuple[str, str]:
    manifest_file, output_file = Path(manifest_path), Path(outputs_path)
    if (
        not manifest_file.is_file()
        or manifest_file.is_symlink()
        or not output_file.is_file()
        or output_file.is_symlink()
    ):
        raise MeasurementV2Error("generation manifest and output must be regular files")
    manifest = _load_json(manifest_file)
    output_sha = _sha(output_file)
    try:
        import yaml

        parsed_config = yaml.safe_load(Path(generation_config).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise MeasurementV2Error("cannot load the exact generation config") from error
    if not isinstance(parsed_config, dict):
        raise MeasurementV2Error("generation config must be a mapping")
    config_sha = _canonical_sha(parsed_config)
    ledger_rows = _load_jsonl(generation_ledger)
    run_id = str(manifest.get("run_id") or "")
    launch_rows = [
        row
        for row in ledger_rows
        if row.get("kind") == "run" and row.get("run_id") == run_id
    ]
    launch = launch_rows[0] if len(launch_rows) == 1 else {}
    accounting = manifest.get("token_accounting") or {}
    if (
        manifest.get("artifact_schema") != "dftr.m2.adapter_native_generation.v1"
        or manifest.get("status") != "completed"
        or manifest.get("arm") != arm
        or manifest.get("adapter_native") is not True
        or manifest.get("checkpoint_sha256") != checkpoint_sha256
        or manifest.get("generation_contract_sha256") != generation_contract_sha256
        or manifest.get("decoding_policy_sha256") != decoding_policy_sha256
        or manifest.get("documents") != N
        or manifest.get("generated_tokens_per_document") != 64
        or manifest.get("output_sha256") != output_sha
        or accounting != {"total_tokens": N * 64}
        or not re.fullmatch(r"dftr-[0-9]+-[0-9a-f]{8}", run_id)
        or manifest.get("config_sha256") != config_sha
        or launch.get("config_hash") != config_sha
        or launch.get("git_sha") != manifest.get("git_sha")
        or launch.get("comparison") != manifest.get("comparison_id")
        or manifest.get("output_path") != f"/checkpoints/runs/{run_id}/outputs.jsonl"
    ):
        raise MeasurementV2Error(
            "generation run manifest does not authenticate the supplied output bytes"
        )
    receipt_sha = _validate_wrapper_generation_receipt(
        wrapper_receipt,
        manifest_file,
        output_file,
        expected_run_id=run_id,
        expected_config_sha256=config_sha,
        expected_git_sha=str(manifest["git_sha"]),
        expected_comparison=str(manifest["comparison_id"]),
    )
    return _sha(manifest_file), receipt_sha


def _validate_wrapper_generation_receipt(
    receipt_path: str | Path,
    manifest_path: str | Path,
    outputs_path: str | Path,
    *,
    expected_run_id: str,
    expected_config_sha256: str,
    expected_git_sha: str,
    expected_comparison: str,
) -> str:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    receipt_file = Path(receipt_path)
    if not receipt_file.is_file() or receipt_file.is_symlink():
        raise MeasurementV2Error("wrapper generation receipt must be a regular file")
    receipt = _load_json(receipt_file)
    signature = receipt.pop("signature", None)
    exact_keys = {
        "artifact_schema", "status", "key_id", "run_id", "comparison_id",
        "config_sha256", "git_sha", "manifest_path", "manifest_sha256",
        "output_path", "output_sha256",
    }
    canonical = _canonical_bytes(receipt)
    try:
        signature_bytes = base64.b64decode(
            str((signature or {}).get("signature_base64") or ""), validate=True
        )
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(WRAPPER_RECEIPT_PUBLIC_KEY_BASE64, validate=True)
        )
        public_key.verify(signature_bytes, canonical)
    except (InvalidSignature, ValueError, TypeError) as error:
        raise MeasurementV2Error("wrapper generation receipt signature is invalid") from error
    if (
        set(receipt) != exact_keys
        or not isinstance(signature, dict)
        or set(signature) != {"algorithm", "signed_payload_sha256", "signature_base64"}
        or signature.get("algorithm") != "ed25519"
        or signature.get("signed_payload_sha256") != hashlib.sha256(canonical).hexdigest()
        or receipt.get("artifact_schema") != "dftr.wrapper.generation_receipt.v1"
        or receipt.get("status") != "completed"
        or receipt.get("key_id") != "humanwrite-modal-wrapper-receipt-v1"
        or receipt.get("run_id") != expected_run_id
        or receipt.get("config_sha256") != expected_config_sha256
        or receipt.get("git_sha") != expected_git_sha
        or receipt.get("comparison_id") != expected_comparison
        or receipt.get("manifest_path") != f"/checkpoints/runs/{expected_run_id}/run_manifest.json"
        or receipt.get("output_path") != f"/checkpoints/runs/{expected_run_id}/outputs.jsonl"
        or receipt.get("manifest_sha256") != _sha(Path(manifest_path))
        or receipt.get("output_sha256") != _sha(Path(outputs_path))
    ):
        raise MeasurementV2Error("wrapper generation receipt identity or byte binding failed")
    return _sha(receipt_file)


def _simulate_power(
    assumptions: dict[str, Any],
    *,
    n: int,
    seed_grid: list[dict[str, Any]],
    metric_sha: str,
) -> dict[str, Any]:
    if assumptions.get("artifact_schema") != POWER_ASSUMPTIONS_SCHEMA:
        raise MeasurementV2Error("unexpected power-assumptions schema")
    trials = assumptions.get("trials")
    seed = assumptions.get("seed")
    effects = assumptions.get("minimally_important_effects") or {}
    scales = assumptions.get("pilot_scales") or {}
    if (
        not isinstance(trials, int)
        or isinstance(trials, bool)
        or trials < 1000
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise MeasurementV2Error(
            "power simulations require >=1000 trials and a frozen seed"
        )
    for field in ("mmd", "auc", "repetition", "coverage"):
        value = effects.get(field)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise MeasurementV2Error(
                f"power minimally important effect is invalid: {field}"
            )
    mmd_se, auc_se = scales.get("mmd_standard_error"), scales.get("auc_standard_error")
    human_rate, margin = (
        scales.get("human_repetition_rate"),
        scales.get("repetition_margin"),
    )
    if (
        not isinstance(mmd_se, (int, float))
        or mmd_se <= 0
        or not isinstance(auc_se, (int, float))
        or auc_se <= 0
        or not isinstance(human_rate, (int, float))
        or not 0 <= human_rate <= 1
        or not isinstance(margin, (int, float))
        or not 0 < margin <= 1
    ):
        raise MeasurementV2Error("power pilot scales are invalid")
    rng = np.random.default_rng(seed)
    null_z = rng.normal(size=trials)
    mmd_alt = rng.normal(loc=-float(effects["mmd"]), scale=float(mmd_se), size=trials)
    auc_alt = rng.normal(loc=float(effects["auc"]), scale=float(auc_se), size=trials)
    candidate_rate = max(
        0.0, min(1.0, float(human_rate) + float(margin) - float(effects["repetition"]))
    )
    human_successes = rng.binomial(n, float(human_rate), size=trials)
    candidate_successes = rng.binomial(n, candidate_rate, size=trials)
    # Normal approximation is used only for prospective simulation; the scored
    # repetition endpoint still uses the exact frozen Newcombe implementation.
    diff = candidate_successes / n - human_successes / n
    rep_se = np.sqrt(
        candidate_rate * (1 - candidate_rate) / n
        + float(human_rate) * (1 - float(human_rate)) / n
    )
    coverage_draws = rng.normal(size=trials)
    successes = {
        "mmd_type_i": int(np.sum(np.abs(null_z) >= 1.959963984540054)),
        "mmd_power": int(np.sum(mmd_alt <= -1.6448536269514722 * float(mmd_se))),
        "auc_power": int(np.sum(auc_alt >= 1.6448536269514722 * float(auc_se))),
        "repetition_power": int(
            np.sum(diff + 1.6448536269514722 * rep_se <= float(margin))
        ),
        "coverage": int(np.sum(np.abs(coverage_draws) <= 1.959963984540054)),
    }
    generator_common = {
        "algorithm": "numpy-pcg64-parametric-prospective-v1",
        "seed": seed,
        "trials": trials,
        "n": n,
        "pilot_scales": scales,
    }
    simulation_contract = {
        "prospective": True,
        "documents_per_cell": n,
        "prompt_clusters": n,
        "seed_grid": seed_grid,
        "minimally_important_effects": effects,
        "null_generator_sha256": _canonical_sha(
            {**generator_common, "scenario": "null"}
        ),
        "alternative_generator_sha256": _canonical_sha(
            {**generator_common, "scenario": "alternative", "effects": effects}
        ),
        "analysis_code_sha256": metric_sha,
    }
    scenarios = {
        "mmd_type_i": ("null", 0.0),
        "mmd_power": ("alternative", effects["mmd"]),
        "auc_power": ("alternative", effects["auc"]),
        "repetition_power": ("alternative", effects["repetition"]),
        "coverage": ("coverage", effects["coverage"]),
    }
    rows = [
        {
            "endpoint": name,
            "scenario": scenario,
            "effect": effect,
            "trials": trials,
            "successes": successes[name],
        }
        for name, (scenario, effect) in scenarios.items()
    ]
    rates = {name: successes[name] / trials for name in successes}
    passed = (
        rates["mmd_type_i"] <= 0.05
        and rates["mmd_power"] >= 0.8
        and rates["auc_power"] >= 0.8
        and rates["repetition_power"] >= 0.8
        and 0.93 <= rates["coverage"] <= 0.97
    )
    return {
        "simulation_contract": simulation_contract,
        "simulation_results": rows,
        "results": {
            "mmd_type_i_rate": rates["mmd_type_i"],
            "mmd_type_i_max": 0.05,
            "mmd_power": rates["mmd_power"],
            "auc_power": rates["auc_power"],
            "repetition_power": rates["repetition_power"],
            "coverage": rates["coverage"],
        },
        "all_targets_pass": passed,
        "multiplicity": {"method": str(assumptions.get("multiplicity") or "holm")},
    }


def _validate_decision_contract(value: dict[str, Any]) -> dict[str, Any]:
    if (
        value.get("artifact_schema") != DECISION_SCHEMA
        or value.get("status") != "frozen"
    ):
        raise MeasurementV2Error(
            "decision contract must be frozen measurement v2 evidence"
        )
    if value.get("evidence_class") != "prospective_screen":
        raise MeasurementV2Error(
            "operator pipeline accepts only prospective decision contracts"
        )
    if (
        not isinstance(value.get("permutation_draws"), int)
        or value["permutation_draws"] < 10_000
    ):
        raise MeasurementV2Error(
            "decision contract requires at least 10000 permutation draws"
        )
    if (
        not isinstance(value.get("authorship_uncertainty_refits"), int)
        or value["authorship_uncertainty_refits"] < 100
    ):
        raise MeasurementV2Error(
            "decision contract requires at least 100 authorship refits"
        )
    fold_seeds = value.get("authorship_fold_seeds")
    if (
        not isinstance(fold_seeds, list)
        or len(fold_seeds) < 3
        or len(fold_seeds) != len(set(fold_seeds))
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in fold_seeds
        )
    ):
        raise MeasurementV2Error(
            "decision contract requires at least three unique fold seeds"
        )
    seeds = value.get("seeds")
    if (
        not isinstance(seeds, dict)
        or set(seeds) != {"permutation", "bootstrap", "authorship_split"}
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in seeds.values()
        )
    ):
        raise MeasurementV2Error("decision contract must freeze all evaluator seeds")
    thresholds = value.get("thresholds")
    required = {
        "candidate_minus_control_mmd_max",
        "paired_mmd_p_max",
        "repetition_noninferiority_margin",
        "authorship_separability_improvement_min",
        "quality_win_rate_min",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != required:
        raise MeasurementV2Error(
            "decision contract threshold set is incomplete or expanded"
        )
    if any(
        not isinstance(thresholds[field], (int, float))
        or isinstance(thresholds[field], bool)
        or not math.isfinite(float(thresholds[field]))
        for field in required
    ):
        raise MeasurementV2Error("decision thresholds must be finite numbers")
    if not 0 < float(thresholds["paired_mmd_p_max"]) <= 0.05:
        raise MeasurementV2Error("paired MMD alpha must be in (0, 0.05]")
    if not 0 <= float(thresholds["repetition_noninferiority_margin"]) <= 1:
        raise MeasurementV2Error("repetition margin must be in [0,1]")
    if not 0 <= float(thresholds["quality_win_rate_min"]) <= 1:
        raise MeasurementV2Error("quality threshold must be in [0,1]")
    return value


def freeze_operator_bundle(
    *,
    artifact_root: str | Path,
    human_source: str | Path,
    prompt_briefs: str | Path,
    control_outputs: str | Path,
    control_generation_manifest: str | Path,
    control_generation_config: str | Path,
    generation_ledger: str | Path,
    control_wrapper_receipt: str | Path,
    human_embeddings: str | Path,
    power_assumptions: str | Path,
    decision_contract: str | Path,
    dependency_lock: str | Path,
    metric_code: str | Path,
    private_key: str | Path,
    trusted_keys: str | Path,
    historical_inventory: str | Path,
    repo_root: str | Path,
    control_checkpoint_sha256: str,
    decoding_policy_sha256: str,
    generation_contract_sha256: str,
    operator: str,
    reviewed_at: str,
    selection_seed: str = "dftr-measurement-v2-panels-v1",
) -> dict[str, Any]:
    root = Path(artifact_root)
    if root.exists() and any(root.iterdir()):
        raise MeasurementV2Error("artifact root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    control_checkpoint_sha256 = _require_sha(
        control_checkpoint_sha256, "control checkpoint"
    )
    decoding_policy_sha256 = _require_sha(decoding_policy_sha256, "decoding policy")
    generation_contract_sha256 = _require_sha(
        generation_contract_sha256, "generation contract"
    )
    decision = _validate_decision_contract(_load_json(decision_contract))

    panel_rows, selected_humans = _select_human_panels(
        _load_jsonl(human_source), selection_seed=selection_seed
    )
    prompts = _validate_prompt_briefs(_load_jsonl(prompt_briefs))
    prompt_ids = [str(row["prompt_id"]) for row in prompts]
    if set(prompt_ids) != {row["document_id"] for row in panel_rows["human_eval"]}:
        raise MeasurementV2Error(
            "prompt brief IDs must exactly equal the frozen human-eval panel IDs"
        )
    raw_control = _load_jsonl(control_outputs)
    seed_grid = _seed_grid(raw_control, prompt_ids)
    _validate_generation_provenance(
        raw_control,
        checkpoint_sha256=control_checkpoint_sha256,
        generation_contract_sha256=generation_contract_sha256,
        decoding_policy_sha256=decoding_policy_sha256,
    )
    control_generation_manifest_sha, control_wrapper_receipt_sha = _validate_generation_run_manifest(
        control_generation_manifest,
        control_outputs,
        control_generation_config,
        generation_ledger,
        control_wrapper_receipt,
        arm="A0",
        checkpoint_sha256=control_checkpoint_sha256,
        generation_contract_sha256=generation_contract_sha256,
        decoding_policy_sha256=decoding_policy_sha256,
    )
    selected_seed = seed_grid[0]["training_seed"]

    human_vectors, embedding_meta = _load_embeddings(human_embeddings)
    selected_ids = {str(row["document_id"]) for row in selected_humans}
    if set(human_vectors) != selected_ids:
        raise MeasurementV2Error(
            "human embedding bundle must contain exactly the selected 192 IDs"
        )
    embeddings_target = root / "inputs/human_embeddings.json"
    human_embedding_sha = _copy_file(human_embeddings, embeddings_target)

    human_contents_path = root / "human_panel_contents.jsonl"
    human_contents_sha = _write_jsonl(human_contents_path, selected_humans)
    eligibility_rows = [
        {
            "document_id": str(row["document_id"]),
            "eligible": True,
            "eligibility_basis": str(row["eligibility_basis"]),
            "exclusion_flags": [],
        }
        for row in selected_humans
    ]
    human_manifest = {
        "artifact_schema": "dftr.measurement.human_panels.v2",
        "status": "materialized",
        "frozen": True,
        "target_n_per_panel": N,
        "required_pool_size": 3 * N,
        "selection_seed": selection_seed,
        "sampling": "disjoint_without_replacement",
        "content_bundle_sha256": human_contents_sha,
        "eligibility_attestation_sha256": _canonical_sha(eligibility_rows),
        "panels": panel_rows,
    }
    human_manifest_path = root / "human_panels.json"
    human_manifest_sha = _write_json(human_manifest_path, human_manifest)

    prompt_brief_path = root / "prompt_briefs.jsonl"
    prompt_brief_sha = _write_jsonl(prompt_brief_path, prompts)
    prompt_panel = {
        "artifact_schema": "dftr.measurement.prompt_panel.v2",
        "status": "frozen",
        "frozen": True,
        "prompt_ids": prompt_ids,
        "full_brief_sha256": prompt_brief_sha,
        "quality_reference_split": "quality_visible_human",
    }
    prompt_panel_path = root / "prompt_panel.json"
    prompt_panel_sha = _write_json(prompt_panel_path, prompt_panel)

    panel_vectors = {
        name: EmbeddingPanel.build(
            name,
            [row["document_id"] for row in panel_rows[name]],
            [human_vectors[row["document_id"]] for row in panel_rows[name]],
        )
        for name in PANEL_NAMES
    }
    bandwidth_values = list(
        human_only_bandwidths(
            panel_vectors["human_floor_a"], panel_vectors["human_floor_b"]
        )
    )
    protocol_panels = {
        name: {
            "status": "materialized",
            "frozen": True,
            "document_count": N,
            "document_ids": [row["document_id"] for row in panel_rows[name]],
            "content_manifest_sha256": _canonical_sha(panel_rows[name]),
        }
        for name in PANEL_NAMES
    }
    bandwidths = {
        "artifact_schema": "dftr.measurement.bandwidths.v2",
        "status": "frozen",
        "frozen": True,
        "source": "human_floor_a_union_human_floor_b_only",
        "values": bandwidth_values,
        "panel_manifest_sha256": human_manifest_sha,
        "floor_a_content_manifest_sha256": protocol_panels["human_floor_a"][
            "content_manifest_sha256"
        ],
        "floor_b_content_manifest_sha256": protocol_panels["human_floor_b"][
            "content_manifest_sha256"
        ],
        "embedder_id": embedding_meta.get("embedder_id"),
        "embedder_revision": embedding_meta.get("embedder_revision"),
        "embedder_sha256": embedding_meta["embedder_sha256"],
        "preprocessing_sha256": embedding_meta["preprocessing_sha256"],
        "embedding_bundle_sha256": human_embedding_sha,
        "bandwidth_sha256": bandwidth_hash(bandwidth_values),
    }
    bandwidth_path = root / "bandwidths.json"
    bandwidth_file_sha = _write_json(bandwidth_path, bandwidths)

    dependency_path, metric_path = (
        root / "inputs/uv.lock",
        root / "inputs/measurement_v2.py",
    )
    dependency_sha = _copy_file(dependency_lock, dependency_path)
    metric_sha = _copy_file(metric_code, metric_path)
    decision_path = root / "inputs/decision_contract.json"
    decision_sha = _copy_file(decision_contract, decision_path)
    assumptions_path = root / "inputs/power_assumptions.json"
    assumptions_sha = _copy_file(power_assumptions, assumptions_path)

    sampling_grid_sha = _canonical_sha(seed_grid)
    decorated_control = [
        {
            "prompt_id": str(row["prompt_id"]),
            "training_seed": row["training_seed"],
            "sampling_seed": row["sampling_seed"],
            "text": str(row["text"]),
            "full_brief_sha256": prompt_brief_sha,
            "prompt_panel_sha256": prompt_panel_sha,
            "sampling_grid_sha256": sampling_grid_sha,
            "checkpoint_sha256": control_checkpoint_sha256,
            "decoding_policy_sha256": decoding_policy_sha256,
            "generation_contract_sha256": generation_contract_sha256,
        }
        for row in sorted(raw_control, key=lambda row: str(row["prompt_id"]))
    ]
    control_output_path = root / "matched_control_outputs.jsonl"
    control_output_sha = _write_jsonl(control_output_path, decorated_control)
    baseline = {
        "artifact_schema": "dftr.measurement.matched_sft_baseline.v2",
        "status": "materialized",
        "frozen": True,
        "documents_per_cell": N,
        "prompt_panel_sha256": prompt_panel_sha,
        "full_brief_sha256": prompt_brief_sha,
        "sampling_grid_sha256": sampling_grid_sha,
        "seed_grid": seed_grid,
        "output_manifest_sha256": control_output_sha,
        "checkpoint_sha256": control_checkpoint_sha256,
        "decoding_policy_sha256": decoding_policy_sha256,
        "generation_contract_sha256": generation_contract_sha256,
        "source_generation_manifest_sha256": control_generation_manifest_sha,
        "source_wrapper_receipt_sha256": control_wrapper_receipt_sha,
    }
    baseline_path = root / "matched_baseline.json"
    baseline_sha = _write_json(baseline_path, baseline)
    calibration = {
        "artifact_schema": "dftr.measurement.calibration.v2",
        "status": "frozen",
        "frozen": True,
        "documents_per_cell": N,
        "human_documents_per_panel": N,
        "source": "matched_current_control_and_disjoint_visible_humans_only",
        "legacy_baseline_stats_admissible": False,
        "raw_paired_effects_required": True,
        "human_panels_sha256": human_manifest_sha,
        "bandwidths_sha256": bandwidth_file_sha,
        "matched_baseline_sha256": baseline_sha,
        "dependency_lock_sha256": dependency_sha,
        "decision_contract_sha256": decision_sha,
    }
    calibration_path = root / "calibration.json"
    calibration_sha = _write_json(calibration_path, calibration)

    power_values = _simulate_power(
        _load_json(power_assumptions), n=N, seed_grid=seed_grid, metric_sha=metric_sha
    )
    power = {
        "artifact_schema": "dftr.measurement.power_plan.v2",
        "status": "frozen" if power_values["all_targets_pass"] else "failed",
        "frozen": bool(power_values["all_targets_pass"]),
        "all_targets_pass": bool(power_values["all_targets_pass"]),
        "documents_per_cell": N,
        "human_panels_sha256": human_manifest_sha,
        "bandwidths_sha256": bandwidth_file_sha,
        "calibration_sha256": calibration_sha,
        "matched_baseline_sha256": baseline_sha,
        "dependency_lock_sha256": dependency_sha,
        "results": power_values["results"],
        "multiplicity": power_values["multiplicity"],
        "simulation_contract": power_values["simulation_contract"],
        "trial_manifest_sha256": _canonical_sha(power_values["simulation_contract"]),
        "simulation_results": power_values["simulation_results"],
        "simulation_results_sha256": _canonical_sha(power_values["simulation_results"]),
        "assumptions_sha256": assumptions_sha,
    }
    power_path = root / "power_plan.json"
    power_sha = _write_json(power_path, power)
    if not power_values["all_targets_pass"]:
        _write_json(
            root / "materialization_status.json",
            {
                "status": "fail_closed",
                "reason": "prospective_power_failed",
                "results": power["results"],
            },
        )
        raise MeasurementV2Error("prospective power simulations did not qualify n=64")

    selection = {
        "artifact_schema": "dftr.measurement.selection_policy.v2",
        "status": "frozen",
        "frozen": True,
        "selection": {"rule_type": "fixed_seed", "seed": selected_seed},
        "declared_before_measurement": True,
    }
    selection_path = root / "selection_policy.json"
    selection_sha = _write_json(selection_path, selection)
    hashes = {
        "dependency_lock_sha256": dependency_sha,
        "metric_code_sha256": metric_sha,
        "prompt_panel_sha256": prompt_panel_sha,
        "prompt_brief_sha256": prompt_brief_sha,
        "human_panels_sha256": human_manifest_sha,
        "human_panel_contents_sha256": human_contents_sha,
        "bandwidths_sha256": bandwidth_file_sha,
        "power_plan_sha256": power_sha,
        "calibration_sha256": calibration_sha,
        "matched_baseline_sha256": baseline_sha,
        "matched_baseline_outputs_sha256": control_output_sha,
        "selection_policy_sha256": selection_sha,
        "decision_contract_sha256": decision_sha,
        "human_embedding_bundle_sha256": human_embedding_sha,
    }
    paths = {
        "dependency_lock": dependency_path,
        "metric_code": metric_path,
        "prompt_panel": prompt_panel_path,
        "prompt_brief": prompt_brief_path,
        "human_panels": human_manifest_path,
        "human_panel_contents": human_contents_path,
        "bandwidths": bandwidth_path,
        "power_plan": power_path,
        "calibration": calibration_path,
        "matched_baseline": baseline_path,
        "matched_baseline_outputs": control_output_path,
        "selection_policy": selection_path,
    }
    artifact_bindings = {
        name: {"path": _relative(root, path), "sha256": _sha(path)}
        for name, path in paths.items()
    }
    protocol = {
        "artifact_schema": "dftr.measurement.protocol.v2",
        "status": "ready",
        "frozen": True,
        "design": {
            "documents_per_cell": N,
            "human_pool_documents": 3 * N,
            "replacement_sampling": False,
            "effective_n_unit": "unique_prompt_cluster",
        },
        "hashes": hashes,
        "panels": protocol_panels,
        "bandwidth_contract": {
            "status": "frozen",
            "source": "human_floor_a_union_human_floor_b",
            "values": bandwidth_values,
            "bandwidth_value_sha256": bandwidth_hash(bandwidth_values),
        },
        "power": {
            "mmd_type_i": "pass",
            "mmd_power": "pass",
            "auc_power": "pass",
            "repetition_power": "pass",
            "coverage": "pass",
        },
        "seeds": dict(decision.get("seeds") or {}),
        "selection_policy": {"selection": selection["selection"]},
        "required_hard_gates": REQUIRED_HARD_GATE_SCHEMAS,
        "matched_design": {
            "candidate_full_brief_sha256": prompt_brief_sha,
            "control_full_brief_sha256": prompt_brief_sha,
            "sampling_grid_sha256": sampling_grid_sha,
            "seed_grid": seed_grid,
            "control_output_manifest_sha256": control_output_sha,
            "control_checkpoint_sha256": control_checkpoint_sha256,
            "decoding_policy_sha256": decoding_policy_sha256,
            "generation_contract_sha256": generation_contract_sha256,
        },
        "decision_contract": decision,
        "artifact_bindings": artifact_bindings,
        "operator_approval": {
            "approved": True,
            "reviewer": operator,
            "reviewed_at": reviewed_at,
        },
    }
    required_seeds = {"permutation", "bootstrap", "authorship_split"}
    if set(protocol["seeds"]) != required_seeds or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in protocol["seeds"].values()
    ):
        raise MeasurementV2Error(
            "decision contract must freeze three nonnegative evaluator seeds"
        )
    _sign(protocol, private_key)
    protocol_path = root / "measurement_protocol_v2.json"
    _write_json(protocol_path, protocol)

    key_id, _, public_base64 = _load_private_key(private_key)
    trusted = _load_json(trusted_keys)
    if trusted.get(key_id) != public_base64:
        raise MeasurementV2Error(
            "operator private key is absent from the supplied trust store"
        )
    trusted_path = root / "trusted_operator_keys_v2.json"
    _write_json(trusted_path, trusted)
    validate_protocol(protocol, artifact_root=root, trusted_public_keys=trusted)

    inventory = _load_json(historical_inventory)
    inventory_check = verify_historical_inventory(inventory, repo_root=repo_root)
    if inventory_check.get("status") != "pass":
        raise MeasurementV2Error(
            "historical v1 inventory changed during materialization"
        )
    _sign(inventory_check, private_key)
    _write_json(root / "historical_inventory_check.json", inventory_check)
    blind_candidate = {
        "artifact_schema": "dftr.measurement.blind_test_manifest.v2",
        "status": "not_run",
        "protocol_sha256": _canonical_sha(protocol),
        "evaluator_commit": metric_sha,
        "dependency_lock_sha256": dependency_sha,
        "fixture_pack_sha256": None,
        "runtime_versions": {},
        "tested_at": None,
        "signer_identity": None,
        "tests": [
            {"name": name, "status": "not_run"}
            for name in sorted(REQUIRED_BLIND_GROUPS)
        ],
        "no_sealed_imitation": False,
        "operator_signature": None,
    }
    _write_json(root / "blind_test_manifest_v2.candidate.json", blind_candidate)
    status = {
        "artifact_schema": "dftr.measurement.operator_materialization.v2",
        "status": "protocol_ready_blind_attestation_pending",
        "protocol_sha256": _canonical_sha(protocol),
        "real_inputs": {
            "human_source_rows_selected": 3 * N,
            "prompt_briefs": N,
            "control_outputs": N,
            "power_trials_per_scenario": (_load_json(power_assumptions))["trials"],
        },
        "remaining": [
            "independent signed 13-group blind manifest",
            "candidate outputs and same-embedder candidate/control embeddings",
            "prompt-level quality results for promotion",
            "four observed hard-gate decisions for promotion",
        ],
    }
    _write_json(root / "materialization_status.json", status)
    return status


def _load_bundle_protocol(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    protocol = _load_json(root / "measurement_protocol_v2.json")
    trusted = _load_json(root / "trusted_operator_keys_v2.json")
    validate_protocol(protocol, artifact_root=root, trusted_public_keys=trusted)
    return protocol, {str(key): str(value) for key, value in trusted.items()}


def _quality_from_results(
    path: str | Path | None, prompt_ids: Sequence[str]
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_measured",
            "reason": "prompt-level quality results absent",
        }
    rows = _load_jsonl(path)
    values = {}
    for row in rows:
        prompt_id, decision = (
            str(row.get("prompt_id") or ""),
            str(row.get("winner") or "").casefold(),
        )
        if prompt_id in values or decision not in {"candidate", "human", "tie"}:
            raise MeasurementV2Error(
                "quality results require unique candidate/human/tie rows"
            )
        values[prompt_id] = decision
    if set(values) != set(prompt_ids):
        raise MeasurementV2Error("quality results do not cover the exact prompt panel")
    scores = [
        1.0 if values[item] == "candidate" else 0.5 if values[item] == "tie" else 0.0
        for item in prompt_ids
    ]
    return {
        "status": "measured",
        "win_rate": float(np.mean(scores)),
        "jmq": float(2 * np.mean(scores)),
        "wins": sum(values[item] == "candidate" for item in prompt_ids),
        "losses": sum(values[item] == "human" for item in prompt_ids),
        "ties": sum(values[item] == "tie" for item in prompt_ids),
    }


def _load_hard_gate_sources(path: str | Path | None) -> dict[str, Path]:
    """Load operator-supplied exact gate evidence without manufacturing it."""
    if path is None:
        return {}
    source_map = _load_json(path)
    if set(source_map) != set(REQUIRED_HARD_GATE_SCHEMAS):
        raise MeasurementV2Error(
            "hard-gate manifest must map every frozen gate to an evidence file"
        )
    result: dict[str, Path] = {}
    for name, schema in REQUIRED_HARD_GATE_SCHEMAS.items():
        source = source_map.get(name)
        if not isinstance(source, str) or not source:
            raise MeasurementV2Error(f"hard-gate evidence path is missing: {name}")
        evidence_path = Path(source)
        if not evidence_path.is_file() or evidence_path.is_symlink():
            raise MeasurementV2Error(f"hard-gate evidence path is invalid: {name}")
        evidence = _load_json(evidence_path)
        if set(evidence) != {"artifact_schema", "name", "decision"} or evidence != {
            "artifact_schema": schema,
            "name": name,
            "decision": "pass",
        }:
            raise MeasurementV2Error(f"hard-gate evidence is not an exact pass: {name}")
        if evidence_path.resolve() in {item.resolve() for item in result.values()}:
            raise MeasurementV2Error("hard-gate evidence file is reused across gates")
        result[name] = evidence_path
    if len({_sha(source) for source in result.values()}) != len(result):
        raise MeasurementV2Error("hard-gate evidence bytes are reused across gates")
    return result


def score_candidate_bundle(
    *,
    artifact_root: str | Path,
    candidate_outputs: str | Path,
    candidate_generation_manifest: str | Path,
    candidate_generation_config: str | Path,
    generation_ledger: str | Path,
    candidate_wrapper_receipt: str | Path,
    score_embeddings: str | Path,
    candidate_checkpoint_sha256: str,
    private_key: str | Path,
    quality_results: str | Path | None = None,
    hard_gate_results: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    protocol, trusted = _load_bundle_protocol(root)
    prompt_rows = _load_jsonl(root / "prompt_briefs.jsonl")
    prompt_ids = [str(row["prompt_id"]) for row in prompt_rows]
    raw_candidate = _load_jsonl(candidate_outputs)
    candidate_checkpoint_sha256 = _require_sha(
        candidate_checkpoint_sha256, "candidate checkpoint"
    )
    matched = protocol["matched_design"]
    _validate_generation_provenance(
        raw_candidate,
        checkpoint_sha256=candidate_checkpoint_sha256,
        generation_contract_sha256=matched["generation_contract_sha256"],
        decoding_policy_sha256=matched["decoding_policy_sha256"],
    )
    _validate_generation_run_manifest(
        candidate_generation_manifest,
        candidate_outputs,
        candidate_generation_config,
        generation_ledger,
        candidate_wrapper_receipt,
        arm="A64",
        checkpoint_sha256=candidate_checkpoint_sha256,
        generation_contract_sha256=matched["generation_contract_sha256"],
        decoding_policy_sha256=matched["decoding_policy_sha256"],
    )
    candidate_seed_grid = _seed_grid(raw_candidate, prompt_ids)
    if candidate_seed_grid != matched["seed_grid"]:
        raise MeasurementV2Error(
            "candidate outputs do not use the frozen matched seed grid"
        )
    candidate_rows = [
        {
            "prompt_id": str(row["prompt_id"]),
            "training_seed": row["training_seed"],
            "sampling_seed": row["sampling_seed"],
            "text": str(row["text"]),
            "full_brief_sha256": matched["candidate_full_brief_sha256"],
            "prompt_panel_sha256": protocol["hashes"]["prompt_panel_sha256"],
            "sampling_grid_sha256": matched["sampling_grid_sha256"],
            "checkpoint_sha256": candidate_checkpoint_sha256,
            "decoding_policy_sha256": matched["decoding_policy_sha256"],
            "generation_contract_sha256": matched["generation_contract_sha256"],
        }
        for row in sorted(raw_candidate, key=lambda row: str(row["prompt_id"]))
    ]
    candidate_path = root / "candidate_outputs.jsonl"
    candidate_sha = _write_jsonl(candidate_path, candidate_rows)
    control_rows = _load_jsonl(root / "matched_control_outputs.jsonl")
    control_by_id = {str(row["prompt_id"]): row for row in control_rows}
    candidate_by_id = {str(row["prompt_id"]): row for row in candidate_rows}

    score_vectors, score_meta = _load_embeddings(score_embeddings)
    human_vectors, human_meta = _load_embeddings(root / "inputs/human_embeddings.json")
    if any(
        score_meta.get(field) != human_meta.get(field)
        for field in (
            "embedder_id",
            "embedder_revision",
            "embedder_sha256",
            "preprocessing_sha256",
        )
    ):
        raise MeasurementV2Error(
            "candidate/control embeddings do not use the frozen dev embedder"
        )
    expected_score_ids = {f"candidate:{item}" for item in prompt_ids} | {
        f"control:{item}" for item in prompt_ids
    }
    if set(score_vectors) != expected_score_ids:
        raise MeasurementV2Error(
            "score embedding bundle must cover exact candidate/control prompt IDs"
        )
    human_manifest = _load_json(root / "human_panels.json")
    panels = human_manifest["panels"]
    embedding_panels = {
        name: EmbeddingPanel.build(
            name,
            [row["document_id"] for row in panels[name]],
            [human_vectors[row["document_id"]] for row in panels[name]],
        )
        for name in PANEL_NAMES
    }
    candidate_panel = EmbeddingPanel.build(
        "candidate",
        prompt_ids,
        [score_vectors[f"candidate:{item}"] for item in prompt_ids],
    )
    control_panel = EmbeddingPanel.build(
        "control", prompt_ids, [score_vectors[f"control:{item}"] for item in prompt_ids]
    )
    decision = protocol["decision_contract"]
    draws = int(decision.get("permutation_draws", 10_000))
    distribution = common_kernel_report(
        candidate_panel,
        control_panel,
        embedding_panels["human_eval"],
        embedding_panels["human_floor_a"],
        embedding_panels["human_floor_b"],
        _load_json(root / "bandwidths.json")["values"],
        permutation_draws=draws,
        seed=int(protocol["seeds"]["permutation"]),
    )
    thresholds = decision.get("thresholds") or {}
    distribution_pass = distribution["candidate_minus_control"] <= float(
        thresholds.get("candidate_minus_control_mmd_max", 0.0)
    ) and distribution["paired_candidate_control_p"] <= float(
        thresholds.get("paired_mmd_p_max", 0.05)
    )
    distribution.update(
        {
            "bandwidth_value_sha256": distribution["bandwidth_sha256"],
            "bandwidth_sha256": protocol["hashes"]["bandwidths_sha256"],
            "permutation_seed": int(protocol["seeds"]["permutation"]),
            "power_plan_passed": True,
            "decision": "pass" if distribution_pass else "fail",
        }
    )
    human_eval_text = {
        str(row["document_id"]): str(row["text"])
        for row in _load_jsonl(root / "human_panel_contents.jsonl")
    }
    human_eval_ids = protocol["panels"]["human_eval"]["document_ids"]
    candidate_texts = [str(candidate_by_id[item]["text"]) for item in prompt_ids]
    control_texts = [str(control_by_id[item]["text"]) for item in prompt_ids]
    human_texts = [human_eval_text[item] for item in human_eval_ids]
    repetition = repetition_noninferiority(
        candidate_texts,
        human_texts,
        margin=float(thresholds.get("repetition_noninferiority_margin", 0.05)),
        power_plan_passed=True,
        minimum_n=N,
    )
    repetition["same_n_self_bleu"] = same_n_self_bleu(candidate_texts, human_texts)
    uncertainty_refits = int(decision.get("authorship_uncertainty_refits", 100))
    fold_seeds = tuple(
        int(value) for value in decision.get("authorship_fold_seeds", [701, 702, 703])
    )
    human_authorship = [
        {"text": text, "cluster_id": f"human:{document_id}"}
        for text, document_id in zip(human_texts, human_eval_ids)
    ]
    candidate_authorship = [
        {"text": text, "cluster_id": f"candidate:{prompt_id}"}
        for text, prompt_id in zip(candidate_texts, prompt_ids)
    ]
    control_authorship = [
        {"text": text, "cluster_id": f"control:{prompt_id}"}
        for text, prompt_id in zip(control_texts, prompt_ids)
    ]
    candidate_auc = grouped_authorship_auc(
        candidate_authorship,
        human_authorship,
        fold_seeds=fold_seeds,
        uncertainty_refits=uncertainty_refits,
        min_effective_clusters=N,
        seed=int(protocol["seeds"]["authorship_split"]),
    )
    control_auc = grouped_authorship_auc(
        control_authorship,
        human_authorship,
        fold_seeds=fold_seeds,
        uncertainty_refits=uncertainty_refits,
        min_effective_clusters=N,
        seed=int(protocol["seeds"]["authorship_split"]) + 1,
    )
    authorship_pass = candidate_auc["separability"] <= (
        control_auc["separability"]
        - float(thresholds.get("authorship_separability_improvement_min", 0.0))
    )
    authorship = {
        **candidate_auc,
        "candidate": candidate_auc,
        "control": control_auc,
        "grouped": True,
        "decision": "pass" if authorship_pass else "fail",
        "fit_count": int(candidate_auc["fit_count"]) + int(control_auc["fit_count"]),
    }
    quality = _quality_from_results(quality_results, prompt_ids)
    if quality["status"] == "measured":
        quality["decision"] = (
            "pass"
            if quality["win_rate"] >= float(thresholds.get("quality_win_rate_min", 0.5))
            else "fail"
        )

    gate_sources = _load_hard_gate_sources(hard_gate_results)
    exact_gate_input = bool(gate_sources)
    promotion = bool(
        distribution_pass
        and quality.get("decision") == "pass"
        and repetition.get("decision") == "pass"
        and authorship_pass
        and exact_gate_input
    )
    hashes = protocol["hashes"]
    report_hashes = {
        "protocol_sha256": _canonical_sha(protocol),
        "prompt_panel_sha256": hashes["prompt_panel_sha256"],
        "human_eval_sha256": protocol["panels"]["human_eval"][
            "content_manifest_sha256"
        ],
        "human_floor_a_sha256": protocol["panels"]["human_floor_a"][
            "content_manifest_sha256"
        ],
        "human_floor_b_sha256": protocol["panels"]["human_floor_b"][
            "content_manifest_sha256"
        ],
        "bandwidths_sha256": hashes["bandwidths_sha256"],
        "power_plan_sha256": hashes["power_plan_sha256"],
        "calibration_sha256": hashes["calibration_sha256"],
        "matched_baseline_sha256": hashes["matched_baseline_sha256"],
        "selection_policy_sha256": hashes["selection_policy_sha256"],
        "candidate_full_brief_sha256": matched["candidate_full_brief_sha256"],
        "control_full_brief_sha256": matched["control_full_brief_sha256"],
        "sampling_grid_sha256": matched["sampling_grid_sha256"],
        "control_output_manifest_sha256": matched["control_output_manifest_sha256"],
        "dependency_lock_sha256": hashes["dependency_lock_sha256"],
        "evaluator_commit_sha256": hashes["metric_code_sha256"],
        "candidate_output_manifest_sha256": candidate_sha,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "score_embedding_bundle_sha256": _sha(Path(score_embeddings)),
    }
    if quality_results is not None:
        report_hashes["quality_results_sha256"] = _sha(Path(quality_results))
    if hard_gate_results is not None:
        report_hashes["hard_gate_source_sha256"] = _sha(Path(hard_gate_results))
    report = {
        "artifact_schema": "dftr.measurement.report.v2",
        "evidence_class": "prospective_screen",
        "counts": {
            "documents_per_cell": N,
            "human_documents_per_panel": N,
            "effective_prompt_clusters": N,
        },
        "hashes": report_hashes,
        "seeds": {
            "training": [matched["seed_grid"][0]["training_seed"]],
            "sampling": matched["seed_grid"][0]["sampling_seeds"],
            "cells": matched["seed_grid"],
        },
        "checkpoint_manifest": {"selection": protocol["selection_policy"]["selection"]},
        "distribution": distribution,
        "quality": quality,
        "repetition": repetition,
        "authorship": authorship,
        "promotion": {"eligible": promotion},
    }
    if quality["status"] == "measured":
        report["quality_linkage"] = {"status": "verified", "matched_pairs": N}
    if promotion:
        report["candidate_output_binding"] = {
            "path": candidate_path.name,
            "sha256": candidate_sha,
        }
        report["hard_gates"] = {}
        for name, schema in REQUIRED_HARD_GATE_SCHEMAS.items():
            evidence_path = root / f"gate-{name}.json"
            evidence_sha = _copy_file(gate_sources[name], evidence_path)
            report["hard_gates"][name] = {
                "version": schema,
                "decision": "pass",
                "evidence_path": evidence_path.name,
                "evidence_sha256": evidence_sha,
            }
    _sign(report, private_key)
    report_path = root / "measurement_report_v2.json"
    _write_json(report_path, report)
    validate_report_v2(
        report, protocol=protocol, artifact_root=root, trusted_public_keys=trusted
    )
    return {
        "artifact_schema": "dftr.measurement.operator_score.v2",
        "status": "pass",
        "promotion_eligible": promotion,
        "report_sha256": _sha(report_path),
        "report_path": str(report_path),
    }


def attest_operator_bundle(
    *,
    artifact_root: str | Path,
    blind_manifest: str | Path,
    repo_root: str | Path,
    operator: str,
    attested_at: str,
) -> dict[str, Any]:
    root = Path(artifact_root)
    protocol, trusted = _load_bundle_protocol(root)
    inventory_check = _load_json(root / "historical_inventory_check.json")
    blind = _load_json(blind_manifest)
    attestation = build_attestation(
        protocol=protocol,
        inventory_check=inventory_check,
        blind_test_manifest=blind,
        operator=operator,
        attested_at=attested_at,
        artifact_root=root,
        trusted_public_keys=trusted,
        repo_root=repo_root,
    )
    _write_json(root / "measurement_attestation_v2.json", attestation)
    return attestation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.measurement_v2_operator")
    commands = parser.add_subparsers(dest="command", required=True)
    key = commands.add_parser("generate-key")
    key.add_argument("--private-key", required=True)
    key.add_argument("--trusted-keys", required=True)
    key.add_argument("--key-id", required=True)
    embed = commands.add_parser("embed")
    embed.add_argument("--input-jsonl", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--model-path", required=True)
    embed.add_argument("--model-id", required=True)
    embed.add_argument("--model-revision", required=True)
    embed.add_argument("--id-field", default="document_id")
    embed.add_argument("--text-field", default="text")
    embed.add_argument("--batch-size", type=int, default=32)
    freeze = commands.add_parser("freeze")
    for field in (
        "artifact-root",
        "human-source",
        "prompt-briefs",
        "control-outputs",
        "control-generation-manifest",
        "control-generation-config",
        "generation-ledger",
        "control-wrapper-receipt",
        "human-embeddings",
        "power-assumptions",
        "decision-contract",
        "dependency-lock",
        "metric-code",
        "private-key",
        "trusted-keys",
        "historical-inventory",
        "repo-root",
        "control-checkpoint-sha256",
        "decoding-policy-sha256",
        "generation-contract-sha256",
        "operator",
        "reviewed-at",
    ):
        freeze.add_argument(f"--{field}", required=True)
    freeze.add_argument("--selection-seed", default="dftr-measurement-v2-panels-v1")
    score = commands.add_parser("score")
    for field in (
        "artifact-root",
        "candidate-outputs",
        "candidate-generation-manifest",
        "candidate-generation-config",
        "generation-ledger",
        "candidate-wrapper-receipt",
        "score-embeddings",
        "candidate-checkpoint-sha256",
        "private-key",
    ):
        score.add_argument(f"--{field}", required=True)
    score.add_argument("--quality-results")
    score.add_argument("--hard-gate-results")
    attest = commands.add_parser("attest")
    for field in (
        "artifact-root",
        "blind-manifest",
        "repo-root",
        "operator",
        "attested-at",
    ):
        attest.add_argument(f"--{field}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate-key":
            result = generate_operator_key(
                args.private_key, args.trusted_keys, key_id=args.key_id
            )
        elif args.command == "embed":
            result = build_embedding_bundle(
                args.input_jsonl,
                args.output,
                model_path=args.model_path,
                model_id=args.model_id,
                model_revision=args.model_revision,
                id_field=args.id_field,
                text_field=args.text_field,
                batch_size=args.batch_size,
            )
        elif args.command == "freeze":
            result = freeze_operator_bundle(
                artifact_root=args.artifact_root,
                human_source=args.human_source,
                prompt_briefs=args.prompt_briefs,
                control_outputs=args.control_outputs,
                control_generation_manifest=args.control_generation_manifest,
                control_generation_config=args.control_generation_config,
                generation_ledger=args.generation_ledger,
                control_wrapper_receipt=args.control_wrapper_receipt,
                human_embeddings=args.human_embeddings,
                power_assumptions=args.power_assumptions,
                decision_contract=args.decision_contract,
                dependency_lock=args.dependency_lock,
                metric_code=args.metric_code,
                private_key=args.private_key,
                trusted_keys=args.trusted_keys,
                historical_inventory=args.historical_inventory,
                repo_root=args.repo_root,
                control_checkpoint_sha256=args.control_checkpoint_sha256,
                decoding_policy_sha256=args.decoding_policy_sha256,
                generation_contract_sha256=args.generation_contract_sha256,
                operator=args.operator,
                reviewed_at=args.reviewed_at,
                selection_seed=args.selection_seed,
            )
        elif args.command == "score":
            result = score_candidate_bundle(
                artifact_root=args.artifact_root,
                candidate_outputs=args.candidate_outputs,
                candidate_generation_manifest=args.candidate_generation_manifest,
                candidate_generation_config=args.candidate_generation_config,
                generation_ledger=args.generation_ledger,
                candidate_wrapper_receipt=args.candidate_wrapper_receipt,
                score_embeddings=args.score_embeddings,
                candidate_checkpoint_sha256=args.candidate_checkpoint_sha256,
                private_key=args.private_key,
                quality_results=args.quality_results,
                hard_gate_results=args.hard_gate_results,
            )
        else:
            result = attest_operator_bundle(
                artifact_root=args.artifact_root,
                blind_manifest=args.blind_manifest,
                repo_root=args.repo_root,
                operator=args.operator,
                attested_at=args.attested_at,
            )
    except (MeasurementV2Error, OSError, ValueError) as error:
        print(f"measurement-v2-operator: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
