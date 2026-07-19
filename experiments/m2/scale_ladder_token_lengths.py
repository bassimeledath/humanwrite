"""Deterministically replace 4K provider length guesses with tokenizer counts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from data.lower_variance_briefs import (
    deterministic_empty_outline_ids,
    validate_assembled_brief,
)
from experiments.m2.lower_variance_train import BASE_MODEL, BASE_REVISION, canonical_hash


TOKEN_LENGTH_SCHEMA = "dftr.m2.scale_ladder_token_lengths.v1"
TOKEN_LENGTH_STEP = "normalize_scale_ladder_token_lengths"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TokenLengthNormalizationError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_length_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in (
        "artifact_schema", "run", "compute", "model", "initial_adapter", "data", "runtime"
    )} | {
        "protocol_version": (config.get("workflow") or {}).get("protocol_version"),
        "step": (config.get("workflow") or {}).get("step"),
    }


def validate_token_length_config(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != {
        "artifact_schema", "run", "compute", "model", "initial_adapter", "data", "runtime", "workflow"
    }:
        raise TokenLengthNormalizationError("token-length config has unexpected keys")
    if config.get("artifact_schema") != TOKEN_LENGTH_SCHEMA:
        raise TokenLengthNormalizationError("unexpected token-length schema")
    run = config.get("run") or {}
    if (
        set(run) != {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"}
        or not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") != "exact-token-lengths-4096"
        or run.get("budget_class") != "smoke"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 0
    ):
        raise TokenLengthNormalizationError("token-length run contract is invalid")
    if config.get("compute") != {"gpu": "L40S", "gpus": 1, "timeout_min": 20}:
        raise TokenLengthNormalizationError("token-length compute contract is invalid")
    if config.get("model") != {
        "base": BASE_MODEL, "revision": BASE_REVISION, "torch_dtype": "bfloat16"
    }:
        raise TokenLengthNormalizationError("token-length model is not frozen")
    adapter = config.get("initial_adapter") or {}
    if set(adapter) != {
        "path", "adapter_model_sha256", "adapter_config_sha256", "file_manifest_sha256"
    } or not str(adapter.get("path") or "").startswith("/checkpoints/"):
        raise TokenLengthNormalizationError("token-length tokenizer binding is invalid")
    data = config.get("data") or {}
    if set(data) != {
        "source_path", "source_sha256", "brief_path", "brief_sha256", "output_path",
        "manifest_path", "expected_documents"
    }:
        raise TokenLengthNormalizationError("token-length data contract is invalid")
    for field in ("source_path", "brief_path", "output_path", "manifest_path"):
        if not str(data.get(field) or "").startswith("/checkpoints/"):
            raise TokenLengthNormalizationError(f"data.{field} must use /checkpoints")
    for field in ("source_sha256", "brief_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(data.get(field) or "")):
            raise TokenLengthNormalizationError(f"data.{field} must be a SHA-256")
    if data.get("expected_documents") != 4096:
        raise TokenLengthNormalizationError("token-length normalization requires 4096 rows")
    runtime = config.get("runtime") or {}
    if runtime != {"transformers_version": "4.57.6", "tokenizer_local_files_only": True}:
        raise TokenLengthNormalizationError("token-length runtime is not frozen")
    workflow = config.get("workflow") or {}
    if (
        set(workflow) != {"protocol_version", "step", "contract_sha256"}
        or workflow.get("protocol_version") != TOKEN_LENGTH_SCHEMA
        or workflow.get("step") != TOKEN_LENGTH_STEP
        or workflow.get("contract_sha256") != canonical_hash(token_length_contract_payload(config))
    ):
        raise TokenLengthNormalizationError("token-length contract hash mismatch")
    return config


def normalize_rows(
    sources: list[dict[str, Any]], briefs: list[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(sources) != 4096 or len(briefs) != 4096:
        raise TokenLengthNormalizationError("normalization requires exact 4096-row inputs")
    source_by_id = {str(row.get("fingerprint") or ""): row for row in sources}
    if "" in source_by_id or len(source_by_id) != 4096:
        raise TokenLengthNormalizationError("normalization sources require unique fingerprints")
    empty_ids = deterministic_empty_outline_ids(sources)
    normalized = json.loads(json.dumps(briefs))
    old_lengths: list[int] = []
    new_lengths: list[int] = []
    for row in normalized:
        fingerprint = str(row.get("fingerprint") or "")
        source = source_by_id.get(fingerprint)
        if source is None:
            raise TokenLengthNormalizationError("brief fingerprint is absent from sources")
        old_lengths.append(int(row["target_length"]))
        length = len(tokenizer.encode(str(row["completion"]), add_special_tokens=False))
        if not 1 <= length <= 4096:
            raise TokenLengthNormalizationError(f"exact token length outside contract: {length}")
        row["target_length"] = length
        new_lengths.append(length)
        validate_assembled_brief(
            row, source=source, force_empty_outline=fingerprint in empty_ids
        )
    ordered = sorted(new_lengths)
    return normalized, {
        "rows": 4096,
        "changed_rows": sum(old != new for old, new in zip(old_lengths, new_lengths)),
        "new_min": ordered[0],
        "new_median": ordered[len(ordered) // 2],
        "new_p95": ordered[int(len(ordered) * 0.95)],
        "new_max": ordered[-1],
        "all_lengths_exact": True,
    }


def run_token_length_normalization(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    validate_token_length_config(config)
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != config["runtime"]["transformers_version"]:
        raise TokenLengthNormalizationError("normalization Transformers version mismatch")
    data = config["data"]
    source_path, brief_path = Path(data["source_path"]), Path(data["brief_path"])
    output_path, manifest_path = Path(data["output_path"]), Path(data["manifest_path"])
    if output_path.exists() or manifest_path.exists():
        raise TokenLengthNormalizationError("normalized output already exists")
    for path, expected, label in (
        (source_path, data["source_sha256"], "source"),
        (brief_path, data["brief_sha256"], "brief"),
    ):
        if not path.is_file() or _sha(path) != expected:
            raise TokenLengthNormalizationError(f"{label} input hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        config["initial_adapter"]["path"], local_files_only=True, trust_remote_code=True
    )
    rows, stats = normalize_rows(_rows(source_path), _rows(brief_path), tokenizer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    manifest = {
        "artifact_schema": "dftr.exact_token_length_normalization.v2",
        "status": "completed",
        "run_id": run_id,
        "config_sha256": canonical_hash(config),
        "source_path": str(source_path),
        "source_sha256": data["source_sha256"],
        "before_path": str(brief_path),
        "before_sha256": data["brief_sha256"],
        "output_path": str(output_path),
        "output_sha256": _sha(output_path),
        "tokenizer": BASE_MODEL,
        "tokenizer_revision": BASE_REVISION,
        **stats,
    }
    manifest_path.write_text(
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
    "TOKEN_LENGTH_SCHEMA", "TOKEN_LENGTH_STEP", "normalize_rows",
    "run_token_length_normalization", "token_length_contract_payload",
    "validate_token_length_config",
]
