"""Freeze the qualified 16K clean-train artifact into immutable 4K/16K prefixes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from data.materialize_scale_ladder_train_prefixes import (
    EXPECTED_CLEAN_RECORDS,
    materialize,
)
from infra.backend.volume_paths import checkpoint_volume_path


SCALE_LADDER_TRAIN_PREFIX_SCHEMA = "dftr.m2.scale_ladder_train_prefixes.v1"
SCALE_LADDER_TRAIN_PREFIX_STEP = "freeze_scale_train_prefixes"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TOP_LEVEL_KEYS = {"artifact_schema", "run", "compute", "data", "workflow"}


class ScaleLadderTrainPrefixError(ValueError):
    pass


def canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ScaleLadderTrainPrefixError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_prefix_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key) for key in ("artifact_schema", "run", "compute", "data")
    } | {
        "protocol_version": (config.get("workflow") or {}).get("protocol_version"),
        "step": (config.get("workflow") or {}).get("step"),
    }


def validate_scale_ladder_train_prefix_config(config: dict[str, Any]) -> dict[str, Any]:
    _exact(config, TOP_LEVEL_KEYS, "scale-ladder train-prefix config")
    if config.get("artifact_schema") != SCALE_LADDER_TRAIN_PREFIX_SCHEMA:
        raise ScaleLadderTrainPrefixError("unexpected scale-ladder train-prefix config schema")
    run = _exact(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") != "freeze-scale-train-prefixes"
        or run.get("budget_class") != "smoke"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 0
    ):
        raise ScaleLadderTrainPrefixError("scale-ladder train-prefix run contract is invalid")
    compute = _exact(config.get("compute"), {"gpu", "gpus", "timeout_min"}, "compute")
    if (
        str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1
        or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int)
        or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= 20
    ):
        raise ScaleLadderTrainPrefixError(
            "scale-ladder train-prefix requires one supported GPU for at most 20 minutes"
        )
    data = _exact(
        config.get("data"),
        {
            "raw_train_uri",
            "raw_train_sha256",
            "clean_train_uri",
            "source_manifest_uri",
            "output_dir_uri",
            "expected_clean_records",
        },
        "data",
    )
    for field in (
        "raw_train_uri",
        "clean_train_uri",
        "source_manifest_uri",
        "output_dir_uri",
    ):
        if not str(data.get(field) or "").startswith("modal-volume://humanwrite-checkpoints/"):
            raise ScaleLadderTrainPrefixError(f"data.{field} must use the checkpoint volume")
    raw_train_sha256 = str(data.get("raw_train_sha256") or "")
    if len(raw_train_sha256) != 64 or any(character not in "0123456789abcdef" for character in raw_train_sha256):
        raise ScaleLadderTrainPrefixError("data.raw_train_sha256 must be a lowercase SHA-256")
    if (
        data.get("expected_clean_records") != EXPECTED_CLEAN_RECORDS
        or not isinstance(data.get("expected_clean_records"), int)
        or isinstance(data.get("expected_clean_records"), bool)
    ):
        raise ScaleLadderTrainPrefixError(
            f"data.expected_clean_records must equal {EXPECTED_CLEAN_RECORDS}"
        )
    workflow = _exact(
        config.get("workflow"),
        {"protocol_version", "step", "train_prefix_contract_sha256"},
        "workflow",
    )
    if (
        workflow.get("protocol_version") != SCALE_LADDER_TRAIN_PREFIX_SCHEMA
        or workflow.get("step") != SCALE_LADDER_TRAIN_PREFIX_STEP
        or canonical_hash(train_prefix_contract_payload(config))
        != str(workflow.get("train_prefix_contract_sha256") or "")
    ):
        raise ScaleLadderTrainPrefixError("scale-ladder train-prefix workflow contract hash mismatch")
    return config


def run_scale_ladder_train_prefixes(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    validate_scale_ladder_train_prefix_config(config)
    data = config["data"]
    raw_train_path = checkpoint_volume_path(str(data["raw_train_uri"]), "/checkpoints")
    clean_train_path = checkpoint_volume_path(str(data["clean_train_uri"]), "/checkpoints")
    source_manifest_path = checkpoint_volume_path(str(data["source_manifest_uri"]), "/checkpoints")
    output_dir = checkpoint_volume_path(str(data["output_dir_uri"]), "/checkpoints")

    if not raw_train_path.is_file():
        raise ScaleLadderTrainPrefixError(f"required artifact is missing: {raw_train_path}")
    if _sha(raw_train_path) != str(data["raw_train_sha256"]):
        raise ScaleLadderTrainPrefixError("artifact hash mismatch: raw-train")
    if not clean_train_path.is_file():
        raise ScaleLadderTrainPrefixError("clean train artifact is missing")
    if not source_manifest_path.is_file():
        raise ScaleLadderTrainPrefixError("source manifest is missing")
    if output_dir.exists():
        raise ScaleLadderTrainPrefixError("scale-train prefix output directory already exists")

    bundle = materialize(
        raw_train_path=raw_train_path,
        clean_train_path=clean_train_path,
        source_manifest_path=source_manifest_path,
        output_dir=output_dir,
        expected_clean_records=int(data["expected_clean_records"]),
    )
    bundle_path = output_dir / "train_prefix_bundle.json"
    prefixes = bundle.get("artifacts") or {}
    clean_train_4096 = prefixes.get("clean_train_4096") or {}
    clean_train_16384 = prefixes.get("clean_train_16384") or {}
    manifest = {
        "run_id": run_id,
        "comparison_id": str(config["run"]["comparison_id"]),
        "arm": str(config["run"]["arm"]),
        "status": "completed",
        "protocol_version": SCALE_LADDER_TRAIN_PREFIX_SCHEMA,
        "step": SCALE_LADDER_TRAIN_PREFIX_STEP,
        "config_hash": canonical_hash(config),
        "expected_clean_records": int(data["expected_clean_records"]),
        "train_prefix_bundle_path": str(bundle_path),
        "train_prefix_bundle_sha256": _sha(bundle_path),
        "clean_train_4096_path": str(clean_train_4096.get("data_path") or ""),
        "clean_train_4096_sha256": str(clean_train_4096.get("data_sha256") or ""),
        "clean_train_16384_path": str(clean_train_16384.get("data_path") or ""),
        "clean_train_16384_sha256": str(clean_train_16384.get("data_sha256") or ""),
        "artifacts": prefixes,
    }
    checkpoint_dir = Path(os.environ["DFTR_CHECKPOINT_DIR"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
