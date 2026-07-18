"""Freeze the qualified 640-document scale-dev panel for the 4K/16K ladder."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from data.materialize_scale_ladder_dev_panels import (
    PARTITION_SEED,
    materialize,
)
from infra.backend.volume_paths import checkpoint_volume_path


SCALE_LADDER_DEV_PANEL_SCHEMA = "dftr.m2.scale_ladder_dev_panel.v1"
SCALE_LADDER_DEV_PANEL_STEP = "freeze_scale_dev_panel"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TOP_LEVEL_KEYS = {"artifact_schema", "run", "compute", "data", "workflow"}


class ScaleLadderDevPanelError(ValueError):
    pass


def canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ScaleLadderDevPanelError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key) for key in ("artifact_schema", "run", "compute", "data")
    } | {
        "protocol_version": (config.get("workflow") or {}).get("protocol_version"),
        "step": (config.get("workflow") or {}).get("step"),
    }


def validate_scale_ladder_dev_panel_config(config: dict[str, Any]) -> dict[str, Any]:
    _exact(config, TOP_LEVEL_KEYS, "scale-ladder dev-panel config")
    if config.get("artifact_schema") != SCALE_LADDER_DEV_PANEL_SCHEMA:
        raise ScaleLadderDevPanelError("unexpected scale-ladder dev-panel config schema")
    run = _exact(
        config.get("run"),
        {"comparison_id", "arm", "budget_class", "task_kind", "command", "seed"},
        "run",
    )
    if (
        not SAFE_ID_RE.fullmatch(str(run.get("comparison_id") or ""))
        or run.get("arm") != "freeze-scale-dev-panel"
        or run.get("budget_class") != "smoke"
        or run.get("task_kind") != "experiment"
        or run.get("command") != ["python", "-m", "experiments.runner"]
        or run.get("seed") != 0
    ):
        raise ScaleLadderDevPanelError("scale-ladder dev-panel run contract is invalid")
    compute = _exact(config.get("compute"), {"gpu", "gpus", "timeout_min"}, "compute")
    if (
        str(compute.get("gpu") or "").upper() not in {"L40S", "A100-80GB", "H100"}
        or compute.get("gpus") != 1
        or isinstance(compute.get("gpus"), bool)
        or not isinstance(compute.get("timeout_min"), int)
        or isinstance(compute.get("timeout_min"), bool)
        or not 0 < compute["timeout_min"] <= 20
    ):
        raise ScaleLadderDevPanelError("scale-ladder dev-panel requires one supported GPU for at most 20 minutes")
    data = _exact(
        config.get("data"),
        {
            "raw_dev_uri",
            "raw_dev_sha256",
            "clean_dev_uri",
            "raw_train_uri",
            "raw_train_sha256",
            "source_manifest_uri",
            "output_dir_uri",
            "expected_clean_records",
            "partition_seed",
        },
        "data",
    )
    for field in (
        "raw_dev_uri",
        "clean_dev_uri",
        "raw_train_uri",
        "source_manifest_uri",
        "output_dir_uri",
    ):
        if not str(data.get(field) or "").startswith("modal-volume://humanwrite-checkpoints/"):
            raise ScaleLadderDevPanelError(f"data.{field} must use the checkpoint volume")
    for field in ("raw_dev_sha256", "raw_train_sha256"):
        value = str(data.get(field) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ScaleLadderDevPanelError(f"data.{field} must be a lowercase SHA-256")
    if (
        data.get("expected_clean_records") != 640
        or not isinstance(data.get("expected_clean_records"), int)
        or isinstance(data.get("expected_clean_records"), bool)
    ):
        raise ScaleLadderDevPanelError("data.expected_clean_records must equal 640")
    if data.get("partition_seed") != PARTITION_SEED:
        raise ScaleLadderDevPanelError("scale-ladder dev-panel partition seed is frozen")
    workflow = _exact(
        config.get("workflow"),
        {"protocol_version", "step", "panel_contract_sha256"},
        "workflow",
    )
    if (
        workflow.get("protocol_version") != SCALE_LADDER_DEV_PANEL_SCHEMA
        or workflow.get("step") != SCALE_LADDER_DEV_PANEL_STEP
        or canonical_hash(panel_contract_payload(config))
        != str(workflow.get("panel_contract_sha256") or "")
    ):
        raise ScaleLadderDevPanelError("scale-ladder dev-panel workflow contract hash mismatch")
    return config


def run_scale_ladder_dev_panel(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    validate_scale_ladder_dev_panel_config(config)
    data = config["data"]
    raw_dev_path = checkpoint_volume_path(str(data["raw_dev_uri"]), "/checkpoints")
    clean_dev_path = checkpoint_volume_path(str(data["clean_dev_uri"]), "/checkpoints")
    raw_train_path = checkpoint_volume_path(str(data["raw_train_uri"]), "/checkpoints")
    source_manifest_path = checkpoint_volume_path(
        str(data["source_manifest_uri"]), "/checkpoints"
    )
    output_dir = checkpoint_volume_path(str(data["output_dir_uri"]), "/checkpoints")

    for path, expected in (
        (raw_dev_path, str(data["raw_dev_sha256"])),
        (raw_train_path, str(data["raw_train_sha256"])),
    ):
        if not path.is_file():
            raise ScaleLadderDevPanelError(f"required artifact is missing: {path}")
        if _sha(path) != expected:
            raise ScaleLadderDevPanelError(f"artifact hash mismatch: {path.name}")
    if not clean_dev_path.is_file():
        raise ScaleLadderDevPanelError("clean scale-dev artifact is missing")
    if not source_manifest_path.is_file():
        raise ScaleLadderDevPanelError("source manifest is missing")
    if output_dir.exists():
        raise ScaleLadderDevPanelError("scale-dev panel output directory already exists")

    bundle = materialize(
        raw_dev_path=raw_dev_path,
        clean_dev_path=clean_dev_path,
        raw_train_path=raw_train_path,
        source_manifest_path=source_manifest_path,
        output_dir=output_dir,
        partition_seed=str(data["partition_seed"]),
    )
    panel_bundle_path = output_dir / "panel_bundle.json"
    prompt_sources = ((bundle.get("artifacts") or {}).get("prompt_sources") or {})
    manifest = {
        "run_id": run_id,
        "comparison_id": str(config["run"]["comparison_id"]),
        "arm": str(config["run"]["arm"]),
        "status": "completed",
        "protocol_version": SCALE_LADDER_DEV_PANEL_SCHEMA,
        "step": SCALE_LADDER_DEV_PANEL_STEP,
        "config_hash": canonical_hash(config),
        "expected_clean_records": int(data["expected_clean_records"]),
        "panel_bundle_path": str(panel_bundle_path),
        "panel_bundle_sha256": _sha(panel_bundle_path),
        "prompt_sources_path": str(prompt_sources.get("data_path") or ""),
        "prompt_sources_sha256": str(prompt_sources.get("data_sha256") or ""),
        "artifacts": bundle.get("artifacts") or {},
    }
    checkpoint_dir = Path(os.environ["DFTR_CHECKPOINT_DIR"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
