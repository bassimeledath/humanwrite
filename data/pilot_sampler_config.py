from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PilotSamplerConfigError(ValueError):
    pass


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_sampler_config(config_path: str | Path) -> dict[str, Any]:
    operator = json.loads(_resolve(config_path).read_text(encoding="utf-8"))
    fixed_path = _resolve(operator["fixed_manifest_path"])
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    if fixed.get("artifact_schema") != "dftr.realdata_pilot_fixed_inputs.v1":
        raise PilotSamplerConfigError("unexpected real-data fixed manifest schema")
    fixed_sha = hashlib.sha256(fixed_path.read_bytes()).hexdigest()
    dev_briefs_path = _resolve(operator["dev_briefs_local_path"])
    if hashlib.sha256(dev_briefs_path.read_bytes()).hexdigest() != fixed["dev_briefs_sha256"]:
        raise PilotSamplerConfigError("dev briefs SHA does not match fixed manifest")
    dev_rows = _load_jsonl(dev_briefs_path)
    if len(dev_rows) != 64:
        raise PilotSamplerConfigError("directional sampler requires the validated 64-record dev split")
    fingerprints = [str(row.get("fingerprint") or "") for row in dev_rows]
    if any(not value for value in fingerprints) or len(fingerprints) != len(set(fingerprints)):
        raise PilotSamplerConfigError("dev brief fingerprints are invalid")
    label = "dftr-m1-realdata-pilot-directional-dev-v1"
    ranked = sorted(
        fingerprints,
        key=lambda value: (hashlib.sha256(f"{label}:{value}".encode()).hexdigest(), value),
    )
    subset = ranked[:16]
    subset_hash = hashlib.sha256("\n".join(sorted(subset)).encode()).hexdigest()

    checkpoint_local_path = _resolve(operator["checkpoint_manifest_local_path"])
    checkpoints = json.loads(checkpoint_local_path.read_text(encoding="utf-8"))
    if checkpoints.get("protocol_version") != "m1.checkpoints.v1":
        raise PilotSamplerConfigError("unexpected checkpoint manifest protocol")
    if checkpoints.get("model_base") != "Qwen/Qwen3-1.7B":
        raise PilotSamplerConfigError("checkpoint model base mismatch")
    if checkpoints.get("model_revision") != operator["model_revision"]:
        raise PilotSamplerConfigError("checkpoint model revision mismatch")
    checkpoint_rows = checkpoints.get("checkpoints") or []
    if [row.get("seed") for row in checkpoint_rows] != [11, 29, 47]:
        raise PilotSamplerConfigError("checkpoint seeds must be [11,29,47]")
    adapter_hashes = []
    for row in checkpoint_rows:
        files = row.get("checkpoint_files") or {}
        adapter_hash = str(files.get("adapter_model.safetensors") or "")
        if len(adapter_hash) != 64 or int(row.get("train_tokens", 0)) <= 0:
            raise PilotSamplerConfigError("checkpoint lacks adapter hash or train tokens")
        adapter_hashes.append(adapter_hash)
    if len(set(adapter_hashes)) != 3:
        raise PilotSamplerConfigError("checkpoint adapter hashes must differ across seeds")

    subset_manifest_path = _resolve(operator["subset_manifest_path"])
    subset_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    subset_manifest = {
        "artifact_schema": "dftr.realdata_pilot_directional_dev_subset.v1",
        "count": 16,
        "fingerprints": subset,
        "label": label,
        "subset_hash": subset_hash,
        "fixed_manifest_sha256": fixed_sha,
        "dev_briefs_sha256": fixed["dev_briefs_sha256"],
    }
    subset_manifest_path.write_text(
        json.dumps(subset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = {
        "run": {
            "comparison_id": "M1-realdata-pilot-directional-sampler-qwen3-1p7b-v1",
            "arm": "SFT",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seeds": [11, 29, 47],
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 120},
        "model": {
            "base": "Qwen/Qwen3-1.7B",
            "revision": str(operator["model_revision"]),
            "requested_revision": "main",
        },
        "data": {
            "train_split_hash": fixed["train_split_hash"],
            "dev_split_hash": fixed["dev_split_hash"],
            "train_path": fixed["train_path"],
            "dev_path": fixed["dev_path"],
            "prompt_format": fixed["prompt_format"],
            "max_input_tokens": fixed["max_input_tokens"],
            "max_new_tokens": fixed["max_new_tokens"],
        },
        "sampling": {
            "stage": "directional_default",
            "checkpoints_manifest": str(operator["checkpoint_manifest_volume_path"]),
            "checkpoints_manifest_sha256": hashlib.sha256(
                checkpoint_local_path.read_bytes()
            ).hexdigest(),
            "sampler_grid": "configs/m1/manifests/default_sampler_grid_v1.json",
            "seeds": [101, 202, 303],
            "dev_subset_fingerprints": subset,
            "dev_subset_hash": subset_hash,
            "dev_subset_manifest": _display(subset_manifest_path),
        },
        "workflow": {
            "protocol_version": "m1.realdata-pilot.v1",
            "step": "sample_sweep",
            "fixed_manifest": _display(fixed_path),
            "fixed_manifest_sha256": fixed_sha,
        },
    }
    output_path = _resolve(operator["sampler_config_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "artifact_schema": "dftr.realdata_pilot_sampler_config_index.v1",
        "checkpoint_manifest_sha256": config["sampling"]["checkpoints_manifest_sha256"],
        "sampler_config_hash": _canonical_hash(config),
        "sampler_config_path": _display(output_path),
        "subset_hash": subset_hash,
        "subset_manifest_path": _display(subset_manifest_path),
        "expected_documents": 3 * 3 * 1 * 16,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the directional real-data pilot sampler config")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(build_sampler_config(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
