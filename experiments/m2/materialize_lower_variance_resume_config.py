"""Bind one durable step-boundary checkpoint into a recovery config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from experiments.m2.lower_variance_train import (
    _directory_file_map,
    canonical_hash,
    validate_lower_variance_config,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(
    *,
    source_config: Path,
    checkpoint_dir: Path,
    remote_checkpoint_path: str,
    output: Path,
) -> dict:
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    validate_lower_variance_config(config)
    arm = str(config["execution"]["arm"])
    if arm != "TOKEN_MOMENT" or any(config["resume"].values()):
        raise ValueError("recovery materializer requires the original token-moment config")
    required = (
        checkpoint_dir / "adapter_model.safetensors",
        checkpoint_dir / "adapter_config.json",
        checkpoint_dir / "training_state.pt",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("resume checkpoint is incomplete")
    source_config_sha256 = canonical_hash(config)
    config["resume"][arm] = {
        "path": remote_checkpoint_path,
        "adapter_model_sha256": _sha(required[0]),
        "adapter_config_sha256": _sha(required[1]),
        "training_state_sha256": _sha(required[2]),
        "file_manifest_sha256": canonical_hash(
            _directory_file_map(checkpoint_dir, "token-moment durable resume")
        ),
        "source_config_sha256": source_config_sha256,
    }
    validate_lower_variance_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "arm": arm,
        "source_config_sha256": source_config_sha256,
        "resume_path": remote_checkpoint_path,
        "resume_file_manifest_sha256": config["resume"][arm][
            "file_manifest_sha256"
        ],
        "output": str(output),
        "output_config_sha256": canonical_hash(config),
        "method_contract_sha256": config["workflow"]["method_contract_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--remote-checkpoint-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                source_config=args.source_config,
                checkpoint_dir=args.checkpoint_dir,
                remote_checkpoint_path=args.remote_checkpoint_path,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
