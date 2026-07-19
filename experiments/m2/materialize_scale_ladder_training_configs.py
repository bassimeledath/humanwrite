"""Bind the exact 4K corpus and baseline witness into matched training arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.m2.lower_variance_train import (
    EPOCH_SCHEDULE,
    LOWER_VARIANCE_STEP,
    canonical_hash,
    method_contract_payload,
    validate_lower_variance_config,
)


def materialize(
    *, template_path: Path, briefs_sha256: str, witness_manifest_path: Path,
    output_dir: Path,
) -> dict:
    base = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    witness = json.loads(witness_manifest_path.read_text(encoding="utf-8"))
    if (
        witness.get("artifact_schema") != "dftr.m2.lower_variance_baseline_witness.v2"
        or witness.get("scientific_role") != "training_only_not_evaluation"
        or witness.get("documents") != 4096
        or witness.get("briefs_sha256") != briefs_sha256
    ):
        raise ValueError("4K witness manifest does not bind the exact training corpus")
    base["run"]["comparison_id"] = "M2-scale-ladder-4b-4096-v1"
    base["compute"] = {"gpu": "H100", "gpus": 1, "timeout_min": 120}
    base["data"].update(
        anchor_path="/checkpoints/data/m2-scale-ladder-v1/train-briefs-4096-token-normalized-v1.jsonl",
        anchor_sha256=briefs_sha256,
        witness_generated_path=witness["output_path"],
        witness_generated_sha256=witness["output_sha256"],
        witness_generation_contract_sha256=witness["generation_contract_sha256"],
    )
    base["representation"]["max_tokens"] = 128
    base["training"].update(
        steps=2048,
        batch_size=2,
        learning_rate=0.00001,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        max_input_tokens=1024,
        checkpoint_every=64,
        schedule=EPOCH_SCHEDULE,
    )
    base["resume"] = {"SFT": None, "MMD_WITNESS": None}
    base["workflow"]["step"] = LOWER_VARIANCE_STEP
    base["workflow"]["method_contract_sha256"] = canonical_hash(
        method_contract_payload(base)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"configs": {}, "method_contract_sha256": base["workflow"]["method_contract_sha256"]}
    for arm_id in ("SFT", "MMD_WITNESS"):
        config = json.loads(json.dumps(base))
        config["execution"]["arm"] = arm_id
        validate_lower_variance_config(config)
        path = output_dir / f"m2_scale_ladder_4b_4096_{arm_id.casefold()}_v1.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result["configs"][arm_id] = {"path": str(path), "sha256": canonical_hash(config)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--briefs-sha256", required=True)
    parser.add_argument("--witness-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        template_path=args.template,
        briefs_sha256=args.briefs_sha256,
        witness_manifest_path=args.witness_manifest,
        output_dir=args.output_dir,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
