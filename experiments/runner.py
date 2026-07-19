from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from data.pipeline import DEFAULT_OUTPUT as DEFAULT_DATA_OUTPUT
from experiments.m1.workflow import run_m1
from experiments.m2.dft import DFT_SCHEMA, DFT_STEP, run_dft
from experiments.m2.generate_dft import (
    GENERATION_SCHEMA,
    GENERATION_STEP,
    run_generate_dft,
)
from experiments.m2.generate_lower_variance import (
    GENERATION_SCHEMA as LOWER_VARIANCE_GENERATION_SCHEMA,
    GENERATION_STEP as LOWER_VARIANCE_GENERATION_STEP,
    run_generate_lower_variance,
)
from experiments.m2.estimator_audit import (
    ESTIMATOR_AUDIT_SCHEMA,
    ESTIMATOR_AUDIT_STEP,
    run_estimator_audit,
)
from experiments.m2.prepare_dft import (
    PREPARE_DFT_SCHEMA,
    PREPARE_DFT_STEP,
    run_prepare_dft,
)
from experiments.m2.lower_variance_train import (
    LOWER_VARIANCE_CONFIRMATION_SCHEMA,
    LOWER_VARIANCE_SCHEMA,
    LOWER_VARIANCE_STEP,
    run_lower_variance,
)
from experiments.m2.scale_ladder_dev_panel import (
    SCALE_LADDER_DEV_PANEL_SCHEMA,
    SCALE_LADDER_DEV_PANEL_STEP,
    run_scale_ladder_dev_panel,
)
from experiments.m2.scale_ladder_train_prefixes import (
    SCALE_LADDER_TRAIN_PREFIX_SCHEMA,
    SCALE_LADDER_TRAIN_PREFIX_STEP,
    run_scale_ladder_train_prefixes,
)
from experiments.m2.scale_ladder_witness import (
    SCALE_LADDER_WITNESS_SCHEMA,
    SCALE_LADDER_WITNESS_STEP,
    run_scale_ladder_witness,
)
from experiments.m2.scale_ladder_token_lengths import (
    TOKEN_LENGTH_SCHEMA,
    TOKEN_LENGTH_STEP,
    run_token_length_normalization,
)
from experiments.tier0.metrics import batch_diagnostics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_DATA = DEFAULT_DATA_OUTPUT / "dev_briefs.jsonl"
DEFAULT_SPLIT_HASHES = DEFAULT_DATA_OUTPUT / "split_hashes.json"


def canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML experiment configs") from exc
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("config must decode to a mapping")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _toy_embedder(texts: list[str]):
    rows = []
    for text in texts:
        lower = text.casefold()
        rows.append(
            [
                len(text),
                lower.count("e"),
                lower.count("a"),
                lower.count("the"),
                lower.count("and"),
            ]
        )
    import numpy as np

    return np.asarray(rows, dtype=float)


def _generate_completion(record: dict[str, Any]) -> str:
    completion = str(record["completion"]).strip()
    return completion


def _validate_train_hash(config: dict[str, Any]) -> str:
    expected = str((config.get("data") or {}).get("train_split_hash", ""))
    actual = json.loads(DEFAULT_SPLIT_HASHES.read_text(encoding="utf-8"))["train"]
    if expected and expected != actual:
        raise ValueError(f"train_split_hash mismatch: config={expected} actual={actual}")
    return actual


def run_smoke(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    comparison_id = str((config.get("run") or {}).get("comparison_id", "unknown-comparison"))
    output_dir = ROOT / "experiments" / comparison_id / run_id
    checkpoint_dir = Path(os.environ.get("DFTR_CHECKPOINT_DIR", output_dir / "checkpoint"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_hash = _validate_train_hash(config)
    dev_records = _load_jsonl(DEFAULT_DEV_DATA)
    generated = []
    for record in dev_records:
        generated.append(
            {
                "fineweb_id": record["fineweb_id"],
                "prompt": record["user_prompt"],
                "outline": record["outline"],
                "target_length": record["target_length"],
                "reference_completion": record["completion"],
                "generated_completion": _generate_completion(record),
                "output": _generate_completion(record),
            }
        )

    diagnostics = batch_diagnostics(
        [row["generated_completion"] for row in generated],
        [row["reference_completion"] for row in generated],
        [row["outline"] for row in generated],
        _toy_embedder,
        targets=[int(row["target_length"]) for row in generated],
    )
    manifest = {
        "run_id": run_id,
        "comparison_id": comparison_id,
        "arm": str((config.get("run") or {}).get("arm", "M0")),
        "status": "completed",
        "mode": "offline_smoke",
        "train_split_hash": train_hash,
        "config_hash": canonical_hash(config),
        "sample_count": len(generated),
        "artifacts": {
            "samples_jsonl": str((checkpoint_dir / "samples.jsonl").resolve()),
            "metrics_json": str((checkpoint_dir / "metrics.json").resolve()),
        },
    }
    sample_payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in generated) + "\n"
    metrics_payload = json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    config_payload = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    for base in (output_dir, checkpoint_dir):
        (base / "samples.jsonl").write_text(sample_payload, encoding="utf-8")
        (base / "metrics.json").write_text(metrics_payload, encoding="utf-8")
        (base / "run_manifest.json").write_text(manifest_payload, encoding="utf-8")
        (base / "config.json").write_text(config_payload, encoding="utf-8")
    if checkpoint_dir != output_dir:
        shutil.copy2(checkpoint_dir / "run_manifest.json", output_dir / "checkpoint_manifest.json")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline M0 experiment smoke runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(Path(args.config).resolve())
    workflow = config.get("workflow") or {}
    protocol, step = workflow.get("protocol_version"), workflow.get("step")
    if protocol == PREPARE_DFT_SCHEMA and step != PREPARE_DFT_STEP:
        raise ValueError("training-bandwidth protocol requires prepare_dft")
    if protocol == DFT_SCHEMA and step != DFT_STEP:
        raise ValueError("score-function MMD protocol requires train_dft")
    if step == PREPARE_DFT_STEP and protocol != PREPARE_DFT_SCHEMA:
        raise ValueError("prepare_dft requires the frozen training-bandwidth protocol")
    if step == DFT_STEP and protocol != DFT_SCHEMA:
        raise ValueError("train_dft requires the frozen M2 score-function MMD protocol")
    if protocol == GENERATION_SCHEMA and step != GENERATION_STEP:
        raise ValueError("adapter-native generation protocol requires generate_dft")
    if step == GENERATION_STEP and protocol != GENERATION_SCHEMA:
        raise ValueError("generate_dft requires the frozen adapter-native generation protocol")
    if (
        protocol == LOWER_VARIANCE_GENERATION_SCHEMA
        and step != LOWER_VARIANCE_GENERATION_STEP
    ):
        raise ValueError(
            "lower-variance generation protocol requires generate_lower_variance"
        )
    if (
        step == LOWER_VARIANCE_GENERATION_STEP
        and protocol != LOWER_VARIANCE_GENERATION_SCHEMA
    ):
        raise ValueError(
            "generate_lower_variance requires the frozen lower-variance generation protocol"
        )
    if protocol == ESTIMATOR_AUDIT_SCHEMA and step != ESTIMATOR_AUDIT_STEP:
        raise ValueError("estimator-audit protocol requires audit_estimator")
    if step == ESTIMATOR_AUDIT_STEP and protocol != ESTIMATOR_AUDIT_SCHEMA:
        raise ValueError("audit_estimator requires the frozen estimator-audit protocol")
    if protocol == SCALE_LADDER_DEV_PANEL_SCHEMA and step != SCALE_LADDER_DEV_PANEL_STEP:
        raise ValueError("scale-ladder dev-panel protocol requires freeze_scale_dev_panel")
    if step == SCALE_LADDER_DEV_PANEL_STEP and protocol != SCALE_LADDER_DEV_PANEL_SCHEMA:
        raise ValueError("freeze_scale_dev_panel requires the frozen scale-ladder dev-panel protocol")
    if protocol == SCALE_LADDER_TRAIN_PREFIX_SCHEMA and step != SCALE_LADDER_TRAIN_PREFIX_STEP:
        raise ValueError("scale-ladder train-prefix protocol requires freeze_scale_train_prefixes")
    if step == SCALE_LADDER_TRAIN_PREFIX_STEP and protocol != SCALE_LADDER_TRAIN_PREFIX_SCHEMA:
        raise ValueError("freeze_scale_train_prefixes requires the frozen scale-ladder train-prefix protocol")
    if protocol == SCALE_LADDER_WITNESS_SCHEMA and step != SCALE_LADDER_WITNESS_STEP:
        raise ValueError("scale-ladder witness protocol requires generate_scale_ladder_witness")
    if step == SCALE_LADDER_WITNESS_STEP and protocol != SCALE_LADDER_WITNESS_SCHEMA:
        raise ValueError("generate_scale_ladder_witness requires the frozen scale-ladder witness protocol")
    if protocol == TOKEN_LENGTH_SCHEMA and step != TOKEN_LENGTH_STEP:
        raise ValueError("token-length protocol requires normalize_scale_ladder_token_lengths")
    if step == TOKEN_LENGTH_STEP and protocol != TOKEN_LENGTH_SCHEMA:
        raise ValueError("normalize_scale_ladder_token_lengths requires its frozen protocol")
    lower_variance_protocols = {
        LOWER_VARIANCE_SCHEMA,
        LOWER_VARIANCE_CONFIRMATION_SCHEMA,
    }
    if protocol in lower_variance_protocols and step != LOWER_VARIANCE_STEP:
        raise ValueError("lower-variance protocol requires train_lower_variance")
    if step == LOWER_VARIANCE_STEP and protocol not in lower_variance_protocols:
        raise ValueError("train_lower_variance requires a frozen lower-variance protocol")
    if protocol == PREPARE_DFT_SCHEMA:
        manifest = run_prepare_dft(config, args.run_id)
    elif protocol == DFT_SCHEMA:
        manifest = run_dft(config, args.run_id)
    elif protocol == GENERATION_SCHEMA:
        manifest = run_generate_dft(config, args.run_id)
    elif protocol == LOWER_VARIANCE_GENERATION_SCHEMA:
        manifest = run_generate_lower_variance(config, args.run_id)
    elif protocol == ESTIMATOR_AUDIT_SCHEMA:
        manifest = run_estimator_audit(config, args.run_id)
    elif protocol == SCALE_LADDER_DEV_PANEL_SCHEMA:
        manifest = run_scale_ladder_dev_panel(config, args.run_id)
    elif protocol == SCALE_LADDER_TRAIN_PREFIX_SCHEMA:
        manifest = run_scale_ladder_train_prefixes(config, args.run_id)
    elif protocol == SCALE_LADDER_WITNESS_SCHEMA:
        manifest = run_scale_ladder_witness(config, args.run_id)
    elif protocol == TOKEN_LENGTH_SCHEMA:
        manifest = run_token_length_normalization(config, args.run_id)
    elif protocol in lower_variance_protocols:
        manifest = run_lower_variance(config, args.run_id)
    elif (
        str(workflow.get("protocol_version", "")).casefold().startswith("m1")
        or str(workflow.get("step", "")).casefold() == "replay_equivalence"
    ):
        manifest = run_m1(config, args.run_id)
    else:
        manifest = run_smoke(config, args.run_id)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
