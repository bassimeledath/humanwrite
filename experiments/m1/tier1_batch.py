from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

from .contracts import M1ConfigError, load_jsonl, read_structured, resolve_repo_path, write_json


ROOT = Path(__file__).resolve().parents[2]
HARNESS_SRC = ROOT / "harness" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from harness import cli as harness_cli  # noqa: E402
from harness.metrics import distribution  # noqa: E402


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_hash(path: Path, expected: Any, field_name: str) -> str:
    expected_text = str(expected or "")
    if len(expected_text) != 64:
        raise M1ConfigError(f"{field_name} must be a SHA-256 hex digest")
    actual = _sha256(path)
    if actual != expected_text:
        raise M1ConfigError(f"{field_name} mismatch: expected {expected_text}, found {actual}")
    return actual


def _local_sample_path(remote_path: str, materialized_root: Path, run_id: str) -> Path:
    marker = f"/runs/{run_id}/"
    if marker not in remote_path:
        raise M1ConfigError(f"sampler path is outside run {run_id}")
    relative = remote_path.split(marker, 1)[1]
    if not relative.startswith("samples/") or ".." in Path(relative).parts:
        raise M1ConfigError("sampler path is not a safe samples-relative path")
    local = (materialized_root / relative).resolve()
    if not local.is_file() or materialized_root.resolve() not in local.parents:
        raise M1ConfigError(f"materialized sample is missing: {relative}")
    return local


@contextmanager
def _evaluation_environment(bank: Path, manifest: Path, judge_mode: str):
    names = (
        "HARNESS_HUMAN_REFERENCE",
        "HARNESS_HUMAN_REFERENCE_MANIFEST",
        "HARNESS_JUDGE_URL",
        "HARNESS_JUDGE_TOKEN",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ["HARNESS_HUMAN_REFERENCE"] = str(bank)
    os.environ["HARNESS_HUMAN_REFERENCE_MANIFEST"] = str(manifest)
    if judge_mode == "neutral":
        os.environ.pop("HARNESS_JUDGE_URL", None)
        os.environ.pop("HARNESS_JUDGE_TOKEN", None)
    elif judge_mode == "gateway":
        if not os.environ.get("HARNESS_JUDGE_URL") or not os.environ.get("HARNESS_JUDGE_TOKEN"):
            raise M1ConfigError("gateway judge mode requires HARNESS_JUDGE_URL and HARNESS_JUDGE_TOKEN")
    else:
        raise M1ConfigError("quality_judge_mode must be neutral or gateway")
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_batch(
    config_path: str | Path,
    *,
    evaluate_fn: Callable[..., Any] = harness_cli.evaluate,
    embedder: Any = None,
) -> dict[str, Any]:
    resolved_config = resolve_repo_path(config_path)
    config = read_structured(resolved_config)
    source_index_path = resolve_repo_path(config["source_index_path"])
    materialized_root = resolve_repo_path(config["materialized_root"])
    bank_path = resolve_repo_path(config["human_bank_path"])
    manifest_path = resolve_repo_path(config["human_manifest_path"])
    calibration_path = resolve_repo_path(config["calibration_path"])
    baseline_path = resolve_repo_path(config["baseline_path"])
    _require_hash(source_index_path, config.get("source_index_sha256"), "source_index_sha256")
    _require_hash(bank_path, config.get("human_bank_sha256"), "human_bank_sha256")
    bank_id = _require_hash(
        manifest_path, config.get("human_manifest_sha256"), "human_manifest_sha256"
    )
    calibration_sha = _require_hash(
        calibration_path, config.get("calibration_sha256"), "calibration_sha256"
    )
    baseline_sha = _require_hash(baseline_path, config.get("baseline_sha256"), "baseline_sha256")
    source_index = read_structured(source_index_path)
    entries = source_index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise M1ConfigError("source index lacks entries")
    sampler_ids = [str(value) for value in config.get("sampler_ids") or []]
    if not sampler_ids:
        raise M1ConfigError("sampler_ids is required")
    selected = [entry for entry in entries if str(entry.get("sampler_id")) in sampler_ids]
    expected_count = int(config.get("expected_entry_count", 0))
    if len(selected) != expected_count:
        raise M1ConfigError(
            f"selected entry count mismatch: expected {expected_count}, found {len(selected)}"
        )
    identities = {
        (int(entry["checkpoint_seed"]), str(entry["sampler_id"]), int(entry["sampling_seed"]))
        for entry in selected
    }
    if len(identities) != len(selected):
        raise M1ConfigError("selected Tier-1 entries are not unique")
    reports_root = resolve_repo_path(config["reports_root"])
    output_index_path = resolve_repo_path(config["output_index_path"])
    reports_root.mkdir(parents=True, exist_ok=True)
    active_embedder = embedder
    if active_embedder is None:
        active_embedder = distribution._resolve_embedder(harness_cli.DEV_EMBEDDER_ID)
    output_entries = []
    run_id = str(config["sampler_run_id"])
    judge_mode = str(config.get("quality_judge_mode", "neutral"))
    with _evaluation_environment(bank_path, manifest_path, judge_mode):
        for entry in selected:
            sample_path = _local_sample_path(str(entry["samples_path"]), materialized_root, run_id)
            if len(load_jsonl(sample_path)) < 2:
                raise M1ConfigError("each Tier-1 cell requires at least two generated records")
            report_path = (
                reports_root
                / f"checkpoint-seed-{int(entry['checkpoint_seed'])}"
                / str(entry["sampler_id"])
                / f"sampling-seed-{int(entry['sampling_seed'])}.report.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = evaluate_fn(
                str(sample_path),
                str(report_path),
                embedder=active_embedder,
            )
            report_value = (
                report if isinstance(report, dict) else vars(report)
            )
            if str(report_value.get("human_reference_bank_id")) != bank_id:
                raise M1ConfigError("Tier-1 report used the wrong human bank")
            if str(report_value.get("calibration_sha256")) != calibration_sha:
                raise M1ConfigError("Tier-1 report used the wrong calibration")
            if str(report_value.get("baseline_sha256")) != baseline_sha:
                raise M1ConfigError("Tier-1 report used the wrong baseline")
            output_entry = dict(entry)
            output_entry["samples_path"] = _display_path(sample_path)
            output_entry["samples_sha256"] = _sha256(sample_path)
            output_entry["report_path"] = _display_path(report_path)
            output_entry["report_sha256"] = _sha256(report_path)
            output_entries.append(output_entry)
    artifact = {
        "artifact_schema": "m1.tier1_eval_index.v1",
        "batch_id": str(config["batch_id"]),
        "source_index_path": _display_path(source_index_path),
        "source_index_sha256": _sha256(source_index_path),
        "human_reference_bank_id": bank_id,
        "human_bank_sha256": _sha256(bank_path),
        "calibration_sha256": calibration_sha,
        "baseline_sha256": baseline_sha,
        "quality_judge_mode": judge_mode,
        "entry_count": len(output_entries),
        "entries": output_entries,
    }
    write_json(output_index_path, artifact)
    artifact["output_index_path"] = _display_path(output_index_path)
    artifact["output_index_sha256"] = _sha256(output_index_path)
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a provenance-bound Tier-1 evaluation batch")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run_batch(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
