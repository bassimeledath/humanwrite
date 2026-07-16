from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

from .contracts import M1ConfigError, file_sha256, load_fixed_split_hashes, load_jsonl, read_structured, resolve_repo_path, write_json


ROOT = Path(__file__).resolve().parents[2]
HARNESS_SRC = ROOT / "harness" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from harness.metrics import validity  # noqa: E402


CONTINUOUS_INTERVAL_METHOD = "deterministic central order-statistic interval"
REPETITION_INTERVAL_METHOD = "Wilson score interval for binomial proportion"
CALIBRATION_CONFIDENCE_LEVEL = 0.95
CALIBRATION_Z_95 = 1.959963984540054


def _wilson_interval(successes: int, trials: int, confidence_level: float) -> dict[str, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise M1ConfigError("Wilson interval requires 0 <= successes <= positive trials")
    if confidence_level != CALIBRATION_CONFIDENCE_LEVEL:
        raise M1ConfigError("Wilson interval confidence level must be the frozen 0.95")
    z = CALIBRATION_Z_95
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (z / denominator) * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return {"low": max(0.0, center - half_width), "high": min(1.0, center + half_width)}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_non_placeholder(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("__M1_"):
        raise M1ConfigError(f"{field_name} must be replaced before analysis")
    return text


def _load_eval_index(path: str | Path) -> list[dict[str, Any]]:
    value = read_structured(resolve_repo_path(path))
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise M1ConfigError("Tier 1 eval index must contain a non-empty entries list")
    return entries


def _read_report(entry: dict[str, Any]) -> dict[str, Any]:
    report_path = _require_non_placeholder(entry.get("report_path"), field_name="report_path")
    report = read_structured(resolve_repo_path(report_path))
    required = {
        "S",
        "semantic_mmd",
        "semantic_mmd_delta_vs_human_floor",
        "lexical_l2",
        "structural_dist",
        "gate_outline_fact_recall",
        "gate_unsupported_claim_rate",
        "gate_language_integrity",
        "gate_no_collapse",
        "human_reference_bank_id",
        "calibration_sha256",
        "baseline_sha256",
    }
    missing = sorted(key for key in required if key not in report)
    if missing:
        raise M1ConfigError("Tier 1 report is missing fields: " + ", ".join(missing))
    return report


def _samples_metrics(samples_path: str | Path) -> dict[str, float]:
    records = load_jsonl(resolve_repo_path(samples_path))
    generated = [str(record["generated_completion"]) for record in records]
    outlines = [record.get("outline", []) for record in records]
    return {
        "outline_fact_recall": float(validity.outline_fact_recall(generated, outlines)),
        "unsupported_claim_rate": float(validity.unsupported_claim_rate(generated, outlines)),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        raise M1ConfigError("at least two Tier 1 reports are required to compute a baseline standard deviation")
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def build_baseline_stats(config_path: str) -> dict[str, Any]:
    config = read_structured(resolve_repo_path(config_path))
    fixed_hashes = load_fixed_split_hashes()
    baseline_sampler_id = _require_non_placeholder(
        config.get("baseline_sampler_id"), field_name="baseline_sampler_id"
    )
    if baseline_sampler_id != "default_t1.0_p1.0":
        raise M1ConfigError("M1 bootstrap baseline must use default_t1.0_p1.0")
    eval_index = _load_eval_index(_require_non_placeholder(config.get("report_index"), field_name="report_index"))
    selected = [entry for entry in eval_index if str(entry.get("sampler_id")) == baseline_sampler_id]
    if not selected:
        raise M1ConfigError(f"no eval index rows matched sampler {baseline_sampler_id}")
    if len(selected) < 2:
        raise M1ConfigError("baseline statistics require at least two selected Tier 1 reports")
    reports = [_read_report(entry) for entry in selected]
    expected_calibration = _require_non_placeholder(
        config.get("expected_calibration_sha256"), field_name="expected_calibration_sha256"
    )
    expected_human_bank = _require_non_placeholder(
        config.get("expected_human_reference_bank_id"),
        field_name="expected_human_reference_bank_id",
    )
    if any(str(report["calibration_sha256"]) != expected_calibration for report in reports):
        raise M1ConfigError("bootstrap reports do not use the operator-frozen calibration")
    if any(str(report["human_reference_bank_id"]) != expected_human_bank for report in reports):
        raise M1ConfigError("bootstrap reports do not use the frozen independent human bank")
    sample_metrics = [_samples_metrics(entry["samples_path"]) for entry in selected]
    baseline = {
        "artifact_schema": "m1.baseline_stats.review.v1",
        "semantic_mmd": {
            "mean": _mean([float(report["semantic_mmd"]) for report in reports]),
            "std": _std([float(report["semantic_mmd"]) for report in reports]),
        },
        "lexical_l2": {
            "mean": _mean([float(report["lexical_l2"]) for report in reports]),
            "std": _std([float(report["lexical_l2"]) for report in reports]),
        },
        "structural_dist": {
            "mean": _mean([float(report["structural_dist"]) for report in reports]),
            "std": _std([float(report["structural_dist"]) for report in reports]),
        },
        "outline_fact_recall": {
            "mean": _mean([metric["outline_fact_recall"] for metric in sample_metrics]),
        },
        "unsupported_claim_rate": {
            "mean": _mean([metric["unsupported_claim_rate"] for metric in sample_metrics]),
        },
        "baseline_sampler_id": baseline_sampler_id,
        "sample_count": len(selected),
        "train_split_hash": fixed_hashes["train"],
        "dev_split_hash": fixed_hashes["dev"],
        "human_reference_bank_id": expected_human_bank,
        "calibration_sha256": expected_calibration,
    }
    output_path = resolve_repo_path(_require_non_placeholder(config.get("output_path"), field_name="output_path"))
    write_json(output_path, baseline)
    baseline["output_path"] = str(output_path.resolve())
    return baseline


def freeze_sampler(config_path: str) -> dict[str, Any]:
    config = read_structured(resolve_repo_path(config_path))
    eval_index = _load_eval_index(_require_non_placeholder(config.get("report_index"), field_name="report_index"))
    default_sampler_id = str(config.get("default_sampler_id", "default_t1.0_p1.0"))
    expected_baseline = _require_non_placeholder(
        config.get("expected_baseline_sha256"), field_name="expected_baseline_sha256"
    )
    expected_calibration = _require_non_placeholder(
        config.get("expected_calibration_sha256"), field_name="expected_calibration_sha256"
    )
    expected_human_bank = _require_non_placeholder(
        config.get("expected_human_reference_bank_id"),
        field_name="expected_human_reference_bank_id",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in eval_index:
        grouped.setdefault(str(entry.get("sampler_id")), []).append(entry)
    if default_sampler_id not in grouped:
        raise M1ConfigError(f"default sampler {default_sampler_id} is absent from the eval index")
    summaries = []
    for sampler_id, entries in grouped.items():
        reports = [_read_report(entry) for entry in entries]
        if any(str(report["baseline_sha256"]) != expected_baseline for report in reports):
            raise M1ConfigError("freeze reports do not use the operator-frozen baseline")
        if any(str(report["calibration_sha256"]) != expected_calibration for report in reports):
            raise M1ConfigError("freeze reports do not use the operator-frozen calibration")
        if any(str(report["human_reference_bank_id"]) != expected_human_bank for report in reports):
            raise M1ConfigError("freeze reports do not use the frozen independent human bank")
        gate_fields = (
            "gate_outline_fact_recall",
            "gate_unsupported_claim_rate",
            "gate_language_integrity",
            "gate_no_collapse",
        )
        if any(any(bool(report[field]) is False for field in gate_fields) for report in reports):
            continue
        scores = [float(report["S"]) for report in reports]
        mean_score = _mean(scores)
        stderr = 0.0 if len(scores) < 2 else _std(scores) / math.sqrt(len(scores))
        ci_half_width = 1.96 * stderr
        kl_values = [entry.get("kl_drift") for entry in entries if entry.get("kl_drift") is not None]
        summary = {
            "sampler_id": sampler_id,
            "mean_S": mean_score,
            "ci_low": mean_score - ci_half_width,
            "ci_high": mean_score + ci_half_width,
            "temperature": float(entries[0]["temperature"]),
            "top_p": float(entries[0]["top_p"]),
            "kl_drift_mean": None if not kl_values else _mean([float(value) for value in kl_values]),
            "sample_count": len(entries),
        }
        summaries.append(summary)
    if not summaries:
        raise M1ConfigError("no sampler settings passed every hard gate")
    summaries.sort(key=lambda item: item["mean_S"])
    best = summaries[0]
    overlapping = [
        item
        for item in summaries
        if not (item["ci_low"] > best["ci_high"] or item["ci_high"] < best["ci_low"])
    ]
    chosen = best
    tie_reason = None
    if len(overlapping) > 1:
        default_candidate = next((item for item in overlapping if item["sampler_id"] == default_sampler_id), None)
        if default_candidate is not None:
            chosen = default_candidate
            tie_reason = "default_sampler_inside_uncertainty"
        else:
            if any(item["kl_drift_mean"] is None for item in overlapping):
                raise M1ConfigError("KL drift is required to break a non-default uncertainty tie")
            overlapping.sort(
                key=lambda item: (
                    float(item["kl_drift_mean"]),
                    float(item["temperature"]),
                    float(item["top_p"]),
                    str(item["sampler_id"]),
                )
            )
            chosen = overlapping[0]
            tie_reason = "kl_drift_then_lower_temperature"
    artifact = {
        "selected_sampler_id": chosen["sampler_id"],
        "selected_temperature": chosen["temperature"],
        "selected_top_p": chosen["top_p"],
        "default_sampler_id": default_sampler_id,
        "decision_rule": "lowest_mean_S subject to all gates; tie inside uncertainty -> default -> lower KL drift -> lower temperature/top_p",
        "tie_break": tie_reason,
        "summaries": summaries,
    }
    output_path = resolve_repo_path(_require_non_placeholder(config.get("output_path"), field_name="output_path"))
    write_json(output_path, artifact)
    artifact["output_path"] = str(output_path.resolve())
    return artifact


def propose_calibration(config_path: str) -> dict[str, Any]:
    resolved_config_path = resolve_repo_path(config_path)
    config = read_structured(resolved_config_path)
    human_split_path = resolve_repo_path(
        _require_non_placeholder(config.get("human_split_path"), field_name="human_split_path")
    )
    records = load_jsonl(human_split_path)
    texts = [str(record.get("completion", "")) for record in records]
    minimum_records = int(config.get("minimum_human_records", 2))
    if len(texts) < minimum_records:
        raise M1ConfigError(
            f"calibration proposal requires at least {minimum_records} human records"
        )
    if len({str(record.get("fingerprint") or "") for record in records}) != len(records):
        raise M1ConfigError("calibration human records require unique non-empty fingerprints")
    settings = config.get("interval_settings") or {}
    confidence_level = float(settings.get("confidence_level", 0.95))
    if confidence_level != CALIBRATION_CONFIDENCE_LEVEL:
        raise M1ConfigError("interval_settings.confidence_level must be the frozen 0.95")
    resampling_seeds = list(settings.get("resampling_seeds") or [])
    if not resampling_seeds:
        raise M1ConfigError("interval_settings.resampling_seeds is required")
    configured_methods = settings.get("metric_methods") or {}
    expected_methods = {
        "continuous": CONTINUOUS_INTERVAL_METHOD,
        "repeated_sentence_start_rate": REPETITION_INTERVAL_METHOD,
    }
    if configured_methods != expected_methods:
        raise M1ConfigError("interval_settings.metric_methods does not match frozen methods")

    def interval(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        low_q = (1.0 - confidence_level) / 2.0
        high_q = 1.0 - low_q
        low_index = min(len(ordered) - 1, max(0, int(low_q * (len(ordered) - 1))))
        high_index = min(len(ordered) - 1, max(0, int(high_q * (len(ordered) - 1))))
        return {"low": float(ordered[low_index]), "high": float(ordered[high_index])}

    def summarize(selected_texts: list[str]) -> dict[str, Any]:
        script_rates = [validity.non_target_script_char_rate(text) for text in selected_texts]
        repetition_rates = [validity.repeated_sentence_start_rate([text]) for text in selected_texts]
        repetition_successes = int(sum(repetition_rates))
        individual_bleu = []
        paragraph_lengths: list[int] = []
        sentence_lengths: list[int] = []
        for index, text in enumerate(selected_texts):
            others = [
                other
                for other_index, other in enumerate(selected_texts)
                if other_index != index
            ]
            individual_bleu.append(validity.sentence_bleu(text, others))
            paragraph_lengths.extend(
                len(validity._tokens(part))
                for part in str(text).split("\n\n")
                if part.strip()
            )
            sentence_lengths.extend(
                len(validity._tokens(sentence)) for sentence in validity._sentences(text)
            )
        return {
            "point_estimates": {
                "self_bleu": float(sum(individual_bleu) / len(individual_bleu)),
                "repeated_sentence_start_rate": float(
                    sum(repetition_rates) / len(repetition_rates)
                ),
                "non_target_script_char_rate": float(sum(script_rates) / len(script_rates)),
            },
            "intervals": {
                "self_bleu": interval(individual_bleu),
                "repeated_sentence_start_rate": _wilson_interval(
                    repetition_successes, len(repetition_rates), confidence_level
                ),
                "non_target_script_char_rate": interval(script_rates),
                "paragraph_len_tokens": interval(
                    [float(value) for value in paragraph_lengths]
                ),
                "sentence_len_tokens": interval([float(value) for value in sentence_lengths]),
            },
            "metric_counts": {
                "repeated_sentence_start_rate": {
                    "successes": repetition_successes,
                    "trials": len(repetition_rates),
                }
            },
        }

    full_summary = summarize(texts)

    subset_fraction = float(settings.get("subset_fraction", 0.8))
    subset_size = max(2, min(len(records), int(math.ceil(len(records) * subset_fraction))))
    sensitivity = []
    for seed in resampling_seeds:
        chooser = random.Random(int(seed))
        chosen = sorted(chooser.sample(range(len(records)), k=subset_size))
        subset = [records[index] for index in chosen]
        subset_summary = summarize([texts[index] for index in chosen])
        fingerprints = [str(record.get("fingerprint") or record.get("fineweb_id")) for record in subset]
        subset_payload = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in subset
        ).encode("utf-8")
        sensitivity.append(
            {
                "seed": int(seed),
                "subset_size": subset_size,
                "fingerprints": fingerprints,
                "subset_hash": hashlib.sha256(subset_payload).hexdigest(),
                "subset_point_estimates": subset_summary["point_estimates"],
                "subset_intervals": subset_summary["intervals"],
                "subset_metric_counts": subset_summary["metric_counts"],
            }
        )

    human_split_sha256 = file_sha256(human_split_path)
    manifest_fields: dict[str, Any] = {}
    raw_manifest_path = config.get("human_manifest_path")
    if raw_manifest_path:
        human_manifest_path = resolve_repo_path(
            _require_non_placeholder(raw_manifest_path, field_name="human_manifest_path")
        )
        manifest = read_structured(human_manifest_path)
        if manifest.get("bank_sha256") != human_split_sha256:
            raise M1ConfigError("human calibration bank does not match its manifest hash")
        if int((manifest.get("counts") or {}).get("bank_size", 0)) != len(records):
            raise M1ConfigError("human calibration bank count does not match its manifest")
        if [str(record["fingerprint"]) for record in records] != [
            str(value) for value in manifest.get("fingerprints") or []
        ]:
            raise M1ConfigError("human calibration bank fingerprints do not match its manifest")
        manifest_fields = {
            "human_manifest_path": _display_path(human_manifest_path),
            "human_manifest_sha256": file_sha256(human_manifest_path),
            "human_bank_source": manifest.get("source"),
        }

    subset_hashes = sorted({item["subset_hash"] for item in sensitivity})
    proposal = {
        "analysis_entrypoint": (
            "python -m experiments.m1.analysis calibration-proposal --config "
            + _display_path(resolved_config_path)
        ),
        "artifact_schema": "m1.calibration_proposal.review.v3",
        "config_path": _display_path(resolved_config_path),
        "config_sha256": file_sha256(resolved_config_path),
        "human_split_path": _display_path(human_split_path),
        "human_split_sha256": human_split_sha256,
        "point_estimates": full_summary["point_estimates"],
        "intervals": full_summary["intervals"],
        "metric_counts": full_summary["metric_counts"],
        "interval_methods": {
            "self_bleu": {"method": CONTINUOUS_INTERVAL_METHOD},
            "repeated_sentence_start_rate": {
                "method": REPETITION_INTERVAL_METHOD,
                "z": CALIBRATION_Z_95,
            },
            "non_target_script_char_rate": {"method": CONTINUOUS_INTERVAL_METHOD},
            "paragraph_len_tokens": {"method": CONTINUOUS_INTERVAL_METHOD},
            "sentence_len_tokens": {"method": CONTINUOUS_INTERVAL_METHOD},
        },
        "confidence_level": confidence_level,
        "sample_count": len(records),
        "split_hashes": load_fixed_split_hashes(),
        "resampling_seeds": [int(seed) for seed in resampling_seeds],
        "sensitivity": sensitivity,
        "sensitivity_summary": {
            "subset_fraction": subset_fraction,
            "computed_subset_size": subset_size,
            "unique_subset_count": len(subset_hashes),
            "unique_subset_hashes": subset_hashes,
        },
        "provenance_note": str(config.get("provenance_note") or ""),
        "review_limitations": list(config.get("review_limitations") or []),
    }
    proposal.update(manifest_fields)
    output_path = resolve_repo_path(_require_non_placeholder(config.get("output_path"), field_name="output_path"))
    write_json(output_path, proposal)
    proposal["output_path"] = str(output_path.resolve())
    return proposal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 local analysis helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    baseline = sub.add_parser("baseline-stats")
    baseline.add_argument("--config", required=True)
    freeze = sub.add_parser("freeze-sampler")
    freeze.add_argument("--config", required=True)
    calibrate = sub.add_parser("calibration-proposal")
    calibrate.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "baseline-stats":
        result = build_baseline_stats(args.config)
    elif args.cmd == "freeze-sampler":
        result = freeze_sampler(args.config)
    else:
        result = propose_calibration(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
