import json
from pathlib import Path

import pytest

from experiments.m1.analysis import build_baseline_stats, freeze_sampler
from experiments.m1.contracts import M1ConfigError


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def report(score, *, calibration="calibration-sha", baseline="bootstrap-placeholder"):
    return {
        "S": score,
        "semantic_mmd": score + 0.1,
        "semantic_mmd_delta_vs_human_floor": score + 0.05,
        "lexical_l2": score + 0.2,
        "structural_dist": score + 0.3,
        "gate_outline_fact_recall": True,
        "gate_unsupported_claim_rate": True,
        "gate_language_integrity": True,
        "gate_no_collapse": True,
        "human_reference_bank_id": "bank-id",
        "calibration_sha256": calibration,
        "baseline_sha256": baseline,
    }


def test_baseline_bootstraps_from_preregistered_default_without_freeze_selection(tmp_path):
    entries = []
    for index, score in enumerate((0.1, 0.3, 0.5)):
        report_path = tmp_path / f"report-{index}.json"
        sample_path = tmp_path / f"samples-{index}.jsonl"
        write_json(report_path, report(score))
        write_jsonl(
            sample_path,
            [
                {
                    "generated_completion": "A supported fact appears in this sentence.",
                    "outline": [{"supported_facts": ["A supported fact"]}],
                }
            ],
        )
        entries.append(
            {
                "sampler_id": "default_t1.0_p1.0",
                "report_path": str(report_path),
                "samples_path": str(sample_path),
            }
        )
    index_path = tmp_path / "index.json"
    write_json(index_path, {"entries": entries})
    output = tmp_path / "baseline-proposal.json"
    config = tmp_path / "baseline-config.json"
    write_json(
        config,
        {
            "output_path": str(output),
            "report_index": str(index_path),
            "baseline_sampler_id": "default_t1.0_p1.0",
            "expected_calibration_sha256": "calibration-sha",
            "expected_human_reference_bank_id": "bank-id",
        },
    )
    result = build_baseline_stats(str(config))
    assert result["artifact_schema"] == "m1.baseline_stats.review.v1"
    assert result["baseline_sampler_id"] == "default_t1.0_p1.0"
    assert result["sample_count"] == 3
    assert "selection_artifact" not in json.loads(config.read_text())


def test_baseline_bootstrap_rejects_nondefault_sampler(tmp_path):
    config = tmp_path / "baseline-config.json"
    write_json(
        config,
        {
            "output_path": str(tmp_path / "out.json"),
            "report_index": "unused",
            "baseline_sampler_id": "selected_after_results",
            "expected_calibration_sha256": "calibration",
            "expected_human_reference_bank_id": "bank",
        },
    )
    with pytest.raises((M1ConfigError, FileNotFoundError)):
        build_baseline_stats(str(config))


def test_sampler_freeze_requires_exact_frozen_input_hashes(tmp_path):
    report_path = tmp_path / "report.json"
    write_json(report_path, report(0.1, baseline="frozen-baseline"))
    index_path = tmp_path / "index.json"
    write_json(
        index_path,
        {
            "entries": [
                {
                    "sampler_id": "default_t1.0_p1.0",
                    "report_path": str(report_path),
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "kl_drift": 0.0,
                }
            ]
        },
    )
    config_path = tmp_path / "freeze.json"
    config = {
        "default_sampler_id": "default_t1.0_p1.0",
        "output_path": str(tmp_path / "freeze-output.json"),
        "report_index": str(index_path),
        "expected_baseline_sha256": "wrong-baseline",
        "expected_calibration_sha256": "calibration-sha",
        "expected_human_reference_bank_id": "bank-id",
    }
    write_json(config_path, config)
    with pytest.raises(M1ConfigError, match="operator-frozen baseline"):
        freeze_sampler(str(config_path))
    config["expected_baseline_sha256"] = "frozen-baseline"
    write_json(config_path, config)
    assert freeze_sampler(str(config_path))["selected_sampler_id"] == "default_t1.0_p1.0"
