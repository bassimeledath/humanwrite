import hashlib
import json

import pytest

from experiments.m1.analysis import propose_calibration
from experiments.m1.contracts import M1ConfigError


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_calibration_proposal_is_complete_and_byte_reproducible(tmp_path):
    rows = []
    for index in range(32):
        completion = f"Human document {index}. A distinct sentence follows here."
        if index == 0:
            completion = "Again the first sentence. Again the second sentence. Again the third sentence."
        rows.append({"completion": completion, "fingerprint": f"fingerprint-{index}"})
    human_path = tmp_path / "humans.jsonl"
    output_path = tmp_path / "proposal.json"
    _write_jsonl(human_path, rows)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "human_split_path": str(human_path),
                "minimum_human_records": 32,
                "interval_settings": {
                    "confidence_level": 0.95,
                    "resampling_seeds": [404, 505, 606],
                    "subset_fraction": 0.75,
                    "metric_methods": {
                        "continuous": "deterministic central order-statistic interval",
                        "repeated_sentence_start_rate": "Wilson score interval for binomial proportion",
                    },
                },
                "output_path": str(output_path),
                "provenance_note": "test human bank",
                "review_limitations": ["test limitation"],
            }
        ),
        encoding="utf-8",
    )

    first = propose_calibration(str(config_path))
    first_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    second = propose_calibration(str(config_path))
    second_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

    assert first_hash == second_hash
    assert first == second
    assert first["artifact_schema"] == "m1.calibration_proposal.review.v2"
    assert first["sample_count"] == 32
    assert first["human_split_sha256"] == hashlib.sha256(human_path.read_bytes()).hexdigest()
    assert first["review_limitations"] == ["test limitation"]
    assert all(item["subset_hash"] for item in first["sensitivity"])
    assert first["point_estimates"]["repeated_sentence_start_rate"] == 1 / 32
    assert first["metric_counts"]["repeated_sentence_start_rate"] == {
        "successes": 1,
        "trials": 32,
    }
    repetition_interval = first["intervals"]["repeated_sentence_start_rate"]
    assert repetition_interval["low"] == pytest.approx(0.005537860164003122)
    assert repetition_interval["high"] == pytest.approx(0.15744263820012558)
    assert first["interval_methods"]["repeated_sentence_start_rate"]["method"] == (
        "Wilson score interval for binomial proportion"
    )


def test_calibration_proposal_requires_explicit_metric_methods(tmp_path):
    human_path = tmp_path / "humans.jsonl"
    _write_jsonl(
        human_path,
        [
            {"completion": f"Human document {index}.", "fingerprint": f"fingerprint-{index}"}
            for index in range(4)
        ],
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "human_split_path": str(human_path),
                "minimum_human_records": 4,
                "interval_settings": {
                    "confidence_level": 0.95,
                    "resampling_seeds": [404],
                    "subset_fraction": 0.75,
                },
                "output_path": str(tmp_path / "proposal.json"),
            }
        )
    )
    with pytest.raises(M1ConfigError, match="metric_methods"):
        propose_calibration(str(config_path))
