import hashlib
import json

from experiments.m1.analysis import propose_calibration


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_calibration_proposal_is_complete_and_byte_reproducible(tmp_path):
    rows = [
        {
            "completion": f"Human document {index}. A distinct sentence follows here.",
            "fingerprint": f"fingerprint-{index}",
        }
        for index in range(4)
    ]
    human_path = tmp_path / "humans.jsonl"
    output_path = tmp_path / "proposal.json"
    _write_jsonl(human_path, rows)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "human_split_path": str(human_path),
                "minimum_human_records": 4,
                "interval_settings": {
                    "confidence_level": 0.95,
                    "resampling_seeds": [404, 505, 606],
                    "subset_fraction": 0.75,
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
    assert first["artifact_schema"] == "m1.calibration_proposal.review.v1"
    assert first["sample_count"] == 4
    assert first["human_split_sha256"] == hashlib.sha256(human_path.read_bytes()).hexdigest()
    assert first["review_limitations"] == ["test limitation"]
    assert all(item["subset_hash"] for item in first["sensitivity"])
