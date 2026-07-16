import json
from pathlib import Path

from experiments.m1.tier1_batch import run_batch


def _sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_tier1_batch_maps_remote_samples_and_binds_provenance(tmp_path):
    materialized = tmp_path / "materialized"
    sample = materialized / "samples" / "checkpoint-seed-11" / "default" / "sampling-seed-101.jsonl"
    sample.parent.mkdir(parents=True)
    sample.write_text('{"generated_completion":"one"}\n{"generated_completion":"two"}\n')
    source_index = tmp_path / "source-index.json"
    source_index.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "checkpoint_seed": 11,
                        "sampler_id": "default",
                        "sampling_seed": 101,
                        "samples_path": "/volume/runs/run-1/samples/checkpoint-seed-11/default/sampling-seed-101.jsonl",
                    }
                ]
            }
        )
    )
    bank = tmp_path / "bank.jsonl"
    bank.write_text('{"completion":"human"}\n')
    manifest = tmp_path / "bank.manifest.json"
    manifest.write_text("{}")
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    reports = tmp_path / "reports"
    output_index = tmp_path / "output-index.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "batch_id": "test",
                "sampler_run_id": "run-1",
                "source_index_path": str(source_index),
                "source_index_sha256": _sha(source_index),
                "materialized_root": str(materialized),
                "human_bank_path": str(bank),
                "human_bank_sha256": _sha(bank),
                "human_manifest_path": str(manifest),
                "human_manifest_sha256": _sha(manifest),
                "calibration_path": str(calibration),
                "calibration_sha256": _sha(calibration),
                "baseline_path": str(baseline),
                "baseline_sha256": _sha(baseline),
                "sampler_ids": ["default"],
                "expected_entry_count": 1,
                "quality_judge_mode": "neutral",
                "reports_root": str(reports),
                "output_index_path": str(output_index),
            }
        )
    )

    def fake_evaluate(target, report_path, *, embedder):
        del target, embedder
        report = {
            "human_reference_bank_id": _sha(manifest),
            "calibration_sha256": _sha(calibration),
            "baseline_sha256": _sha(baseline),
        }
        Path(report_path).write_text(json.dumps(report))
        return report

    result = run_batch(config, evaluate_fn=fake_evaluate, embedder=object())
    assert result["entry_count"] == 1
    assert result["human_reference_bank_id"] == _sha(manifest)
    written = json.loads(output_index.read_text())
    assert written["entries"][0]["samples_sha256"] == _sha(sample)
    assert written["entries"][0]["report_sha256"]
