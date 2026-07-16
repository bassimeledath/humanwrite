import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from harness import cli


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_checkpoint_hash_is_stable_path_aware_and_symlink_safe(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").mkdir()
    (second / "b").mkdir()
    (first / "a" / "same.bin").write_bytes(b"content")
    (second / "b" / "same.bin").write_bytes(b"content")
    assert cli._ckpt_hash(first) == cli._ckpt_hash(first)
    assert cli._ckpt_hash(first) != cli._ckpt_hash(second)
    outside = tmp_path / "outside"
    outside.write_text("secret one")
    link = first / "link"
    link.symlink_to(outside)
    before = cli._ckpt_hash(first)
    outside.write_text("secret two")
    assert cli._ckpt_hash(first) == before


def test_representation_guard_rejects_embedder_and_discriminator_collisions(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text(
        yaml.safe_dump({"objective": {"train_embedder_id": cli.DEV_EMBEDDER_ID}})
    )
    with pytest.raises(ValueError, match="same embedder"):
        cli._guard_representation(checkpoint)
    (checkpoint / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "objective": {"train_discriminator_id": "disc-v1"},
                "evaluation": {"eval_discriminator_id": "disc-v1"},
            }
        )
    )
    with pytest.raises(ValueError, match="GAIL"):
        cli._guard_representation(checkpoint)


def test_representation_guard_requires_metadata(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        cli._guard_representation(tmp_path)


def test_calibrate_reads_canonical_schema_without_changing_repo_file(tmp_path, monkeypatch):
    source = tmp_path / "human.jsonl"
    destination = tmp_path / "calibration.json"
    write_jsonl(
        source,
        [
            {"fineweb_id": "1", "split": "dev", "completion": "First sentence.\n\nSecond paragraph."},
            {"fineweb_id": "2", "split": "dev", "completion": "Another human document. It is useful."},
            {
                "fineweb_id": "3",
                "split": "dev",
                "completion": "Again one sentence. Again two sentences. Again three sentences.",
            },
        ],
    )
    monkeypatch.setattr(cli, "CALIBRATION_PATH", destination)
    result = cli.calibrate(source)
    assert json.loads(destination.read_text()) == result
    assert result["self_bleu"]["low"] is not None
    assert result["paragraph_len_tokens"]["p50"] > 0
    assert result["repeated_sentence_start_rate"]["high"] > 0


def test_evaluate_offline_with_inline_references(tmp_path, monkeypatch):
    samples = tmp_path / "samples.jsonl"
    records = []
    for index in range(4):
        fact = f"planet {index} has moon {index}"
        records.append(
            {
                "fineweb_id": str(index),
                "split": "dev",
                "user_prompt": "Write a fact.",
                "outline": [{"section": "Fact", "supported_facts": [fact]}],
                "generated_completion": fact + ". More context follows here.",
                "reference_completion": f"A human reference number {index}. It has context.",
            }
        )
    write_jsonl(samples, records)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "self_bleu": {"low": 0.0, "high": 1.0},
                "repeated_sentence_start_rate": {"low": 0.0, "high": 1.0},
                "non_target_script_char_rate": {"low": 0.0, "high": 0.1},
            }
        )
    )
    monkeypatch.setattr(cli, "CALIBRATION_PATH", calibration)
    monkeypatch.setattr(cli, "BASELINE_PATH", tmp_path / "absent.json")

    def embed(texts):
        return np.array([[len(text), text.count(".")] for text in texts], dtype=float)

    def judge(**kwargs):
        return "A" if "planet" in kwargs["candidate_a"] else "B"

    def probe(texts):
        return [1.0 if "planet" in text else 0.0 for text in texts]

    report_path = tmp_path / "report.json"
    report = cli.evaluate(samples, report_path, embedder=embed, judge=judge, probe=probe)
    assert report.quality_pref_winrate == 1.0
    assert report.jmq == 2.0
    assert report.authorship_auc == 1.0
    assert report.gate_language_integrity
    assert json.loads(report_path.read_text())["checkpoint_id"] == report.checkpoint_id
    assert json.dumps(asdict(report), allow_nan=False)


def test_evaluate_builds_fresh_probe_when_none_is_injected(tmp_path, monkeypatch):
    samples = tmp_path / "samples.jsonl"
    write_jsonl(
        samples,
        [
            {
                "generated_completion": f"generated sample {index} with enough characters",
                "reference_completion": f"human reference {index} with distinct prose",
                "user_prompt": "Write.",
            }
            for index in range(4)
        ],
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "self_bleu": {"low": 0.0, "high": 1.0},
                "repeated_sentence_start_rate": {"low": 0.0, "high": 1.0},
                "non_target_script_char_rate": {"low": 0.0, "high": 1.0},
            }
        )
    )
    monkeypatch.setattr(cli, "CALIBRATION_PATH", calibration)
    monkeypatch.setattr(cli, "BASELINE_PATH", tmp_path / "missing.json")
    calls = []

    def fresh(generated, humans):
        calls.append((generated, humans))
        return 0.73, 0.61, 0.82

    monkeypatch.setattr(cli.quality, "fresh_authorship_auc", fresh)
    report = cli.evaluate(
        samples,
        embedder=lambda texts: np.array([[len(text), index] for index, text in enumerate(texts)]),
    )
    assert calls
    assert report.authorship_auc == 0.73
    assert "out-of-fold" in " ".join(report.notes)


def test_checkpoint_generation_uses_frozen_canonical_prompt_bank(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text(
        yaml.safe_dump({"objective": {"train_embedder_id": "independent/reward"}})
    )
    prompt_bank = tmp_path / "prompts.jsonl"
    write_jsonl(
        prompt_bank,
        [
            {
                "fineweb_id": str(index),
                "user_prompt": f"Canonical prompt {index}",
                "outline": [],
                "completion": f"Human reference document {index} has natural prose.",
            }
            for index in range(4)
        ],
    )
    sampler_path = tmp_path / "deployment_sampler.json"
    sampler = {
        "frozen": True,
        "prompt_bank": str(prompt_bank),
        "prompt_format": "USER: {user_prompt}\nASSISTANT:",
        "seed": 7,
        "batch_size": 2,
        "max_input_tokens": 128,
        "max_new_tokens": 64,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
    }
    sampler_path.write_text(json.dumps(sampler))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "self_bleu": {"low": 0.0, "high": 1.0},
                "repeated_sentence_start_rate": {"low": 0.0, "high": 1.0},
                "non_target_script_char_rate": {"low": 0.0, "high": 1.0},
            }
        )
    )
    monkeypatch.setattr(cli, "DEPLOYMENT_SAMPLER_PATH", sampler_path)
    monkeypatch.setattr(cli, "CALIBRATION_PATH", calibration)
    monkeypatch.setattr(cli, "BASELINE_PATH", tmp_path / "missing-baseline.json")
    captured = {}

    def generator(**kwargs):
        captured.update(kwargs)
        return [f"Generated document {index} contains enough prose." for index in range(4)]

    report = cli.evaluate(
        checkpoint,
        embedder=lambda texts: np.array([[len(text), text.count("e")] for text in texts]),
        probe=lambda texts: [float("Generated" in text) for text in texts],
        generator=generator,
    )
    assert report.checkpoint_id == cli._ckpt_hash(checkpoint)
    assert [record["user_prompt"] for record in captured["records"]] == [
        f"Canonical prompt {index}" for index in range(4)
    ]
    assert captured["sampler"]["seed"] == 7


def test_checkpoint_generation_fails_closed_for_unfrozen_sampler(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"objective": {"train_embedder_id": "independent/reward"}})
    )
    sampler = tmp_path / "deployment_sampler.json"
    sampler.write_text(json.dumps({"frozen": False, "prompt_bank": None}))
    monkeypatch.setattr(cli, "DEPLOYMENT_SAMPLER_PATH", sampler)
    with pytest.raises(ValueError, match="not frozen"):
        cli.evaluate(checkpoint, embedder=lambda texts: np.ones((len(texts), 2)), generator=lambda **_: [])


def test_checkpoint_generation_fails_closed_for_null_frozen_sampler(tmp_path, monkeypatch):
    sampler = tmp_path / "deployment_sampler.json"
    sampler.write_text(
        json.dumps(
            {
                "frozen": True,
                "prompt_bank": None,
                "prompt_format": None,
                "seed": None,
                "batch_size": None,
                "max_input_tokens": None,
                "max_new_tokens": None,
                "do_sample": None,
                "temperature": None,
                "top_p": None,
            }
        )
    )
    monkeypatch.setattr(cli, "DEPLOYMENT_SAMPLER_PATH", sampler)
    with pytest.raises(ValueError, match="null/unset"):
        cli._load_deployment_sampler()


def test_checkpoint_prefers_existing_samples_over_generator(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"objective": {"train_embedder_id": "independent/reward"}})
    )
    write_jsonl(
        checkpoint / "samples.jsonl",
        [
            {
                "generated_completion": f"Generated existing sample {index}.",
                "reference_completion": f"Human existing reference {index}.",
            }
            for index in range(4)
        ],
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "self_bleu": {"low": 0.0, "high": 1.0},
                "repeated_sentence_start_rate": {"low": 0.0, "high": 1.0},
                "non_target_script_char_rate": {"low": 0.0, "high": 1.0},
            }
        )
    )
    monkeypatch.setattr(cli, "CALIBRATION_PATH", calibration)
    monkeypatch.setattr(cli, "BASELINE_PATH", tmp_path / "missing.json")
    cli.evaluate(
        checkpoint,
        embedder=lambda texts: np.array([[len(text), index] for index, text in enumerate(texts)]),
        probe=lambda texts: [float("Generated" in text) for text in texts],
        generator=lambda **_: pytest.fail("existing samples must bypass generation"),
    )


def test_evaluate_requires_independent_human_reference(tmp_path):
    samples = tmp_path / "samples.jsonl"
    write_jsonl(samples, [{"completion": "one"}, {"completion": "two"}])
    with pytest.raises(ValueError, match="human references"):
        cli.evaluate(samples, embedder=lambda texts: np.ones((len(texts), 2)))


def test_environment_judge_posts_aggregate_request_with_bearer_token(monkeypatch):
    monkeypatch.setenv("HARNESS_JUDGE_URL", "https://judge.invalid/compare")
    monkeypatch.setenv("HARNESS_JUDGE_TOKEN", "judge-secret")
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"winner": "A"}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(cli.requests, "post", post)
    result = cli._environment_judge()(prompt="p", candidate_a="a", candidate_b="b")
    assert result == {"winner": "A"}
    assert captured["json"] == {"prompt": "p", "candidate_a": "a", "candidate_b": "b"}
    assert captured["headers"] == {"Authorization": "Bearer judge-secret"}


def test_environment_judge_errors_do_not_expose_token(monkeypatch):
    monkeypatch.setenv("HARNESS_JUDGE_URL", "https://judge.invalid/compare")
    monkeypatch.setenv("HARNESS_JUDGE_TOKEN", "judge-secret")
    monkeypatch.setattr(
        cli.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(cli.requests.RequestException("network")),
    )
    with pytest.raises(RuntimeError) as caught:
        cli._environment_judge()(prompt="p", candidate_a="a", candidate_b="b")
    assert "judge-secret" not in str(caught.value)


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "window_id": "w1",
            "quota_remaining": 4,
            "primary": {"semantic_mmd": 0.1, "semantic_mmd_delta_vs_floor": 0.01, "S": 0.2},
            "authorship_auc": 0.5,
            "authorship_auc_ci": [0.45, 0.55],
            "gates": {"valid": True},
            "verdict": "confirm",
            "aggregate_only": True,
        }


def test_sealed_submit_matches_contract_and_uses_bearer_token(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"arm": "A", "comparison_id": "registered-comparison"},
                "objective": {"train_embedder_id": "reward/embedder"},
                "artifact_uri": "s3://bucket/checkpoint",
            }
        )
    )
    monkeypatch.setenv("SEALED_EVAL_URL", "https://sealed.invalid")
    monkeypatch.setenv("SEALED_EVAL_TOKEN", "top-secret-token")
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(cli.requests, "post", post)
    result = cli.sealed_submit(checkpoint)
    assert result["aggregate_only"] is True
    assert captured["url"] == "https://sealed.invalid/submit"
    assert captured["headers"] == {"Authorization": "Bearer top-secret-token"}
    assert captured["json"] == {
        "checkpoint_hash": cli._ckpt_hash(checkpoint),
        "artifact_uri": "s3://bucket/checkpoint",
        "arm": "A",
        "train_embedder_id": "reward/embedder",
        "comparison_id": "registered-comparison",
    }


@pytest.mark.parametrize(
    ("status", "message"), [(429, "quota"), (409, "independence")]
)
def test_sealed_submit_handles_contract_errors(tmp_path, monkeypatch, status, message):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"arm": "SFT", "comparison_id": "c", "train_embedder_id": "none"})
    )
    monkeypatch.setenv("SEALED_EVAL_URL", "https://sealed.invalid/submit")
    monkeypatch.setenv("SEALED_EVAL_TOKEN", "secret")
    response = FakeResponse()
    response.status_code = status
    monkeypatch.setattr(cli.requests, "post", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match=message) as caught:
        cli.sealed_submit(checkpoint)
    assert "secret" not in str(caught.value)


def test_main_returns_two_without_leaking_secrets(monkeypatch, capsys):
    monkeypatch.delenv("SEALED_EVAL_URL", raising=False)
    monkeypatch.delenv("SEALED_EVAL_TOKEN", raising=False)
    assert cli.main(["sealed-submit", "/does/not/exist"]) == 2
    assert "harness:" in capsys.readouterr().err
