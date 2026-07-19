from __future__ import annotations

import pytest

from backend.volume_paths import (
    checkpoint_volume_path,
    checkpoint_volume_uri,
    missing_run_artifact_metadata,
    run_artifact_metadata,
    run_worker_log_path,
)


def test_checkpoint_volume_path_preserves_mount_alias_without_resolving():
    assert checkpoint_volume_path(
        "modal-volume://humanwrite-checkpoints/data/pilot/train.jsonl"
    ).as_posix() == "/checkpoints/data/pilot/train.jsonl"


def test_checkpoint_volume_uri_round_trips_mount_alias():
    assert checkpoint_volume_uri("/checkpoints/data/pilot/train.jsonl") == (
        "modal-volume://humanwrite-checkpoints/data/pilot/train.jsonl"
    )


def test_run_worker_log_path_stays_inside_checkpoint_mount(tmp_path):
    path = run_worker_log_path("dftr-example", mount_path=str(tmp_path))
    assert path == tmp_path / "runs" / "dftr-example" / "worker.log"
    assert checkpoint_volume_uri(path, mount_path=str(tmp_path)) == (
        "modal-volume://humanwrite-checkpoints/runs/dftr-example/worker.log"
    )


def test_run_artifact_metadata_surfaces_scale_panel_handoff(tmp_path):
    artifact_dir = tmp_path / "runs" / "example"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "run_manifest.json").write_text(
        (
            "{"
            "\"panel_bundle_path\":\"/checkpoints/data/scale/panel_bundle.json\","
            "\"panel_bundle_sha256\":\"a\","
            "\"prompt_sources_path\":\"/checkpoints/data/scale/prompt_sources.jsonl\","
            "\"prompt_sources_sha256\":\"b\""
            "}\n"
        ),
        encoding="utf-8",
    )
    metadata = run_artifact_metadata(artifact_dir, mount_path=str(tmp_path))
    assert metadata["metrics_ptr"] == "modal-volume://humanwrite-checkpoints/runs/example/run_manifest.json"
    assert metadata["panel_bundle_path"] == "modal-volume://humanwrite-checkpoints/data/scale/panel_bundle.json"
    assert metadata["prompt_sources_path"] == "modal-volume://humanwrite-checkpoints/data/scale/prompt_sources.jsonl"
    assert metadata["panel_bundle_sha256"] == "a"
    assert metadata["prompt_sources_sha256"] == "b"


def test_run_artifact_metadata_surfaces_scale_train_prefix_handoff(tmp_path):
    artifact_dir = tmp_path / "runs" / "example-train"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "run_manifest.json").write_text(
        (
            "{"
            "\"train_prefix_bundle_path\":\"/checkpoints/data/scale/train_prefix_bundle.json\","
            "\"train_prefix_bundle_sha256\":\"a\","
            "\"clean_train_4096_path\":\"/checkpoints/data/scale/clean-train-4096.jsonl\","
            "\"clean_train_4096_sha256\":\"b\","
            "\"clean_train_16384_path\":\"/checkpoints/data/scale/clean-train-16384.jsonl\","
            "\"clean_train_16384_sha256\":\"c\""
            "}\n"
        ),
        encoding="utf-8",
    )
    metadata = run_artifact_metadata(artifact_dir, mount_path=str(tmp_path))
    assert metadata["metrics_ptr"] == "modal-volume://humanwrite-checkpoints/runs/example-train/run_manifest.json"
    assert metadata["train_prefix_bundle_path"] == "modal-volume://humanwrite-checkpoints/data/scale/train_prefix_bundle.json"
    assert metadata["clean_train_4096_path"] == "modal-volume://humanwrite-checkpoints/data/scale/clean-train-4096.jsonl"
    assert metadata["clean_train_16384_path"] == "modal-volume://humanwrite-checkpoints/data/scale/clean-train-16384.jsonl"
    assert metadata["train_prefix_bundle_sha256"] == "a"
    assert metadata["clean_train_4096_sha256"] == "b"
    assert metadata["clean_train_16384_sha256"] == "c"


def test_missing_run_artifact_metadata_backfills_terminal_snapshot(tmp_path):
    artifact_dir = tmp_path / "runs" / "dftr-example"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "run_manifest.json").write_text(
        (
            "{"
            "\"panel_bundle_path\":\"/checkpoints/data/scale/panel_bundle.json\","
            "\"panel_bundle_sha256\":\"a\","
            "\"prompt_sources_path\":\"/checkpoints/data/scale/prompt_sources.jsonl\","
            "\"prompt_sources_sha256\":\"b\""
            "}\n"
        ),
        encoding="utf-8",
    )
    state = {
        "run_id": "dftr-example",
        "status": "completed",
        "artifact_dir": "/__modal/volumes/opaque/runs/dftr-example",
        "panel_bundle_path": "",
        "prompt_sources_path": None,
    }
    missing = missing_run_artifact_metadata(state, mount_path=str(tmp_path))
    assert missing["metrics_ptr"] == (
        "modal-volume://humanwrite-checkpoints/runs/dftr-example/run_manifest.json"
    )
    assert missing["panel_bundle_path"] == (
        "modal-volume://humanwrite-checkpoints/data/scale/panel_bundle.json"
    )
    assert missing["panel_bundle_sha256"] == "a"
    assert missing["prompt_sources_path"] == (
        "modal-volume://humanwrite-checkpoints/data/scale/prompt_sources.jsonl"
    )
    assert missing["prompt_sources_sha256"] == "b"
    assert missing["run_manifest_sha256"]


@pytest.mark.parametrize(
    "uri",
    [
        "file:///tmp/train.jsonl",
        "modal-volume://humanwrite-checkpoints/../state/events.jsonl",
        "modal-volume://humanwrite-checkpoints/data/../../state/events.jsonl",
        "modal-volume://humanwrite-checkpoints/",
    ],
)
def test_checkpoint_volume_path_rejects_outside_or_traversal(uri):
    with pytest.raises(ValueError):
        checkpoint_volume_path(uri)
