from __future__ import annotations

import pytest

from backend.volume_paths import checkpoint_volume_path, checkpoint_volume_uri, run_artifact_metadata


def test_checkpoint_volume_path_preserves_mount_alias_without_resolving():
    assert checkpoint_volume_path(
        "modal-volume://humanwrite-checkpoints/data/pilot/train.jsonl"
    ).as_posix() == "/checkpoints/data/pilot/train.jsonl"


def test_checkpoint_volume_uri_round_trips_mount_alias():
    assert checkpoint_volume_uri("/checkpoints/data/pilot/train.jsonl") == (
        "modal-volume://humanwrite-checkpoints/data/pilot/train.jsonl"
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
