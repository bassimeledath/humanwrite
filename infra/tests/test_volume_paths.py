from __future__ import annotations

import pytest

from backend.volume_paths import checkpoint_volume_path


def test_checkpoint_volume_path_preserves_mount_alias_without_resolving():
    assert checkpoint_volume_path(
        "modal-volume://humanwrite-checkpoints/data/pilot/train.jsonl"
    ).as_posix() == "/checkpoints/data/pilot/train.jsonl"


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
