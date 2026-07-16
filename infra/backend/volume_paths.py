"""Lexically safe Modal volume URI conversion without resolving mount symlinks."""
from __future__ import annotations

from pathlib import Path, PurePosixPath


VOLUME_PREFIX = "modal-volume://humanwrite-checkpoints/"


def checkpoint_volume_path(uri: str, mount_path: str = "/checkpoints") -> Path:
    if not uri.startswith(VOLUME_PREFIX):
        raise ValueError("URI is outside the checkpoint volume")
    relative = PurePosixPath(uri[len(VOLUME_PREFIX):])
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("unsafe volume URI")
    return Path(mount_path).joinpath(*relative.parts)
