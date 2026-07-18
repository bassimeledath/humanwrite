"""Lexically safe Modal volume path and URI conversion helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


VOLUME_PREFIX = "modal-volume://humanwrite-checkpoints/"


def checkpoint_volume_path(uri: str, mount_path: str = "/checkpoints") -> Path:
    if not uri.startswith(VOLUME_PREFIX):
        raise ValueError("URI is outside the checkpoint volume")
    relative = PurePosixPath(uri[len(VOLUME_PREFIX):])
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("unsafe volume URI")
    return Path(mount_path).joinpath(*relative.parts)


def checkpoint_volume_uri(path: str | Path, mount_path: str = "/checkpoints") -> str:
    resolved_mount = Path(mount_path)
    candidate = Path(path)
    try:
        relative = candidate.relative_to(resolved_mount)
    except ValueError as exc:
        raise ValueError("path is outside the checkpoint volume") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("unsafe checkpoint path")
    return VOLUME_PREFIX + PurePosixPath(*relative.parts).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maybe_uri(value: Any, mount_path: str) -> Any:
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    for candidate_mount in (mount_path, "/checkpoints"):
        try:
            return checkpoint_volume_uri(value, candidate_mount)
        except ValueError:
            continue
    return value


def run_artifact_metadata(
    artifact_dir: str | Path,
    *,
    mount_path: str = "/checkpoints",
) -> dict[str, Any]:
    """Return a small sanctioned handoff payload for a completed run artifact."""

    artifact_root = Path(artifact_dir)
    manifest_path = artifact_root / "run_manifest.json"
    if not manifest_path.is_file():
        return {}
    metadata: dict[str, Any] = {
        "metrics_ptr": _maybe_uri(str(manifest_path), mount_path),
        "run_manifest_sha256": _sha256(manifest_path),
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return metadata
    for key in (
        "panel_bundle_path",
        "panel_bundle_sha256",
        "prompt_sources_path",
        "prompt_sources_sha256",
    ):
        value = manifest.get(key)
        if value is not None:
            metadata[key] = _maybe_uri(value, mount_path)
    return metadata
