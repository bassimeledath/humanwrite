from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "experiments"
FIXED_SPLIT_HASHES_PATH = ROOT / "data" / "artifacts" / "m0" / "split_hashes.json"
UNRESOLVED_REVISION_PREFIX = "__M1_RESOLVE_"
PLACEHOLDER_RE = re.compile(r"^__M1_[A-Z0-9_]+__$")
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


class M1ConfigError(ValueError):
    pass


def canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_structured(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(raw)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML configs") from exc
        value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise M1ConfigError(f"expected a mapping in {path}")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    source = Path(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise M1ConfigError(f"invalid JSONL at {source}:{line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise M1ConfigError(f"JSONL record at {source}:{line_number} is not an object")
        records.append(record)
    return records


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def resolve_repo_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    anchor = Path(base) if base is not None else ROOT
    return (anchor / path).resolve()


def load_fixed_split_hashes() -> dict[str, str]:
    value = json.loads(FIXED_SPLIT_HASHES_PATH.read_text(encoding="utf-8"))
    train_hash = str(value.get("train", ""))
    dev_hash = str(value.get("dev", ""))
    if not train_hash or not dev_hash:
        raise M1ConfigError("fixed M0 split hashes are missing")
    return {"train": train_hash, "dev": dev_hash}


def ensure_fixed_hash(value: Any, *, expected: str, field_name: str) -> None:
    if str(value) != expected:
        raise M1ConfigError(f"{field_name} mismatch: expected {expected} but found {value}")


def is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.startswith(UNRESOLVED_REVISION_PREFIX) or bool(PLACEHOLDER_RE.match(text))


def require_resolved_revision(config: dict[str, Any], *, context: str) -> str:
    revision = (config.get("model") or {}).get("revision")
    if is_placeholder(revision):
        raise M1ConfigError(
            f"{context} requires model.revision to be replaced with a resolved immutable revision"
        )
    return str(revision)


def load_resolved_model_manifest() -> dict[str, Any]:
    manifest_path = os.environ.get("DFTR_RESOLVED_MODEL_MANIFEST", "").strip()
    if not manifest_path:
        raise M1ConfigError("DFTR_RESOLVED_MODEL_MANIFEST is required for the resolve_revision step")
    path = Path(manifest_path)
    if not path.is_file():
        raise M1ConfigError(f"resolved model manifest not found: {path}")
    value = read_structured(path)
    required = {"base_model", "resolved_revision", "snapshot_path"}
    missing = sorted(key for key in required if not value.get(key))
    if missing:
        raise M1ConfigError(
            "resolved model manifest is missing fields: " + ", ".join(missing)
        )
    return value


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("unable to resolve git SHA for provenance") from exc


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_text_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def count_output_tokens(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        text = str(
            record.get("output")
            or record.get("generated_completion")
            or record.get("completion")
            or ""
        )
        total += count_text_tokens(text)
    return total


def build_run_paths(config: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    comparison_id = str((config.get("run") or {}).get("comparison_id", "unknown-comparison"))
    output_dir = DEFAULT_OUTPUT_ROOT / comparison_id / run_id
    checkpoint_dir = Path(os.environ.get("DFTR_CHECKPOINT_DIR", output_dir / "checkpoint"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoint_dir

