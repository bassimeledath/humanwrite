"""Replace provider token estimates with exact pinned-tokenizer completion lengths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import modal


APP_NAME = "humanwrite-token-length-normalization"
ROOT = Path("/checkpoints/data/m2-lower-variance-v1")
SOURCE_PATH = ROOT / "clean-train-1024.jsonl"
BRIEF_PATH = ROOT / "train-briefs-1024.jsonl"
BACKUP_PATH = ROOT / "train-briefs-1024.pre-token-normalization.jsonl"
MANIFEST_PATH = ROOT / "token-length-normalization-manifest.json"
MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

source_root = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers==4.57.6", "huggingface-hub>=0.33,<1")
    .add_local_dir(source_root, remote_path="/root/data", copy=True)
)
volume = modal.Volume.from_name("humanwrite-checkpoints")
secret = modal.Secret.from_name("the-other-ones")
app = modal.App(APP_NAME)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.function(
    image=image,
    secrets=[secret],
    volumes={"/checkpoints": volume},
    timeout=15 * 60,
)
def normalize() -> dict:
    from transformers import AutoTokenizer
    from data.lower_variance_briefs import (
        deterministic_empty_outline_ids,
        validate_assembled_brief,
    )

    volume.reload()
    sources = _rows(SOURCE_PATH)
    briefs = _rows(BRIEF_PATH)
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    if len(sources) != len(source_by_id) or len(briefs) != 1024:
        raise RuntimeError("token normalization requires complete unique source and brief corpora")
    empty_ids = deterministic_empty_outline_ids(sources)
    before_sha = _sha(BRIEF_PATH)
    if BACKUP_PATH.exists():
        if _sha(BACKUP_PATH) != before_sha:
            raise RuntimeError("pre-normalization backup already exists with different bytes")
    else:
        shutil.copyfile(BRIEF_PATH, BACKUP_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    old_lengths = [int(row["target_length"]) for row in briefs]
    new_lengths = []
    for row in briefs:
        length = len(tokenizer.encode(str(row["completion"]), add_special_tokens=False))
        if not 1 <= length <= 4096:
            raise RuntimeError(f"exact token length outside contract: {length}")
        row["target_length"] = length
        new_lengths.append(length)
        validate_assembled_brief(
            row,
            source=source_by_id[str(row["fingerprint"])],
            force_empty_outline=str(row["fingerprint"]) in empty_ids,
        )
    temporary = BRIEF_PATH.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in briefs:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(BRIEF_PATH)
    after_sha = _sha(BRIEF_PATH)
    sorted_lengths = sorted(new_lengths)
    manifest = {
        "artifact_schema": "dftr.exact_token_length_normalization.v1",
        "status": "completed",
        "rows": len(briefs),
        "changed_rows": sum(old != new for old, new in zip(old_lengths, new_lengths)),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "backup_sha256": _sha(BACKUP_PATH),
        "tokenizer": MODEL,
        "tokenizer_revision": REVISION,
        "old_over_4096": sum(length > 4096 for length in old_lengths),
        "new_min": sorted_lengths[0],
        "new_median": sorted_lengths[len(sorted_lengths) // 2],
        "new_p95": sorted_lengths[int(len(sorted_lengths) * 0.95)],
        "new_max": sorted_lengths[-1],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(normalize.remote(), indent=2, sort_keys=True))
