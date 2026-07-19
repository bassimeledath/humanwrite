"""Assemble and SHA-freeze the fresh 256-item public rewrite panel."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from data.m3_eval_panel import assemble_eval_panel
from data.m3_scientific_corpus import BASE_MODEL, BASE_REVISION
from experiments.m1.contracts import write_json, write_jsonl
from experiments.m3.rewrite_4k_train import build_config, prepare_batch


OUTPUT = Path("/tmp/m3-fresh-rewrite-eval-panel-256-v1.jsonl")
MANIFEST = Path("/tmp/m3-fresh-rewrite-eval-panel-256-v1.manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, sha256: str) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256) or not path.is_file() or file_sha256(path) != sha256:
        raise ValueError(f"evaluation artifact hash mismatch: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def materialize(clean_path: Path, clean_sha: str, api_path: Path, api_sha: str, output: Path, manifest_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    sources, api_rows = load(clean_path, clean_sha), load(api_path, api_sha)
    if len(sources) != 640 or len(api_rows) != 224:
        raise ValueError("evaluation input cardinality mismatch")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    panel = assemble_eval_panel(
        sources,
        api_rows,
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
    config = build_config("SFT14", "a" * 64)
    sequence_lengths = []
    for row in panel:
        prepared = prepare_batch(
            tokenizer,
            [{"prompt": row["prompt"], "completion": row["human_reference"]}],
            config,
        )
        sequence_lengths.append(int(prepared["attention_mask"].sum().item()))
    if output.exists() or manifest_path.exists():
        raise ValueError("refusing to overwrite frozen evaluation panel")
    write_jsonl(output, panel)
    panel_sha = file_sha256(output)
    result = {
        "artifact_schema": "humanwrite.m3.rewrite_eval_panel_manifest.v1",
        "clean_pool_path": str(clean_path),
        "clean_pool_sha256": clean_sha,
        "api_inputs_path": str(api_path),
        "api_inputs_sha256": api_sha,
        "panel_path": str(output),
        "panel_sha256": panel_sha,
        "records": 256,
        "category_counts": dict(sorted(Counter(row["category"] for row in panel).items())),
        "unique_domains": len({row["domain"] for row in panel}),
        "max_sequence_tokens_against_human_reference": max(sequence_lengths),
        "selection_opened": False,
    }
    write_json(manifest_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--clean-sha256", required=True)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--api-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(materialize(args.clean, args.clean_sha256, args.api, args.api_sha256, args.output, args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
