#!/usr/bin/env python3
"""Freeze a non-evidentiary lower-variance plumbing snapshot.

The witness proxy intentionally copies human completions. It is valid only for
an SFT-arm runtime smoke, where witness weights are computed but never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output_dir).resolve()
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.count <= 0 or len(rows) < args.count:
        raise ValueError("not enough completed briefs for the smoke snapshot")
    selected = sorted(rows, key=lambda row: row["fingerprint"])[: args.count]
    if len({row["fingerprint"] for row in selected}) != args.count:
        raise ValueError("smoke snapshot contains duplicate fingerprints")
    anchors_sha = _write_jsonl(output / "anchors-128.jsonl", selected)
    witness = [
        {
            "fingerprint": row["fingerprint"],
            "generated_completion": row["completion"],
            "smoke_only_proxy": "human_completion_not_a_model_rollout",
        }
        for row in selected
    ]
    witness_sha = _write_jsonl(output / "witness-proxy-128.jsonl", witness)
    manifest = {
        "artifact_schema": "dftr.m2.lower_variance_smoke_snapshot.v1",
        "count": args.count,
        "source_path": str(source),
        "anchors_sha256": anchors_sha,
        "witness_proxy_sha256": witness_sha,
        "scientific_use": "plumbing_only_not_evidence",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
