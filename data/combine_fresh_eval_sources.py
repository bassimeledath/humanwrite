"""Combine independently selected fresh evaluation pools with strict identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def combine(inputs: list[Path], output: Path, manifest_path: Path) -> dict[str, Any]:
    if output.exists() or manifest_path.exists():
        raise FileExistsError("combined source artifacts are immutable")
    rows = [row for path in inputs for row in _rows(path)]
    fingerprints = [str(row.get("fingerprint") or "") for row in rows]
    domains = [str(row.get("domain") or "").casefold() for row in rows]
    if (
        any(not item for item in fingerprints)
        or any(not item for item in domains)
        or len(set(fingerprints)) != len(rows)
        or len(set(domains)) != len(rows)
        or any(row.get("split") != "dev" for row in rows)
    ):
        raise ValueError("fresh source pools are not identity-disjoint dev records")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    output.write_text(payload, encoding="utf-8")
    manifest = {
        "artifact_schema": "dftr.measurement.combined_fresh_source_pool.v1",
        "status": "frozen",
        "candidate_outputs_opened": False,
        "rows": len(rows),
        "input_paths": [str(path) for path in inputs],
        "input_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs],
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "unique_fingerprints": len(set(fingerprints)),
        "unique_domains": len(set(domains)),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(combine(args.input, args.output, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
