from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.pipeline import DEFAULT_INPUT, DEFAULT_OUTPUT, build_dataset


ROOT = Path(__file__).resolve().parents[1]


def verify() -> dict[str, object]:
    expected = build_dataset(DEFAULT_INPUT)
    output = DEFAULT_OUTPUT
    if not output.is_dir():
        raise FileNotFoundError(f"missing generated data artifacts: {output}")
    split_hashes = json.loads((output / "split_hashes.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "experiments" / "reproducibility" / "m0.json").read_text(encoding="utf-8"))
    required = [
        ROOT / "data" / "tests" / "test_pipeline.py",
        ROOT / "experiments" / "tests" / "test_tier0_metrics.py",
        ROOT / "infra" / "tests" / "test_local_backend.py",
        ROOT / "ledger" / "tests" / "test_ledger.py",
        ROOT / "configs" / "m0_offline_smoke.yaml",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    return {
        "fixture_records": expected["source"]["record_count"],
        "train_hash_matches": split_hashes["train"] == expected["split_hashes"]["train"],
        "dev_hash_matches": split_hashes["dev"] == expected["split_hashes"]["dev"],
        "empty_outline_count": summary["empty_outline_count"],
        "empty_outline_expected": expected["summary"]["empty_outline_count"],
        "hidden_test_materialized_locally": json.loads(
            (output / "hidden_test_boundary.json").read_text(encoding="utf-8")
        )["materialized_locally"],
        "reproducibility_metadata_version": metadata["version"],
        "missing_required_files": missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the offline M0 repository surface.")
    parser.parse_args(argv)
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

