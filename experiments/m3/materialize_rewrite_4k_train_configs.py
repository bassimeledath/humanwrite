"""Materialize SHA-bound configs for the matched M3 4K training arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.m3.rewrite_4k_train import build_config


DEFAULT_SFT_OUTPUT = Path("configs/m3/m3_rewrite_train_14b_4k_sft_v1.yaml")
DEFAULT_TREATMENT_OUTPUT = Path(
    "configs/m3/m3_rewrite_train_14b_4k_humanwrite_v1.yaml"
)


def materialize(corpus_sha256: str, sft_output: Path, treatment_output: Path) -> None:
    outputs = {
        sft_output: build_config("SFT14", corpus_sha256),
        treatment_output: build_config("HUMANWRITE14", corpus_sha256),
    }
    if sft_output == treatment_output or any(path.exists() for path in outputs):
        raise ValueError("refusing to overwrite or alias frozen 4K training configs")
    for path, config in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--sft-output", type=Path, default=DEFAULT_SFT_OUTPUT)
    parser.add_argument(
        "--treatment-output", type=Path, default=DEFAULT_TREATMENT_OUTPUT
    )
    args = parser.parse_args()
    materialize(args.corpus_sha256, args.sft_output, args.treatment_output)
    print(args.sft_output)
    print(args.treatment_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
