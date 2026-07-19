"""Materialize SHA-bound fresh-panel generation configs for all M3 arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.m3.rewrite_generate_14b import build_config


DEFAULT_DIR = Path("configs/m3")


def materialize(
    *,
    panel_sha256: str,
    sft_manifest_path: str,
    sft_manifest_sha256: str,
    treatment_manifest_path: str,
    treatment_manifest_sha256: str,
    output_dir: Path,
) -> list[Path]:
    configs = {
        output_dir / "m3_rewrite_generate_base_256_v1.yaml": build_config(
            "BASE", panel_sha256
        ),
        output_dir / "m3_rewrite_generate_sft14_256_v1.yaml": build_config(
            "SFT14",
            panel_sha256,
            training_manifest_path=sft_manifest_path,
            training_manifest_sha256=sft_manifest_sha256,
        ),
        output_dir / "m3_rewrite_generate_humanwrite14_256_v1.yaml": build_config(
            "HUMANWRITE14",
            panel_sha256,
            training_manifest_path=treatment_manifest_path,
            training_manifest_sha256=treatment_manifest_sha256,
        ),
    }
    if any(path.exists() for path in configs):
        raise ValueError("refusing to overwrite frozen generation configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, config in configs.items():
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return list(configs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--sft-manifest-path", required=True)
    parser.add_argument("--sft-manifest-sha256", required=True)
    parser.add_argument("--treatment-manifest-path", required=True)
    parser.add_argument("--treatment-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    paths = materialize(
        panel_sha256=args.panel_sha256,
        sft_manifest_path=args.sft_manifest_path,
        sft_manifest_sha256=args.sft_manifest_sha256,
        treatment_manifest_path=args.treatment_manifest_path,
        treatment_manifest_sha256=args.treatment_manifest_sha256,
        output_dir=args.output_dir,
    )
    print("\n".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
