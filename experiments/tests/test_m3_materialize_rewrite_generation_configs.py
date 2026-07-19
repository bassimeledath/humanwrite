from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.m3.materialize_rewrite_generation_configs import materialize
from experiments.m3.rewrite_generate_14b import validate_config


def test_materializes_three_sha_bound_generation_configs(tmp_path: Path) -> None:
    paths = materialize(
        panel_sha256="a" * 64,
        sft_manifest_path="/checkpoints/runs/sft/training_manifest.json",
        sft_manifest_sha256="b" * 64,
        treatment_manifest_path="/checkpoints/runs/treatment/training_manifest.json",
        treatment_manifest_sha256="c" * 64,
        output_dir=tmp_path,
    )
    assert len(paths) == 3
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    assert {config["checkpoint"]["arm"] for config in configs} == {
        "BASE",
        "SFT14",
        "HUMANWRITE14",
    }
    assert all(validate_config(config) == config for config in configs)


def test_refuses_to_overwrite_generation_configs(tmp_path: Path) -> None:
    kwargs = {
        "panel_sha256": "a" * 64,
        "sft_manifest_path": "/checkpoints/runs/sft/training_manifest.json",
        "sft_manifest_sha256": "b" * 64,
        "treatment_manifest_path": "/checkpoints/runs/treatment/training_manifest.json",
        "treatment_manifest_sha256": "c" * 64,
        "output_dir": tmp_path,
    }
    materialize(**kwargs)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        materialize(**kwargs)
