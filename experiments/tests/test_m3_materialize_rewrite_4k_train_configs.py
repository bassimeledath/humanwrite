from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.m3.materialize_rewrite_4k_train_configs import materialize
from experiments.m3.rewrite_4k_train import validate_config


def test_materializes_exact_matched_configs(tmp_path: Path) -> None:
    sft = tmp_path / "sft.yaml"
    treatment = tmp_path / "treatment.yaml"
    materialize("a" * 64, sft, treatment)
    sft_config = yaml.safe_load(sft.read_text(encoding="utf-8"))
    treatment_config = yaml.safe_load(treatment.read_text(encoding="utf-8"))
    assert validate_config(sft_config) == sft_config
    assert validate_config(treatment_config) == treatment_config
    assert sft_config["run"]["arm"] == "SFT14"
    assert treatment_config["run"]["arm"] == "HUMANWRITE14"
    assert sft_config["data"] == treatment_config["data"]


def test_refuses_to_overwrite_frozen_config(tmp_path: Path) -> None:
    sft = tmp_path / "sft.yaml"
    treatment = tmp_path / "treatment.yaml"
    materialize("a" * 64, sft, treatment)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        materialize("a" * 64, sft, treatment)
