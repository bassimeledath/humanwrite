from __future__ import annotations

import yaml

from experiments.m3.materialize_rewrite_sft_smoke_config import build_config, materialize
from experiments.m3.rewrite_sft_smoke import validate_m3_rewrite_sft_smoke_config


def test_materializer_binds_terminal_rewrite_sha_and_writes_valid_yaml(tmp_path) -> None:
    rewrite_sha = "c" * 64
    config = build_config(rewrite_sha)
    assert config["data"]["rewrite_tasks_sha256"] == rewrite_sha
    assert validate_m3_rewrite_sft_smoke_config(config) is config

    output = tmp_path / "smoke.yaml"
    materialize(rewrite_sha, output)
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert loaded == config
