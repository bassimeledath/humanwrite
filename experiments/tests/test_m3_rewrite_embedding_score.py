from __future__ import annotations

import numpy as np

from backend.policy import canonical_hash, validate_launch
from experiments.m3.rewrite_embedding_score import (
    bandwidths,
    build_config,
    human_floor_rows,
    mmd2_unbiased,
)


def test_unused_human_floors_are_deterministic_and_disjoint() -> None:
    clean = [{"fingerprint": f"{index:064x}", "completion": str(index)} for index in range(640)]
    panel = [{"fingerprint": f"{index:064x}"} for index in range(256)]
    first_a, first_b = human_floor_rows(clean, panel)
    second_a, second_b = human_floor_rows(clean, panel)
    assert first_a == second_a and first_b == second_b
    assert len(first_a) == len(first_b) == 128
    assert not ({row["fingerprint"] for row in first_a} & {row["fingerprint"] for row in first_b})


def test_human_bandwidths_and_mmd_detect_shift() -> None:
    rng = np.random.default_rng(4)
    floor_a = rng.normal(size=(128, 8))
    floor_b = rng.normal(size=(128, 8))
    scales = bandwidths(floor_a, floor_b)
    reference = rng.normal(size=(256, 8))
    near = reference + rng.normal(scale=0.01, size=(256, 8))
    far = reference + 3.0
    assert len(scales) == 5 and all(value > 0 for value in scales)
    assert mmd2_unbiased(near, reference, scales) < mmd2_unbiased(far, reference, scales)


def test_embedding_config_passes_frozen_gateway_policy() -> None:
    config = build_config(
        clean_pool_sha256="a" * 64,
        panel_sha256="b" * 64,
        sft_sha256="c" * 64,
        treatment_sha256="d" * 64,
    )
    policy = validate_launch(
        {
            "config": config,
            "config_hash": canonical_hash(config),
            "budget_class": "screen",
            "preregistration": {
                "kind": "prereg",
                "status": "open",
                "comparison": config["run"]["comparison_id"],
            },
        }
    )
    assert policy.gpu == "L40S"
    assert policy.task_kind == "experiment"
