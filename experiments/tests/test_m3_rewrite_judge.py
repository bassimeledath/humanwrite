from __future__ import annotations

import copy

import pytest

from backend.policy import PolicyError, canonical_hash, validate_launch
from data.m3_eval_panel import EVAL_PANEL_PROTOCOL
from data.m3_rewrite_judge import (
    DIMENSIONS,
    MODELS,
    build_tasks,
    summarize,
    treatment_side,
)
from experiments.m3.materialize_rewrite_judge_config import config


def fixtures() -> tuple[list[dict], list[dict], list[dict]]:
    panel = []
    sft = []
    treatment = []
    for index in range(256):
        fingerprint = f"{index:064x}"
        panel.append(
            {
                "artifact_schema": EVAL_PANEL_PROTOCOL,
                "fingerprint": fingerprint,
                "prompt": f"Rewrite source passage {index} while preserving its facts.",
            }
        )
        sft.append({"fingerprint": fingerprint, "arm": "SFT14", "output": f"SFT {index}"})
        treatment.append(
            {"fingerprint": fingerprint, "arm": "HUMANWRITE14", "output": f"Treatment {index}"}
        )
    return panel, sft, treatment


def test_tasks_are_balanced_hash_randomized_and_complete() -> None:
    panel, sft, treatment = fixtures()
    tasks = build_tasks(panel, sft, treatment)
    assert len(tasks) == 1024
    assert {(row["model"], row["dimension"]) for row in tasks} == {
        (model, dimension) for model in MODELS for dimension in DIMENSIONS
    }
    sides = [row["treatment_side"] for row in tasks]
    assert 450 < sides.count("A") < 574
    first = tasks[0]
    assert first["treatment_side"] == treatment_side(
        first["model"], first["dimension"], first["fingerprint"]
    )
    assert "Return exactly A, B, or TIE" in first["prompt"]


def test_summary_keeps_models_separate_and_counts_ties() -> None:
    panel, sft, treatment = fixtures()
    rows = []
    for index, task in enumerate(build_tasks(panel, sft, treatment)):
        outcome = ("win", "loss", "tie")[index % 3]
        choice = (
            task["treatment_side"]
            if outcome == "win"
            else ("B" if task["treatment_side"] == "A" else "A")
            if outcome == "loss"
            else "TIE"
        )
        rows.append({**task, "choice": choice, "outcome": outcome})
    result = summarize(rows)
    assert result["comparisons"] == 1024
    for model in MODELS:
        for dimension in DIMENSIONS:
            cell = result["models"][model][dimension]
            assert cell["comparisons"] == 256
            assert cell["wins"] + cell["losses"] + cell["ties"] == 256


def test_gateway_policy_accepts_only_frozen_judge_contract() -> None:
    value = config(panel_sha256="a" * 64, sft_sha256="b" * 64, treatment_sha256="c" * 64)
    payload = {
        "config": value,
        "config_hash": canonical_hash(value),
        "budget_class": "promo",
        "preregistration": {
            "kind": "prereg",
            "status": "open",
            "comparison": value["run"]["comparison_id"],
        },
    }
    policy = validate_launch(payload)
    assert policy.task_kind == "rewrite_judging"
    assert policy.api_reserved_cost_usd == pytest.approx(10.0)
    drift = copy.deepcopy(value)
    drift["judge"]["models"].reverse()
    payload["config"] = drift
    payload["config_hash"] = canonical_hash(drift)
    with pytest.raises(PolicyError, match="frozen contract"):
        validate_launch(payload)
