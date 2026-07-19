from __future__ import annotations

import copy

import pytest

from backend.policy import PolicyError, canonical_hash, validate_launch
from data.m3_eval_panel import EVAL_PANEL_PROTOCOL
from data.m3_rewrite_judge import (
    MODELS,
    PAIRWISE_DIMENSIONS,
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
                "input_text": f"Source {index} was published in 2026.",
                "protected_literals": [str(index), "2026"],
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
    assert len(tasks) == 2048
    pairwise = [row for row in tasks if row["task_type"] == "pairwise"]
    preservation = [row for row in tasks if row["task_type"] == "preservation"]
    assert len(pairwise) == len(preservation) == 1024
    assert {(row["model"], row["dimension"]) for row in pairwise} == {
        (model, dimension) for model in MODELS for dimension in PAIRWISE_DIMENSIONS
    }
    sides = [row["treatment_side"] for row in pairwise]
    assert 450 < sides.count("A") < 574
    first = pairwise[0]
    assert first["treatment_side"] == treatment_side(
        first["model"], first["dimension"], first["fingerprint"]
    )
    assert "Return exactly A, B, or TIE" in first["prompt"]
    assert all("Return exactly PASS or FAIL" in row["prompt"] for row in preservation)


def test_summary_keeps_models_separate_and_counts_ties() -> None:
    panel, sft, treatment = fixtures()
    rows = []
    for index, task in enumerate(build_tasks(panel, sft, treatment)):
        if task["task_type"] == "pairwise":
            outcome = ("win", "loss", "tie")[index % 3]
            choice = (
                task["treatment_side"]
                if outcome == "win"
                else ("B" if task["treatment_side"] == "A" else "A")
                if outcome == "loss"
                else "TIE"
            )
            rows.append({**task, "choice": choice, "outcome": outcome})
        else:
            choice = "PASS" if index % 4 else "FAIL"
            rows.append({**task, "choice": choice, "passed": choice == "PASS"})
    result = summarize(rows)
    assert result["comparisons"] == 2048
    for model in MODELS:
        for dimension in PAIRWISE_DIMENSIONS:
            cell = result["models"][model]["pairwise"][dimension]
            assert cell["comparisons"] == 256
            assert cell["wins"] + cell["losses"] + cell["ties"] == 256
        for arm in ("SFT14", "HUMANWRITE14"):
            cell = result["models"][model]["content_preservation"][arm]
            assert cell["trials"] == 256


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
