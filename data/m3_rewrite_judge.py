"""Frozen blinded pairwise-judge contract for the M3 rewrite screen."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
from typing import Any

from .m3_eval_panel import EVAL_PANEL_PROTOCOL


PROTOCOL = "humanwrite.m3.rewrite_pairwise_judge.v1"
MODELS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite",
]
DIMENSIONS = {
    "human_style": (
        "Choose the response that reads more like careful human-authored prose. "
        "Favor natural variation, specific development, and context-appropriate "
        "phrasing; penalize formulaic structure, canned transitions, repetition, "
        "and conspicuous model-like artifacts."
    ),
    "overall_quality": (
        "Choose the response with better writing quality while following the rewrite "
        "assignment. Favor clear, coherent, specific, engaging prose, and penalize "
        "generic, repetitive, awkward, unsupported, or instruction-violating writing."
    ),
}
RANDOMIZATION = {
    "algorithm": "sha256(master_seed:model:dimension:fingerprint).parity.v1",
    "master_seed": 8903,
}
RESPONSE_CONTRACT = "single uppercase A, B, or TIE"


class M3RewriteJudgeError(ValueError):
    pass


def treatment_side(model: str, dimension: str, fingerprint: str) -> str:
    digest = hashlib.sha256(
        f"{RANDOMIZATION['master_seed']}:{model}:{dimension}:{fingerprint}".encode()
    ).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def judge_prompt(
    *, assignment: str, rubric: str, candidate_a: str, candidate_b: str
) -> str:
    return f"""Compare two rewrites of the same source text.

Criterion: {rubric}

Judge only this criterion. Do not infer which system wrote either candidate. If neither
candidate is meaningfully better on this criterion, return TIE.

Return exactly A, B, or TIE.

=== REWRITE ASSIGNMENT ===
{assignment}

=== CANDIDATE A ===
{candidate_a}

=== CANDIDATE B ===
{candidate_b}
"""


def _output_map(rows: list[dict[str, Any]], arm: str) -> dict[str, str]:
    if len(rows) != 256:
        raise M3RewriteJudgeError(f"{arm} output cardinality mismatch")
    result: dict[str, str] = {}
    for row in rows:
        fingerprint = str(row.get("fingerprint") or "")
        output = str(row.get("output") or "").strip()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
            or row.get("arm") != arm
            or not output
            or fingerprint in result
        ):
            raise M3RewriteJudgeError(f"{arm} output identity mismatch")
        result[fingerprint] = output
    return result


def build_tasks(
    panel: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        len(panel) != 256
        or len({str(row.get("fingerprint") or "") for row in panel}) != 256
        or any(row.get("artifact_schema") != EVAL_PANEL_PROTOCOL for row in panel)
    ):
        raise M3RewriteJudgeError("fresh evaluation panel invariants failed")
    sft = _output_map(sft_rows, "SFT14")
    treatment = _output_map(treatment_rows, "HUMANWRITE14")
    panel_ids = {str(row["fingerprint"]) for row in panel}
    if set(sft) != panel_ids or set(treatment) != panel_ids:
        raise M3RewriteJudgeError("candidate/panel pairing mismatch")

    tasks: list[dict[str, Any]] = []
    for row in sorted(panel, key=lambda value: str(value["fingerprint"])):
        fingerprint = str(row["fingerprint"])
        for model in MODELS:
            for dimension, rubric in DIMENSIONS.items():
                side = treatment_side(model, dimension, fingerprint)
                candidates = {
                    side: treatment[fingerprint],
                    "B" if side == "A" else "A": sft[fingerprint],
                }
                tasks.append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "fingerprint": fingerprint,
                        "treatment_side": side,
                        "prompt": judge_prompt(
                            assignment=str(row["prompt"]),
                            rubric=rubric,
                            candidate_a=candidates["A"],
                            candidate_b=candidates["B"],
                        ),
                    }
                )
    if len(tasks) != 1024:
        raise M3RewriteJudgeError("judge task cardinality mismatch")
    return tasks


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float]:
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return {"low": center - half, "high": center + half, "coverage": 0.95}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        (model, dimension, fingerprint)
        for model in MODELS
        for dimension in DIMENSIONS
        for fingerprint in {
            str(row.get("fingerprint") or "") for row in rows
        }
    }
    keys = {
        (str(row.get("model")), str(row.get("dimension")), str(row.get("fingerprint")))
        for row in rows
    }
    fingerprints = {str(row.get("fingerprint") or "") for row in rows}
    if len(rows) != 1024 or len(fingerprints) != 256 or keys != expected:
        raise M3RewriteJudgeError("judge result cardinality mismatch")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("choice") not in {"A", "B", "TIE"}:
            raise M3RewriteJudgeError("judge result choice mismatch")
        cells[(str(row["model"]), str(row["dimension"]))].append(row)
    reports: dict[str, Any] = {}
    for model in MODELS:
        reports[model] = {}
        for dimension in DIMENSIONS:
            values = cells[(model, dimension)]
            wins = sum(row["outcome"] == "win" for row in values)
            losses = sum(row["outcome"] == "loss" for row in values)
            ties = sum(row["outcome"] == "tie" for row in values)
            decisive = wins + losses
            reports[model][dimension] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "comparisons": len(values),
                "win_rate_all": wins / len(values),
                "preference_rate_ties_half": (wins + 0.5 * ties) / len(values),
                "decisive_win_rate": wins / decisive if decisive else None,
                "wilson_95_all": _wilson(wins, len(values)),
            }
    return {
        "artifact_schema": "humanwrite.m3.rewrite_pairwise_judge_summary.v1",
        "comparisons": len(rows),
        "models": reports,
    }


__all__ = [
    "DIMENSIONS",
    "MODELS",
    "M3RewriteJudgeError",
    "PROTOCOL",
    "RANDOMIZATION",
    "RESPONSE_CONTRACT",
    "build_tasks",
    "judge_prompt",
    "summarize",
    "treatment_side",
]
