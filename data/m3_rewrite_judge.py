"""Frozen blinded quality and preservation judge for the M3 rewrite screen."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
from typing import Any

from .m3_eval_panel import EVAL_PANEL_PROTOCOL


PROTOCOL = "humanwrite.m3.rewrite_judge.v2"
MODELS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite",
]
PAIRWISE_DIMENSIONS = {
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
PAIRWISE_RESPONSE_CONTRACT = "single uppercase A, B, or TIE"
PRESERVATION_RESPONSE_CONTRACT = "single uppercase PASS or FAIL"


class M3RewriteJudgeError(ValueError):
    pass


def task_id(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode()).hexdigest()


def treatment_side(model: str, dimension: str, fingerprint: str) -> str:
    digest = hashlib.sha256(
        f"{RANDOMIZATION['master_seed']}:{model}:{dimension}:{fingerprint}".encode()
    ).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def pairwise_prompt(
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


def preservation_prompt(
    *, assignment: str, source: str, protected_literals: list[str], candidate: str
) -> str:
    literal_text = "\n".join(f"- {value}" for value in protected_literals) or "(none)"
    return f"""Decide whether a rewrite preserves the source's factual content and intent.

Return PASS only if the candidate preserves every material claim, relationship, name,
number, date, quotation, and required literal from the source; introduces no unsupported
claim; and follows the rewrite assignment. Stylistic rewording is allowed. Return FAIL if
anything material is added, removed, contradicted, or altered.

Return exactly PASS or FAIL.

=== REWRITE ASSIGNMENT ===
{assignment}

=== SOURCE TEXT ===
{source}

=== REQUIRED LITERALS ===
{literal_text}

=== CANDIDATE REWRITE ===
{candidate}
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
    outputs = {
        "SFT14": _output_map(sft_rows, "SFT14"),
        "HUMANWRITE14": _output_map(treatment_rows, "HUMANWRITE14"),
    }
    panel_ids = {str(row["fingerprint"]) for row in panel}
    if any(set(values) != panel_ids for values in outputs.values()):
        raise M3RewriteJudgeError("candidate/panel pairing mismatch")

    tasks: list[dict[str, Any]] = []
    for row in sorted(panel, key=lambda value: str(value["fingerprint"])):
        fingerprint = str(row["fingerprint"])
        for model in MODELS:
            for dimension, rubric in PAIRWISE_DIMENSIONS.items():
                side = treatment_side(model, dimension, fingerprint)
                candidates = {
                    side: outputs["HUMANWRITE14"][fingerprint],
                    "B" if side == "A" else "A": outputs["SFT14"][fingerprint],
                }
                tasks.append(
                    {
                        "task_id": task_id("pairwise", model, dimension, fingerprint),
                        "task_type": "pairwise",
                        "model": model,
                        "dimension": dimension,
                        "fingerprint": fingerprint,
                        "treatment_side": side,
                        "prompt": pairwise_prompt(
                            assignment=str(row["prompt"]),
                            rubric=rubric,
                            candidate_a=candidates["A"],
                            candidate_b=candidates["B"],
                        ),
                    }
                )
            for arm in ("SFT14", "HUMANWRITE14"):
                tasks.append(
                    {
                        "task_id": task_id("preservation", model, arm, fingerprint),
                        "task_type": "preservation",
                        "model": model,
                        "dimension": "content_preservation",
                        "fingerprint": fingerprint,
                        "arm": arm,
                        "prompt": preservation_prompt(
                            assignment=str(row["prompt"]),
                            source=str(row["input_text"]),
                            protected_literals=[str(value) for value in row.get("protected_literals") or []],
                            candidate=outputs[arm][fingerprint],
                        ),
                    }
                )
    if len(tasks) != 2048 or len({row["task_id"] for row in tasks}) != 2048:
        raise M3RewriteJudgeError("judge task cardinality mismatch")
    return tasks


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float]:
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return {"low": center - half, "high": center + half, "coverage": 0.95}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2048 or len({str(row.get("task_id") or "") for row in rows}) != 2048:
        raise M3RewriteJudgeError("judge result cardinality mismatch")
    fingerprints = {str(row.get("fingerprint") or "") for row in rows}
    if len(fingerprints) != 256:
        raise M3RewriteJudgeError("judge fingerprint cardinality mismatch")

    pairwise_cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    preservation_cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task_type") == "pairwise":
            if row.get("choice") not in {"A", "B", "TIE"} or row.get("outcome") not in {
                "win",
                "loss",
                "tie",
            }:
                raise M3RewriteJudgeError("pairwise judge result mismatch")
            pairwise_cells[(str(row["model"]), str(row["dimension"]))].append(row)
        elif row.get("task_type") == "preservation":
            if row.get("choice") not in {"PASS", "FAIL"} or row.get("arm") not in {
                "SFT14",
                "HUMANWRITE14",
            }:
                raise M3RewriteJudgeError("preservation judge result mismatch")
            preservation_cells[(str(row["model"]), str(row["arm"]))].append(row)
        else:
            raise M3RewriteJudgeError("unknown judge task type")

    reports: dict[str, Any] = {}
    for model in MODELS:
        reports[model] = {"pairwise": {}, "content_preservation": {}}
        for dimension in PAIRWISE_DIMENSIONS:
            values = pairwise_cells[(model, dimension)]
            if len(values) != 256:
                raise M3RewriteJudgeError("pairwise judge cell cardinality mismatch")
            wins = sum(row["outcome"] == "win" for row in values)
            losses = sum(row["outcome"] == "loss" for row in values)
            ties = sum(row["outcome"] == "tie" for row in values)
            decisive = wins + losses
            reports[model]["pairwise"][dimension] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "comparisons": len(values),
                "win_rate_all": wins / len(values),
                "preference_rate_ties_half": (wins + 0.5 * ties) / len(values),
                "decisive_win_rate": wins / decisive if decisive else None,
                "wilson_95_all": _wilson(wins, len(values)),
            }
        preservation_rates: dict[str, float] = {}
        for arm in ("SFT14", "HUMANWRITE14"):
            values = preservation_cells[(model, arm)]
            if len(values) != 256:
                raise M3RewriteJudgeError("preservation judge cell cardinality mismatch")
            passes = sum(row["choice"] == "PASS" for row in values)
            preservation_rates[arm] = passes / len(values)
            reports[model]["content_preservation"][arm] = {
                "passes": passes,
                "failures": len(values) - passes,
                "trials": len(values),
                "pass_rate": preservation_rates[arm],
                "wilson_95": _wilson(passes, len(values)),
            }
        reports[model]["content_preservation"]["treatment_minus_sft"] = (
            preservation_rates["HUMANWRITE14"] - preservation_rates["SFT14"]
        )
    return {
        "artifact_schema": "humanwrite.m3.rewrite_judge_summary.v2",
        "comparisons": len(rows),
        "models": reports,
    }


__all__ = [
    "MODELS",
    "M3RewriteJudgeError",
    "PAIRWISE_DIMENSIONS",
    "PAIRWISE_RESPONSE_CONTRACT",
    "PRESERVATION_RESPONSE_CONTRACT",
    "PROTOCOL",
    "RANDOMIZATION",
    "build_tasks",
    "pairwise_prompt",
    "preservation_prompt",
    "summarize",
    "task_id",
    "treatment_side",
]
