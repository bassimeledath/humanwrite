"""Fail-closed promotion decision for the frozen M3 4K rewrite screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from data.m3_rewrite_judge import MODELS
from experiments.m1.contracts import write_json
from experiments.m3.rewrite_embedding_score import FAMILIES


SCHEMA = "humanwrite.m3.rewrite_4k_decision.v1"
AUTOMATIC_SCHEMA = "humanwrite.m3.rewrite_automatic_score.v1"
JUDGE_SCHEMA = "humanwrite.m3.rewrite_judge_summary.v2"
EMBEDDING_SCHEMA = "humanwrite.m3.rewrite_embedding_score.v1"


class M3RewriteDecisionError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bound(path: Path, sha256: str) -> dict[str, Any]:
    if (
        not re.fullmatch(r"[0-9a-f]{64}", sha256)
        or not path.is_file()
        or path.is_symlink()
        or file_sha256(path) != sha256
    ):
        raise M3RewriteDecisionError(f"artifact binding failed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M3RewriteDecisionError(f"artifact is not an object: {path}")
    return value


def _check(name: str, value: float | int, operator: str, threshold: float | int) -> dict[str, Any]:
    operations = {
        ">=": lambda: value >= threshold,
        "<=": lambda: value <= threshold,
        "==": lambda: value == threshold,
        "<": lambda: value < threshold,
    }
    if operator not in operations:
        raise M3RewriteDecisionError(f"unsupported decision operator: {operator}")
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(operations[operator]()),
    }


def decide(
    automatic: dict[str, Any],
    judge: dict[str, Any],
    embedding: dict[str, Any],
) -> dict[str, Any]:
    if automatic.get("artifact_schema") != AUTOMATIC_SCHEMA or automatic.get("records") != 256:
        raise M3RewriteDecisionError("automatic score contract failed")
    if judge.get("artifact_schema") != JUDGE_SCHEMA or judge.get("comparisons") != 2048:
        raise M3RewriteDecisionError("judge summary contract failed")
    if embedding.get("artifact_schema") != EMBEDDING_SCHEMA:
        raise M3RewriteDecisionError("embedding score contract failed")
    if set(judge.get("models") or {}) != set(MODELS):
        raise M3RewriteDecisionError("judge family contract failed")
    if set(embedding.get("families") or {}) != set(FAMILIES):
        raise M3RewriteDecisionError("embedding family contract failed")

    treatment = automatic["arms"]["HUMANWRITE14"]
    control = automatic["arms"]["SFT14"]
    checks: list[dict[str, Any]] = [
        _check(
            "meaningful_edit_ai_inputs",
            treatment["meaningful_edit_ai_inputs"]["point"],
            ">=",
            0.70,
        ),
        _check(
            "treatment_protected_literal_and_nonempty_sentinel",
            treatment["hard_content_preservation"]["point"],
            "==",
            1.0,
        ),
        _check(
            "replacement_character_count",
            treatment["replacement_character_count"],
            "==",
            0,
        ),
        _check(
            "unexpected_non_latin_treatment_minus_sft",
            treatment["unexpected_non_latin"]["point"]
            - control["unexpected_non_latin"]["point"],
            "<=",
            0.02,
        ),
    ]

    for model in MODELS:
        report = judge["models"][model]
        checks.extend(
            [
                _check(
                    f"{model}:human_style_preference_ties_half",
                    report["pairwise"]["human_style"]["preference_rate_ties_half"],
                    ">=",
                    0.55,
                ),
                _check(
                    f"{model}:overall_quality_preference_ties_half",
                    report["pairwise"]["overall_quality"]["preference_rate_ties_half"],
                    ">=",
                    0.50,
                ),
                _check(
                    f"{model}:content_preservation_treatment_minus_sft",
                    report["content_preservation"]["treatment_minus_sft"],
                    ">=",
                    -0.03,
                ),
            ]
        )

    lexical_effects: dict[str, Any] = {}
    for metric, report in sorted(automatic["lexical"].items()):
        effect = float(report["treatment_minus_sft"])
        sd = float(report["human_split"]["sd"])
        lexical_effects[metric] = {
            "treatment_minus_sft": effect,
            "human_split_sd": sd,
            "directionally_improved": effect < 0.0,
            "within_noninferiority_margin": effect <= sd,
        }
    checks.extend(
        [
            _check(
                "at_least_one_lexical_distribution_metric_improves",
                sum(value["directionally_improved"] for value in lexical_effects.values()),
                ">=",
                1,
            ),
            _check(
                "no_lexical_metric_worsens_over_one_human_split_sd",
                sum(not value["within_noninferiority_margin"] for value in lexical_effects.values()),
                "==",
                0,
            ),
        ]
    )

    embedding_diagnostics = {
        family: {
            "treatment_minus_sft": float(report["treatment_minus_sft"]),
            "directionally_improved": float(report["treatment_minus_sft"]) < 0.0,
            "promotion_gate_at_4k": False,
        }
        for family, report in sorted(embedding["families"].items())
    }
    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "artifact_schema": SCHEMA,
        "stage": "4K-to-16K",
        "decision": "promote_to_16k" if not failures else "stop_after_4k",
        "passed": not failures,
        "checks": checks,
        "failed_checks": failures,
        "lexical_effects": lexical_effects,
        "embedding_diagnostics": embedding_diagnostics,
        "note": (
            "Both frozen judge families must independently pass. Embedding MMD is "
            "reported but is not a 4K promotion endpoint."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("automatic", "judge", "embedding"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        name: load_bound(getattr(args, name), getattr(args, f"{name}_sha256"))
        for name in ("automatic", "judge", "embedding")
    }
    report = decide(inputs["automatic"], inputs["judge"], inputs["embedding"])
    report["input_hashes"] = {
        name: getattr(args, f"{name}_sha256")
        for name in ("automatic", "judge", "embedding")
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

