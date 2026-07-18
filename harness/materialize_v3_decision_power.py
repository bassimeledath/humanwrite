"""Materialize the candidate-blind measurement-v3 decision and power contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from harness.measurement_v3 import prospective_exact_decision_power
from harness.measurement_v3_operator import DECISION_ENDPOINTS


SCHEMA = "dftr.measurement.decision_power_config.v3"
RULE_ID = "measurement-v3-lower-variance-intersection-v1"
MASTER_SEED = 95173
TRIALS = 10_000

# Candidate-blind calibration from 5,000 random human-only 128-vs-128 splits.
# The meaningful-effect boundary is the rounded joint 95% noise envelope; the
# powered alternative is twice that improvement, as preregistered in the audit.
FAMILY_EFFECT_BOUNDARY = -0.0006
ALTERNATIVE_EFFECT = -0.0012
FAMILY_EFFECT_SDS = (0.000295, 0.000196)
FAMILY_EFFECT_CORRELATION = 0.50

# Token L2 non-inferiority is one quarter of the human-only random-split SD
# (0.00085788), rounded outward.  The rate margins are absolute proportions.
TOKEN_L2_MARGIN = 0.00022
TOKEN_L2_EFFECT_SD = 0.00008
EQUIVALENCE_MARGIN = 0.05
EQUIVALENCE_EFFECT_SD = 0.01
NONINFERIORITY_MARGIN = 0.02
NONINFERIORITY_EFFECT_SD = 0.0075
ALPHA = 0.05


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _generator(mean_effect: float):
    covariance = np.asarray(
        [
            [
                FAMILY_EFFECT_SDS[0] ** 2,
                FAMILY_EFFECT_CORRELATION
                * FAMILY_EFFECT_SDS[0]
                * FAMILY_EFFECT_SDS[1],
            ],
            [
                FAMILY_EFFECT_CORRELATION
                * FAMILY_EFFECT_SDS[0]
                * FAMILY_EFFECT_SDS[1],
                FAMILY_EFFECT_SDS[1] ** 2,
            ],
        ],
        dtype=np.float64,
    )

    def draw(rng: np.random.Generator, _index: int) -> dict[str, Any]:
        effects = rng.multivariate_normal(
            [mean_effect, mean_effect], covariance
        ).astype(float)
        return {
            "family_effects": effects.tolist(),
            "family_pvalues": [
                _normal_cdf(effect / standard_deviation)
                for effect, standard_deviation in zip(effects, FAMILY_EFFECT_SDS)
            ],
            "token_l2_difference": float(rng.normal(0.0, TOKEN_L2_EFFECT_SD)),
            "equivalence_effect": float(rng.normal(0.0, EQUIVALENCE_EFFECT_SD)),
            "noninferiority_effect": float(
                rng.normal(0.0, NONINFERIORITY_EFFECT_SD)
            ),
            "positive_controls_qualified": True,
        }

    return draw


def _exact_decision(trial: dict[str, Any]) -> bool:
    effects = trial["family_effects"]
    pvalues = trial["family_pvalues"]
    family_passes = [
        effect <= FAMILY_EFFECT_BOUNDARY and pvalue <= ALPHA
        for effect, pvalue in zip(effects, pvalues)
    ]
    return bool(
        len(effects) == 2
        and all(family_passes)
        and all(effect < 0 for effect in effects)
        and trial["token_l2_difference"] <= TOKEN_L2_MARGIN
        and abs(trial["equivalence_effect"]) <= EQUIVALENCE_MARGIN
        and trial["noninferiority_effect"] <= NONINFERIORITY_MARGIN
        and trial["positive_controls_qualified"] is True
    )


def materialize(output_path: Path) -> dict[str, Any]:
    analysis_paths = {
        "measurement_v3.py": Path(__file__).resolve().parent
        / "src/harness/measurement_v3.py",
        "score_v3_candidates.py": Path(__file__).resolve().parent
        / "score_v3_candidates.py",
    }
    analysis_spec = {
        name: _file_sha(path) for name, path in analysis_paths.items()
    }
    rule = {
        "rule_id": RULE_ID,
        "endpoints": DECISION_ENDPOINTS,
        "intersection_required": True,
        "alpha": ALPHA,
        "family_effect_boundary": FAMILY_EFFECT_BOUNDARY,
        "token_l2_margin": TOKEN_L2_MARGIN,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
    }
    null_spec = {
        "generator": "correlated_gaussian_candidate_blind_human_floor.v1",
        "mean_effect": 0.0,
        "family_effect_sds": FAMILY_EFFECT_SDS,
        "family_effect_correlation": FAMILY_EFFECT_CORRELATION,
        "token_l2_effect_sd": TOKEN_L2_EFFECT_SD,
        "equivalence_effect_sd": EQUIVALENCE_EFFECT_SD,
        "noninferiority_effect_sd": NONINFERIORITY_EFFECT_SD,
        "positive_controls_qualified": True,
    }
    alternative_spec = {**null_spec, "mean_effect": ALTERNATIVE_EFFECT}
    power = prospective_exact_decision_power(
        null_generator=_generator(0.0),
        alternative_generator=_generator(ALTERNATIVE_EFFECT),
        decision_rule=_exact_decision,
        rule_id=RULE_ID,
        trials=TRIALS,
        seed=MASTER_SEED,
        decision_boundary=FAMILY_EFFECT_BOUNDARY,
        alternative_effect=ALTERNATIVE_EFFECT,
        effect_direction="less",
        type_i_max=ALPHA,
        power_min=0.80,
    )
    artifact = {
        "artifact_schema": SCHEMA,
        "status": "frozen",
        "candidate_outputs_opened": False,
        "decision_rule": rule,
        "decision_rule_sha256": _canonical_sha(rule),
        "analysis_code_sha256": _canonical_sha(analysis_spec),
        "null_generator_sha256": _canonical_sha(null_spec),
        "alternative_generator_sha256": _canonical_sha(alternative_spec),
        "training_reward_model_ids": ["Qwen/Qwen3-4B"],
        "power": power,
    }
    if not power["all_targets_pass"]:
        raise RuntimeError(f"decision power did not qualify: {power}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_path": str(output_path),
        "output_sha256": _file_sha(output_path),
        "null_rate": power["null"]["rate"],
        "alternative_power": power["alternative"]["rate"],
        "decision_rule_sha256": artifact["decision_rule_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/m2/m2_measurement_v3_decision_power_v1.json"),
    )
    args = parser.parse_args()
    print(json.dumps(materialize(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
