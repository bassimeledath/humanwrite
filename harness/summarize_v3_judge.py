"""Summarize the frozen blinded v3 quality-judge comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _upper_tail(wins: int, trials: int) -> float:
    return float(
        sum(math.comb(trials, value) for value in range(wins, trials + 1))
        / (2**trials)
    )


def _lower_tail(wins: int, trials: int) -> float:
    return float(
        sum(math.comb(trials, value) for value in range(0, wins + 1))
        / (2**trials)
    )


def _wilson(wins: int, trials: int, z: float = 1.959963984540054) -> dict:
    rate = wins / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return {"low": center - half, "high": center + half, "coverage": 0.95}


def summarize(results_path: Path, contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = _rows(results_path)
    treatments = [item.removesuffix("_vs_SFT") for item in contract["comparison_arms"]]
    expected = len(treatments) * len(contract["dimensions"]) * contract["prompt_count"]
    if len(rows) != expected:
        raise RuntimeError(f"quality judge incomplete: {len(rows)}/{expected}")
    seen = set()
    reports = {}
    for treatment in treatments:
        reports[treatment] = {}
        for dimension in contract["dimensions"]:
            cells = [
                row
                for row in rows
                if row["treatment"] == treatment and row["dimension"] == dimension
            ]
            keys = {(row["treatment"], row["dimension"], row["prompt_id"]) for row in cells}
            if len(cells) != contract["prompt_count"] or len(keys) != len(cells):
                raise RuntimeError(f"quality judge cell invalid: {treatment}/{dimension}")
            if seen & keys:
                raise RuntimeError("quality judge contains duplicate comparison identities")
            seen |= keys
            wins = sum(bool(row["treatment_win"]) for row in cells)
            trials = len(cells)
            reports[treatment][dimension] = {
                "wins": wins,
                "trials": trials,
                "win_rate": wins / trials,
                "jmq_vs_control": 2 * wins / trials,
                "wilson_95": _wilson(wins, trials),
                "one_sided_better_pvalue": _upper_tail(wins, trials),
                "one_sided_worse_pvalue": _lower_tail(wins, trials),
            }
        primary = reports[treatment][contract["decision"]["primary_dimension"]]
        overall = reports[treatment]["overall_quality"]
        reports[treatment]["quality_promotion_pass"] = bool(
            primary["win_rate"] >= contract["decision"]["minimum_win_rate"]
            and primary["one_sided_better_pvalue"]
            <= contract["decision"]["one_sided_alpha"]
            and overall["one_sided_worse_pvalue"]
            > contract["decision"]["one_sided_alpha"]
        )
    return {
        "artifact_schema": "dftr.measurement.quality_judge_summary.v3",
        "status": "completed",
        "contract_sha256": _sha(contract_path),
        "results_sha256": _sha(results_path),
        "reports": reports,
        "quality_winners": [
            treatment
            for treatment in treatments
            if reports[treatment]["quality_promotion_pass"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": _sha(args.output),
                "quality_winners": result["quality_winners"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
