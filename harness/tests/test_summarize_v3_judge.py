from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_v3_judge.py"
SPEC = importlib.util.spec_from_file_location("summarize_v3_judge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_quality_summary_uses_frozen_primary_and_noninferiority_rules(tmp_path) -> None:
    contract = {
        "comparison_arms": ["GOOD_vs_SFT", "BAD_vs_SFT"],
        "dimensions": {"overall_quality": "x", "human_style": "y"},
        "prompt_count": 128,
        "decision": {
            "primary_dimension": "human_style",
            "minimum_win_rate": 0.55,
            "one_sided_alpha": 0.05,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    rows = []
    for treatment, wins in (("GOOD", 90), ("BAD", 30)):
        for dimension in contract["dimensions"]:
            for index in range(128):
                rows.append(
                    {
                        "treatment": treatment,
                        "dimension": dimension,
                        "prompt_id": f"p-{index}",
                        "treatment_win": index < wins,
                    }
                )
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = module.summarize(results_path, contract_path)
    assert summary["quality_winners"] == ["GOOD"]
    assert summary["reports"]["GOOD"]["human_style"]["win_rate"] == 90 / 128
    assert summary["reports"]["BAD"]["quality_promotion_pass"] is False
