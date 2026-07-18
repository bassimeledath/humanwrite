from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "materialize_v3_decision_power.py"
SPEC = importlib.util.spec_from_file_location("materialize_v3_decision_power", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_materialized_decision_power_is_deterministic_and_qualified(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_result = module.materialize(first)
    second_result = module.materialize(second)
    value = json.loads(first.read_text(encoding="utf-8"))

    assert first.read_bytes() == second.read_bytes()
    assert first_result["output_sha256"] == second_result["output_sha256"]
    assert value["candidate_outputs_opened"] is False
    assert (
        value["decision_rule"]["family_effect_boundary"]
        == module.FAMILY_EFFECT_BOUNDARY
    )
    assert value["power"]["alternative_effect"] == module.ALTERNATIVE_EFFECT
    assert module.ALTERNATIVE_EFFECT < module.FAMILY_EFFECT_BOUNDARY
    assert value["power"]["type_i_pass"] is True
    assert value["power"]["power_pass"] is True
    assert value["power"]["all_targets_pass"] is True
