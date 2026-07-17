from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.measurement_v3_operator import (
    DECISION_ENDPOINTS,
    DECISION_POWER_SCHEMA,
    EMBEDDING_SCHEMA,
    PANEL_COUNTS,
    PANEL_SCHEMA,
    POSITIVE_CONTROL_SCHEMA,
    POSITIVE_CONTROLS,
    TOKENIZATION_SCHEMA,
    MeasurementV3OperatorError,
    freeze_measurement_v3,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write(path: Path, value) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(role: str, shared: dict[str, str], *, overlap: str | None = None) -> dict:
    records = []
    for index in range(PANEL_COUNTS[role]):
        prefix = role.replace("human_", "")
        content = sha(f"{prefix}:content:{index}")
        source = overlap if index == 0 and overlap else sha(f"{prefix}:source:{index}")
        common = {
            "source_document_id": source,
            "content_sha256": content,
            "domain": f"{prefix}-{index}.example",
        }
        if role == "prompt_sources":
            records.append({**common, "prompt_id": f"prompt-{index}"})
        else:
            records.append({**common, "document_id": content})
    return {
        "artifact_schema": PANEL_SCHEMA,
        "status": "qualified",
        "frozen": True,
        "role": role,
        "partition_seed": shared["partition_seed"],
        "qualification_contract_sha256": shared["qualification"],
        "source_bundle_sha256": shared["source_bundle"],
        "records": records,
    }


def embedding_bundle(family_id: str, model_id: str, panels: dict[str, dict]) -> dict:
    rows = []
    human_roles = ("distribution_references", "human_floor_a", "human_floor_b")
    for role_index, role in enumerate(human_roles):
        for index, record in enumerate(panels[role]["records"]):
            rows.append(
                {
                    "document_id": record["document_id"],
                    "embedding": [
                        float(index + 1),
                        float(role_index + 1),
                        float((index * (role_index + 1)) % 17),
                    ],
                }
            )
    return {
        "artifact_schema": EMBEDDING_SCHEMA,
        "status": "materialized",
        "family_id": family_id,
        "model_id": model_id,
        "model_revision": ("a" if family_id == "family-a" else "b") * 40,
        "model_artifact_sha256": sha(f"model:{family_id}"),
        "preprocessing_sha256": sha(f"preprocessing:{family_id}"),
        "rows": rows,
    }


def token_contract() -> dict:
    return {
        "artifact_schema": TOKENIZATION_SCHEMA,
        "status": "frozen",
        "tokenizer_id": "Qwen/Qwen3-4B",
        "tokenizer_revision": "c" * 40,
        "tokenizer_artifact_sha256": sha("tokenizer"),
        "implementation": "transformers.AutoTokenizer.encode.v1",
        "input_encoding": "utf8_verbatim",
        "casefold": False,
        "add_special_tokens": False,
        "truncation": "none",
        "aggregation": "corpus_relative_frequency",
    }


def decision_power() -> dict:
    rule = {
        "rule_id": "measurement-v3-intersection-v1",
        "endpoints": DECISION_ENDPOINTS,
        "intersection_required": True,
        "alpha": 0.05,
        "family_effect_boundary": -0.001,
        "token_l2_margin": 0.02,
        "equivalence_margin": 0.05,
        "noninferiority_margin": 0.05,
    }
    null = {
        "scenario": "null",
        "rule_id": rule["rule_id"],
        "trials": 1000,
        "seed": 11,
        "successes": 40,
        "rate": 0.04,
        "decision_vector_sha256": sha("null-decisions"),
    }
    alternative = {
        "scenario": "alternative",
        "rule_id": rule["rule_id"],
        "trials": 1000,
        "seed": 12,
        "successes": 850,
        "rate": 0.85,
        "decision_vector_sha256": sha("alternative-decisions"),
    }
    power = {
        "artifact_schema": "dftr.measurement.exact_decision_power.v3",
        "rule_id": rule["rule_id"],
        "trials_per_scenario": 1000,
        "master_seed": 10,
        "decision_boundary": -0.001,
        "alternative_effect": -0.002,
        "effect_direction": "less",
        "alternative_strictly_beyond_boundary": True,
        "type_i_max": 0.05,
        "power_min": 0.80,
        "null": null,
        "alternative": alternative,
        "type_i_pass": True,
        "power_pass": True,
        "all_targets_pass": True,
    }
    canonical_rule = json.dumps(rule, sort_keys=True, separators=(",", ":"))
    return {
        "artifact_schema": DECISION_POWER_SCHEMA,
        "status": "frozen",
        "candidate_outputs_opened": False,
        "decision_rule": rule,
        "decision_rule_sha256": hashlib.sha256(canonical_rule.encode()).hexdigest(),
        "analysis_code_sha256": sha("analysis"),
        "null_generator_sha256": sha("null-generator"),
        "alternative_generator_sha256": sha("alternative-generator"),
        "training_reward_model_ids": ["Qwen/Qwen3-4B"],
        "power": power,
    }


def positive_controls(
    *, panel_design_sha: str, token_sha: str, family_ids: list[str]
) -> dict:
    controls = {}
    for name, expectation in POSITIVE_CONTROLS.items():
        detected = expectation == "detected"
        pvalue = 0.001 if detected else 0.5
        controls[name] = {
            "expected": expectation,
            "family_pvalues": {family_id: pvalue for family_id in family_ids},
            "token_unigram_l2": 0.02 if detected else 0.001,
            "detected": detected,
        }
    return {
        "artifact_schema": POSITIVE_CONTROL_SCHEMA,
        "status": "qualified",
        "candidate_outputs_opened": False,
        "panel_design_sha256": panel_design_sha,
        "tokenization_contract_sha256": token_sha,
        "embedding_family_ids": family_ids,
        "alpha": 0.05,
        "controls": controls,
    }


def inputs(tmp_path: Path, *, mutations=None):
    shared = {
        "partition_seed": "fresh-v3-partition-v1",
        "qualification": sha("qualification-contract"),
        "source_bundle": sha("source-bundle"),
    }
    panels = {role: panel(role, shared) for role in PANEL_COUNTS}
    if mutations:
        mutations(panels)
    paths = {
        role: write(tmp_path / f"{role}.json", value) for role, value in panels.items()
    }
    family_a = write(
        tmp_path / "family-a.json",
        embedding_bundle("family-a", "BAAI/bge-small-en-v1.5", panels),
    )
    family_b = write(
        tmp_path / "family-b.json",
        embedding_bundle("family-b", "intfloat/e5-base-v2", panels),
    )
    token = write(tmp_path / "tokenization.json", token_contract())
    decision = write(tmp_path / "decision-power.json", decision_power())

    from harness.measurement_v3_operator import _panel_design, _qualified_panel

    loaded_panels = {role: _qualified_panel(paths[role], role) for role in PANEL_COUNTS}
    _, design_artifact = _panel_design(loaded_panels)
    design_path = tmp_path / "expected-panel-design.json"
    write(design_path, design_artifact)
    controls = write(
        tmp_path / "positive-controls.json",
        positive_controls(
            panel_design_sha=file_sha(design_path),
            token_sha=file_sha(token),
            family_ids=["family-a", "family-b"],
        ),
    )
    return {
        **paths,
        "family_a": family_a,
        "family_b": family_b,
        "token": token,
        "decision": decision,
        "controls": controls,
    }


def freeze(tmp_path: Path, source=None):
    source = source or inputs(tmp_path)
    root = tmp_path / "artifact"
    result = freeze_measurement_v3(
        artifact_root=root,
        prompt_manifest=source["prompt_sources"],
        semantic_reference_manifest=source["distribution_references"],
        floor_a_manifest=source["human_floor_a"],
        floor_b_manifest=source["human_floor_b"],
        embedding_family_a=source["family_a"],
        embedding_family_b=source["family_b"],
        tokenization_contract=source["token"],
        decision_power_config=source["decision"],
        positive_controls=source["controls"],
    )
    return root, result


def test_complete_candidate_blind_bundle_freezes(tmp_path: Path) -> None:
    root, result = freeze(tmp_path)
    protocol = json.loads((root / "measurement_protocol_v3.json").read_text())
    bandwidths = json.loads((root / "bandwidths_v3.json").read_text())
    assert result["status"] == "protocol_ready_candidate_unopened"
    assert protocol["candidate_outputs_opened"] is False
    assert protocol["design"] == {
        "prompt_count": 128,
        "distribution_reference_count": 256,
        "human_floor_a_count": 128,
        "human_floor_b_count": 128,
        "prompt_distribution_pairing": "prohibited",
        "treatment_control_pairing": "within_prompt",
    }
    assert set(bandwidths["families"]) == {"family-a", "family-b"}
    assert all(
        item["source"] == "human_floor_a_union_human_floor_b_only"
        for item in bandwidths["families"].values()
    )
    assert protocol["power"]["alternative_strictly_beyond_boundary"] is True
    assert protocol["positive_controls"]["status"] == "qualified"


def test_wrong_panel_cardinality_fails_closed(tmp_path: Path) -> None:
    source = inputs(tmp_path)
    value = json.loads(source["prompt_sources"].read_text())
    value["records"].pop()
    write(source["prompt_sources"], value)
    with pytest.raises(MeasurementV3OperatorError, match="128-record"):
        freeze(tmp_path, source)


def test_cross_panel_source_overlap_fails_closed(tmp_path: Path) -> None:
    source = inputs(tmp_path)
    prompt_value = json.loads(source["prompt_sources"].read_text())
    reference_value = json.loads(source["distribution_references"].read_text())
    reference_value["records"][0]["source_document_id"] = prompt_value["records"][0][
        "source_document_id"
    ]
    write(source["distribution_references"], reference_value)
    with pytest.raises(MeasurementV3OperatorError, match="overlap"):
        freeze(tmp_path, source)


def test_embedding_families_must_be_distinct_and_exact(tmp_path: Path) -> None:
    source = inputs(tmp_path)
    second = json.loads(source["family_b"].read_text())
    first = json.loads(source["family_a"].read_text())
    second["model_id"] = first["model_id"]
    write(source["family_b"], second)
    with pytest.raises(MeasurementV3OperatorError, match="genuinely distinct"):
        freeze(tmp_path, source)

    other = tmp_path / "other"
    other.mkdir()
    source = inputs(other)
    second = json.loads(source["family_b"].read_text())
    second["rows"].pop()
    write(source["family_b"], second)
    with pytest.raises(MeasurementV3OperatorError, match="exact human panels"):
        freeze(other, source)


def test_positive_control_failure_blocks_freeze(tmp_path: Path) -> None:
    source = inputs(tmp_path)
    controls = json.loads(source["controls"].read_text())
    controls["controls"]["base_model_vs_unpaired_humans"]["family_pvalues"][
        "family-a"
    ] = 0.5
    controls["controls"]["base_model_vs_unpaired_humans"]["detected"] = False
    write(source["controls"], controls)
    with pytest.raises(MeasurementV3OperatorError, match="expected behavior"):
        freeze(tmp_path, source)

    other = tmp_path / "other"
    other.mkdir()
    source = inputs(other)
    controls = json.loads(source["controls"].read_text())
    controls["controls"]["human_vs_human_null"]["family_pvalues"]["family-a"] = 0.001
    write(source["controls"], controls)
    with pytest.raises(MeasurementV3OperatorError, match="expected behavior"):
        freeze(other, source)


def test_training_reward_models_cannot_be_evaluation_embedders(
    tmp_path: Path,
) -> None:
    source = inputs(tmp_path)
    value = json.loads(source["decision"].read_text())
    value["training_reward_model_ids"] = ["intfloat/e5-base-v2"]
    write(source["decision"], value)
    with pytest.raises(MeasurementV3OperatorError, match="training reward model"):
        freeze(tmp_path, source)
    assert not (tmp_path / "artifact").exists()


def test_power_must_use_exact_rule_beyond_boundary(tmp_path: Path) -> None:
    source = inputs(tmp_path)
    value = json.loads(source["decision"].read_text())
    value["power"]["alternative_effect"] = value["power"]["decision_boundary"]
    write(source["decision"], value)
    with pytest.raises(MeasurementV3OperatorError, match="decision boundary"):
        freeze(tmp_path, source)
    assert not (tmp_path / "artifact").exists()

    other = tmp_path / "other"
    other.mkdir()
    source = inputs(other)
    value = json.loads(source["decision"].read_text())
    value["power"]["alternative"]["successes"] = 1
    write(source["decision"], value)
    with pytest.raises(MeasurementV3OperatorError, match="arithmetic"):
        freeze(other, source)


def test_candidate_opened_flags_and_nonempty_root_are_rejected(tmp_path: Path) -> None:
    source = inputs(tmp_path)
    value = json.loads(source["decision"].read_text())
    value["candidate_outputs_opened"] = True
    write(source["decision"], value)
    with pytest.raises(MeasurementV3OperatorError, match="decision rule"):
        freeze(tmp_path, source)

    other = tmp_path / "other"
    other.mkdir()
    source = inputs(other)
    root = other / "artifact"
    root.mkdir()
    (root / "existing").write_text("occupied")
    with pytest.raises(MeasurementV3OperatorError, match="absent or empty"):
        freeze_measurement_v3(
            artifact_root=root,
            prompt_manifest=source["prompt_sources"],
            semantic_reference_manifest=source["distribution_references"],
            floor_a_manifest=source["human_floor_a"],
            floor_b_manifest=source["human_floor_b"],
            embedding_family_a=source["family_a"],
            embedding_family_b=source["family_b"],
            tokenization_contract=source["token"],
            decision_power_config=source["decision"],
            positive_controls=source["controls"],
        )
