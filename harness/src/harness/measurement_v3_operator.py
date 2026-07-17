"""Candidate-blind materialization of a fresh measurement-v3 protocol.

Only already-qualified, fixed-cardinality panel manifests are accepted.  This
operator binds two independent human-only embedding geometries, the token
unigram contract, the exact decision/power contract, and manipulation checks
before any candidate output exists.  It is intentionally independent of v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Sequence

import numpy as np

from .measurement_v3 import (
    MeasurementV3Error,
    UnpairedPanelDesign,
    human_floor_bandwidths,
)


PANEL_SCHEMA = "dftr.measurement.qualified_panel.v3"
EMBEDDING_SCHEMA = "dftr.measurement.embedding_family.v3"
TOKENIZATION_SCHEMA = "dftr.measurement.tokenization_contract.v3"
DECISION_POWER_SCHEMA = "dftr.measurement.decision_power_config.v3"
POSITIVE_CONTROL_SCHEMA = "dftr.measurement.positive_controls.v3"
PROTOCOL_SCHEMA = "dftr.measurement.protocol.v3"

PANEL_COUNTS = {
    "prompt_sources": 128,
    "distribution_references": 256,
    "human_floor_a": 128,
    "human_floor_b": 128,
}
POSITIVE_CONTROLS = {
    "human_vs_human_null": "not_detected",
    "prefix64_vs_full_human": "detected",
    "sft_vs_unpaired_humans": "detected",
    "base_model_vs_unpaired_humans": "detected",
}
DECISION_ENDPOINTS = [
    "embedding_family_a_paired_mmd",
    "embedding_family_b_paired_mmd",
    "embedding_family_direction_agreement",
    "token_unigram_l2",
    "human_calibrated_equivalence",
    "human_calibrated_noninferiority",
    "positive_control_qualification",
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


class MeasurementV3OperatorError(MeasurementV3Error):
    """Raised when prospective v3 evidence cannot be frozen."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _pretty_sha(value: Any) -> str:
    return hashlib.sha256(_pretty_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    text = str(value or "")
    if SHA256_RE.fullmatch(text) is None:
        raise MeasurementV3OperatorError(f"{label} must be a lowercase SHA-256")
    return text


def _load(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise MeasurementV3OperatorError(f"input must be a regular file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasurementV3OperatorError(
            f"invalid JSON input {source}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise MeasurementV3OperatorError(f"JSON input must be an object: {source}")
    return value


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pretty_bytes(value))
    return _file_sha(path)


def _copy(source: str | Path, target: Path) -> str:
    path = Path(source)
    if not path.is_file() or path.is_symlink():
        raise MeasurementV3OperatorError(f"bound input must be a regular file: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return _file_sha(target)


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise MeasurementV3OperatorError(f"{label} schema mismatch")
    return value


def _qualified_panel(path: str | Path, expected_role: str) -> dict[str, Any]:
    value = _exact(
        _load(path),
        {
            "artifact_schema",
            "status",
            "frozen",
            "role",
            "partition_seed",
            "qualification_contract_sha256",
            "source_bundle_sha256",
            "records",
        },
        f"{expected_role} manifest",
    )
    expected_count = PANEL_COUNTS[expected_role]
    records = value.get("records")
    if (
        value.get("artifact_schema") != PANEL_SCHEMA
        or value.get("status") != "qualified"
        or value.get("frozen") is not True
        or value.get("role") != expected_role
        or not isinstance(value.get("partition_seed"), str)
        or not value["partition_seed"]
        or not isinstance(records, list)
        or len(records) != expected_count
    ):
        raise MeasurementV3OperatorError(
            f"{expected_role} must be a frozen qualified {expected_count}-record manifest"
        )
    _sha(value.get("qualification_contract_sha256"), "qualification contract")
    _sha(value.get("source_bundle_sha256"), "source bundle")
    expected_record_keys = (
        {"prompt_id", "source_document_id", "content_sha256", "domain"}
        if expected_role == "prompt_sources"
        else {"document_id", "source_document_id", "content_sha256", "domain"}
    )
    primary_key = "prompt_id" if expected_role == "prompt_sources" else "document_id"
    seen_primary: set[str] = set()
    seen_source: set[str] = set()
    seen_content: set[str] = set()
    seen_domain: set[str] = set()
    for record in records:
        row = _exact(record, expected_record_keys, f"{expected_role} record")
        primary = str(row.get(primary_key) or "")
        source_id = _sha(row.get("source_document_id"), "source document ID")
        content = _sha(row.get("content_sha256"), "content identity")
        domain = str(row.get("domain") or "").casefold()
        if (
            not primary
            or not domain
            or primary in seen_primary
            or source_id in seen_source
            or content in seen_content
            or domain in seen_domain
        ):
            raise MeasurementV3OperatorError(
                f"{expected_role} records require unique IDs, content, source, and domain"
            )
        if expected_role != "prompt_sources" and primary != content:
            raise MeasurementV3OperatorError(
                f"{expected_role} document IDs must equal content SHA-256"
            )
        seen_primary.add(primary)
        seen_source.add(source_id)
        seen_content.add(content)
        seen_domain.add(domain)
    return value


def _panel_design(
    panels: dict[str, dict[str, Any]],
) -> tuple[UnpairedPanelDesign, dict[str, Any]]:
    seeds = {panel["partition_seed"] for panel in panels.values()}
    qualification_hashes = {
        panel["qualification_contract_sha256"] for panel in panels.values()
    }
    source_hashes = {panel["source_bundle_sha256"] for panel in panels.values()}
    if len(seeds) != 1 or len(qualification_hashes) != 1 or len(source_hashes) != 1:
        raise MeasurementV3OperatorError(
            "all four panels must share one partition, qualification, and source bundle"
        )
    prompt_rows = panels["prompt_sources"]["records"]
    reference_rows = panels["distribution_references"]["records"]
    floor_a_rows = panels["human_floor_a"]["records"]
    floor_b_rows = panels["human_floor_b"]["records"]
    try:
        design = UnpairedPanelDesign.build(
            prompt_ids=[row["prompt_id"] for row in prompt_rows],
            prompt_source_document_ids=[
                row["source_document_id"] for row in prompt_rows
            ],
            distribution_reference_ids=[row["document_id"] for row in reference_rows],
            human_floor_a_ids=[row["document_id"] for row in floor_a_rows],
            human_floor_b_ids=[row["document_id"] for row in floor_b_rows],
        )
    except MeasurementV3Error as error:
        raise MeasurementV3OperatorError(str(error)) from error
    all_source_ids = [
        row["source_document_id"]
        for panel in panels.values()
        for row in panel["records"]
    ]
    all_content_ids = [
        row["content_sha256"] for panel in panels.values() for row in panel["records"]
    ]
    all_domains = [
        row["domain"].casefold()
        for panel in panels.values()
        for row in panel["records"]
    ]
    if (
        len(all_source_ids) != len(set(all_source_ids))
        or len(all_content_ids) != len(set(all_content_ids))
        or len(all_domains) != len(set(all_domains))
    ):
        raise MeasurementV3OperatorError(
            "the four qualified panels overlap by source, content, or domain"
        )
    materialized = {
        "artifact_schema": "dftr.measurement.panel_design.v3",
        "status": "frozen",
        "candidate_outputs_opened": False,
        "partition_seed": seeds.pop(),
        "qualification_contract_sha256": qualification_hashes.pop(),
        "source_bundle_sha256": source_hashes.pop(),
        "counts": dict(PANEL_COUNTS),
        "panel_identity_sha256": design.identity_sha256,
        "roles": {
            "prompt_sources": "quality_and_adherence_only",
            "distribution_references": "unpaired_semantic_reference",
            "human_floor_a": "human_only_kernel_and_calibration",
            "human_floor_b": "human_only_kernel_and_calibration",
        },
    }
    return design, materialized


def _embedding_bundle(
    path: str | Path,
    expected_ids: set[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    value = _exact(
        _load(path),
        {
            "artifact_schema",
            "status",
            "family_id",
            "model_id",
            "model_revision",
            "model_artifact_sha256",
            "preprocessing_sha256",
            "rows",
        },
        "embedding-family bundle",
    )
    revision = str(value.get("model_revision") or "")
    if (
        value.get("artifact_schema") != EMBEDDING_SCHEMA
        or value.get("status") != "materialized"
        or not str(value.get("family_id") or "")
        or not str(value.get("model_id") or "")
        or REVISION_RE.fullmatch(revision) is None
    ):
        raise MeasurementV3OperatorError(
            "embedding family identity is incomplete or mutable"
        )
    _sha(value.get("model_artifact_sha256"), "embedding model artifact")
    _sha(value.get("preprocessing_sha256"), "embedding preprocessing")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise MeasurementV3OperatorError(
            "embedding family does not cover exact human panels"
        )
    vectors: dict[str, np.ndarray] = {}
    dimension: int | None = None
    for row in rows:
        item = _exact(row, {"document_id", "embedding"}, "embedding row")
        document_id = str(item.get("document_id") or "")
        vector = np.asarray(item.get("embedding"), dtype=np.float64)
        if (
            document_id not in expected_ids
            or document_id in vectors
            or vector.ndim != 1
            or vector.size < 1
            or not np.isfinite(vector).all()
        ):
            raise MeasurementV3OperatorError(
                "embedding family row is invalid or duplicated"
            )
        dimension = vector.size if dimension is None else dimension
        if vector.size != dimension:
            raise MeasurementV3OperatorError(
                "embedding dimensions differ within a family"
            )
        vectors[document_id] = vector
    if set(vectors) != expected_ids:
        raise MeasurementV3OperatorError("embedding family human ID set is not exact")
    return value, vectors


def _tokenization_contract(path: str | Path) -> dict[str, Any]:
    value = _exact(
        _load(path),
        {
            "artifact_schema",
            "status",
            "tokenizer_id",
            "tokenizer_revision",
            "tokenizer_artifact_sha256",
            "implementation",
            "input_encoding",
            "casefold",
            "add_special_tokens",
            "truncation",
            "aggregation",
        },
        "tokenization contract",
    )
    if (
        value.get("artifact_schema") != TOKENIZATION_SCHEMA
        or value.get("status") != "frozen"
        or not str(value.get("tokenizer_id") or "")
        or REVISION_RE.fullmatch(str(value.get("tokenizer_revision") or "")) is None
        or not str(value.get("implementation") or "")
        or value.get("input_encoding") != "utf8_verbatim"
        or type(value.get("casefold")) is not bool
        or value.get("add_special_tokens") is not False
        or value.get("truncation") != "none"
        or value.get("aggregation") != "corpus_relative_frequency"
    ):
        raise MeasurementV3OperatorError(
            "token unigram contract is incomplete or mutable"
        )
    _sha(value.get("tokenizer_artifact_sha256"), "tokenizer artifact")
    return value


def _decision_power_config(path: str | Path) -> dict[str, Any]:
    value = _exact(
        _load(path),
        {
            "artifact_schema",
            "status",
            "candidate_outputs_opened",
            "decision_rule",
            "decision_rule_sha256",
            "analysis_code_sha256",
            "null_generator_sha256",
            "alternative_generator_sha256",
            "training_reward_model_ids",
            "power",
        },
        "decision/power config",
    )
    rule = _exact(
        value.get("decision_rule"),
        {
            "rule_id",
            "endpoints",
            "intersection_required",
            "alpha",
            "family_effect_boundary",
            "token_l2_margin",
            "equivalence_margin",
            "noninferiority_margin",
        },
        "decision rule",
    )
    numeric_rule_fields = (
        "alpha",
        "family_effect_boundary",
        "token_l2_margin",
        "equivalence_margin",
        "noninferiority_margin",
    )
    if (
        value.get("artifact_schema") != DECISION_POWER_SCHEMA
        or value.get("status") != "frozen"
        or value.get("candidate_outputs_opened") is not False
        or not str(rule.get("rule_id") or "")
        or rule.get("endpoints") != DECISION_ENDPOINTS
        or rule.get("intersection_required") is not True
        or any(
            not isinstance(rule.get(field), (int, float))
            or isinstance(rule.get(field), bool)
            or not math.isfinite(float(rule[field]))
            for field in numeric_rule_fields
        )
        or not 0 < float(rule["alpha"]) <= 0.05
        or any(float(rule[field]) < 0 for field in numeric_rule_fields[2:])
        or value.get("decision_rule_sha256") != _canonical_sha(rule)
    ):
        raise MeasurementV3OperatorError(
            "decision rule is incomplete or not hash-bound"
        )
    for field in (
        "analysis_code_sha256",
        "null_generator_sha256",
        "alternative_generator_sha256",
    ):
        _sha(value.get(field), field)
    reward_models = value.get("training_reward_model_ids")
    if (
        not isinstance(reward_models, list)
        or not reward_models
        or any(not isinstance(item, str) or not item for item in reward_models)
        or len(reward_models) != len(set(reward_models))
    ):
        raise MeasurementV3OperatorError(
            "training reward model identities must be frozen and unique"
        )
    power = _exact(
        value.get("power"),
        {
            "artifact_schema",
            "rule_id",
            "trials_per_scenario",
            "master_seed",
            "decision_boundary",
            "alternative_effect",
            "effect_direction",
            "alternative_strictly_beyond_boundary",
            "type_i_max",
            "power_min",
            "null",
            "alternative",
            "type_i_pass",
            "power_pass",
            "all_targets_pass",
        },
        "exact decision power",
    )
    trials = power.get("trials_per_scenario")
    direction = power.get("effect_direction")
    boundary = power.get("decision_boundary")
    effect = power.get("alternative_effect")
    beyond = (
        isinstance(boundary, (int, float))
        and not isinstance(boundary, bool)
        and isinstance(effect, (int, float))
        and not isinstance(effect, bool)
        and math.isfinite(float(boundary))
        and math.isfinite(float(effect))
        and (
            (direction == "greater" and float(effect) > float(boundary))
            or (direction == "less" and float(effect) < float(boundary))
        )
    )
    if (
        power.get("artifact_schema") != "dftr.measurement.exact_decision_power.v3"
        or power.get("rule_id") != rule["rule_id"]
        or not isinstance(trials, int)
        or isinstance(trials, bool)
        or trials < 1_000
        or power.get("alternative_strictly_beyond_boundary") is not True
        or not beyond
        or float(boundary) != float(rule["family_effect_boundary"])
    ):
        raise MeasurementV3OperatorError(
            "power does not exercise the frozen decision boundary"
        )
    for scenario_name in ("null", "alternative"):
        scenario = _exact(
            power.get(scenario_name),
            {
                "scenario",
                "rule_id",
                "trials",
                "seed",
                "successes",
                "rate",
                "decision_vector_sha256",
            },
            f"power {scenario_name} scenario",
        )
        successes = scenario.get("successes")
        if (
            scenario.get("scenario") != scenario_name
            or scenario.get("rule_id") != rule["rule_id"]
            or scenario.get("trials") != trials
            or not isinstance(successes, int)
            or isinstance(successes, bool)
            or not 0 <= successes <= trials
            or scenario.get("rate") != successes / trials
        ):
            raise MeasurementV3OperatorError(
                f"power {scenario_name} arithmetic is invalid"
            )
        _sha(scenario.get("decision_vector_sha256"), f"power {scenario_name} decisions")
    type_i_max, power_min = power.get("type_i_max"), power.get("power_min")
    if (
        not isinstance(type_i_max, (int, float))
        or isinstance(type_i_max, bool)
        or not 0 <= float(type_i_max) <= 0.05
        or float(type_i_max) != float(rule["alpha"])
        or not isinstance(power_min, (int, float))
        or isinstance(power_min, bool)
        or not 0.8 <= float(power_min) <= 1
        or power["type_i_pass"] is not (power["null"]["rate"] <= float(type_i_max))
        or power["power_pass"] is not (power["alternative"]["rate"] >= float(power_min))
        or power["all_targets_pass"]
        is not (power["type_i_pass"] and power["power_pass"])
        or power["all_targets_pass"] is not True
    ):
        raise MeasurementV3OperatorError("exact decision power targets did not qualify")
    return value


def _positive_controls(
    path: str | Path,
    *,
    family_ids: set[str],
    panel_design_sha256: str,
    tokenization_sha256: str,
    alpha: float,
) -> dict[str, Any]:
    value = _exact(
        _load(path),
        {
            "artifact_schema",
            "status",
            "candidate_outputs_opened",
            "panel_design_sha256",
            "tokenization_contract_sha256",
            "embedding_family_ids",
            "alpha",
            "controls",
        },
        "positive controls",
    )
    if (
        value.get("artifact_schema") != POSITIVE_CONTROL_SCHEMA
        or value.get("status") != "qualified"
        or value.get("candidate_outputs_opened") is not False
        or value.get("panel_design_sha256") != panel_design_sha256
        or value.get("tokenization_contract_sha256") != tokenization_sha256
        or set(value.get("embedding_family_ids") or []) != family_ids
        or value.get("alpha") != alpha
        or not isinstance(value.get("controls"), dict)
        or set(value["controls"]) != set(POSITIVE_CONTROLS)
    ):
        raise MeasurementV3OperatorError("positive-control bindings are incomplete")
    for name, expectation in POSITIVE_CONTROLS.items():
        control = _exact(
            value["controls"].get(name),
            {"expected", "family_pvalues", "token_unigram_l2", "detected"},
            f"positive control {name}",
        )
        pvalues = control.get("family_pvalues")
        if (
            control.get("expected") != expectation
            or not isinstance(pvalues, dict)
            or set(pvalues) != family_ids
            or any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not 0 <= float(item) <= 1
                for item in pvalues.values()
            )
            or not isinstance(control.get("token_unigram_l2"), (int, float))
            or isinstance(control.get("token_unigram_l2"), bool)
            or not math.isfinite(float(control["token_unigram_l2"]))
            or float(control["token_unigram_l2"]) < 0
        ):
            raise MeasurementV3OperatorError(f"positive control {name} is malformed")
        expected_detected = expectation == "detected"
        if expected_detected:
            reproduced = all(float(item) <= alpha for item in pvalues.values())
        else:
            reproduced = all(float(item) > alpha for item in pvalues.values())
        if control.get("detected") is not expected_detected or not reproduced:
            raise MeasurementV3OperatorError(
                f"positive control {name} did not reproduce its expected behavior"
            )
    return value


def freeze_measurement_v3(
    *,
    artifact_root: str | Path,
    prompt_manifest: str | Path,
    semantic_reference_manifest: str | Path,
    floor_a_manifest: str | Path,
    floor_b_manifest: str | Path,
    embedding_family_a: str | Path,
    embedding_family_b: str | Path,
    tokenization_contract: str | Path,
    decision_power_config: str | Path,
    positive_controls: str | Path,
) -> dict[str, Any]:
    """Freeze a candidate-blind v3 protocol from complete qualified evidence."""
    root = Path(artifact_root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise MeasurementV3OperatorError("artifact root must be absent or empty")
    panels = {
        "prompt_sources": _qualified_panel(prompt_manifest, "prompt_sources"),
        "distribution_references": _qualified_panel(
            semantic_reference_manifest, "distribution_references"
        ),
        "human_floor_a": _qualified_panel(floor_a_manifest, "human_floor_a"),
        "human_floor_b": _qualified_panel(floor_b_manifest, "human_floor_b"),
    }
    design, panel_design = _panel_design(panels)
    panel_design_path = root / "panel_design_v3.json"
    panel_design_sha = _pretty_sha(panel_design)

    human_ids = (
        set(design.distribution_reference_ids)
        | set(design.human_floor_a_ids)
        | set(design.human_floor_b_ids)
    )
    family_inputs = (embedding_family_a, embedding_family_b)
    family_values: list[dict[str, Any]] = []
    family_vectors: list[dict[str, np.ndarray]] = []
    for source in family_inputs:
        value, vectors = _embedding_bundle(source, human_ids)
        family_values.append(value)
        family_vectors.append(vectors)
    family_ids = {value["family_id"] for value in family_values}
    model_ids = {value["model_id"] for value in family_values}
    model_artifacts = {value["model_artifact_sha256"] for value in family_values}
    if len(family_ids) != 2 or len(model_ids) != 2 or len(model_artifacts) != 2:
        raise MeasurementV3OperatorError(
            "two genuinely distinct embedding families and model artifacts are required"
        )

    bandwidth_families: dict[str, Any] = {}
    for value, vectors in zip(family_values, family_vectors):
        floor_a = np.asarray([vectors[item] for item in design.human_floor_a_ids])
        floor_b = np.asarray([vectors[item] for item in design.human_floor_b_ids])
        bandwidths = human_floor_bandwidths(floor_a, floor_b)
        bandwidth_families[value["family_id"]] = {
            "model_id": value["model_id"],
            "model_revision": value["model_revision"],
            "model_artifact_sha256": value["model_artifact_sha256"],
            "preprocessing_sha256": value["preprocessing_sha256"],
            "source": "human_floor_a_union_human_floor_b_only",
            "values": list(bandwidths),
            "value_sha256": _canonical_sha(
                [format(item, ".17g") for item in bandwidths]
            ),
        }
    bandwidth_artifact = {
        "artifact_schema": "dftr.measurement.bandwidths.v3",
        "status": "frozen",
        "candidate_outputs_opened": False,
        "panel_design_sha256": panel_design_sha,
        "families": bandwidth_families,
    }
    bandwidth_path = root / "bandwidths_v3.json"
    bandwidth_sha = _pretty_sha(bandwidth_artifact)

    _tokenization_contract(tokenization_contract)
    decision_value = _decision_power_config(decision_power_config)
    if model_ids & set(decision_value["training_reward_model_ids"]):
        raise MeasurementV3OperatorError(
            "evaluation embedding families cannot reuse a training reward model"
        )
    token_input_sha = _file_sha(Path(tokenization_contract))
    controls_value = _positive_controls(
        positive_controls,
        family_ids=family_ids,
        panel_design_sha256=panel_design_sha,
        tokenization_sha256=token_input_sha,
        alpha=float(decision_value["decision_rule"]["alpha"]),
    )

    root.mkdir(parents=True, exist_ok=True)
    if _write(panel_design_path, panel_design) != panel_design_sha:
        raise MeasurementV3OperatorError(
            "panel-design serialization changed during freeze"
        )
    if _write(bandwidth_path, bandwidth_artifact) != bandwidth_sha:
        raise MeasurementV3OperatorError(
            "bandwidth serialization changed during freeze"
        )

    input_paths = {
        "prompt_manifest": Path(prompt_manifest),
        "semantic_reference_manifest": Path(semantic_reference_manifest),
        "floor_a_manifest": Path(floor_a_manifest),
        "floor_b_manifest": Path(floor_b_manifest),
        "embedding_family_a": Path(embedding_family_a),
        "embedding_family_b": Path(embedding_family_b),
        "tokenization_contract": Path(tokenization_contract),
        "decision_power_config": Path(decision_power_config),
        "positive_controls": Path(positive_controls),
    }
    copied: dict[str, dict[str, str]] = {}
    for name, source in input_paths.items():
        target = root / "inputs" / f"{name}{source.suffix or '.json'}"
        copied[name] = {
            "path": target.relative_to(root).as_posix(),
            "sha256": _copy(source, target),
        }
    hashes = {f"{name}_sha256": binding["sha256"] for name, binding in copied.items()}
    hashes.update(
        panel_design_sha256=panel_design_sha,
        bandwidths_sha256=bandwidth_sha,
    )
    protocol = {
        "artifact_schema": PROTOCOL_SCHEMA,
        "status": "ready",
        "frozen": True,
        "candidate_outputs_opened": False,
        "design": {
            "prompt_count": PANEL_COUNTS["prompt_sources"],
            "distribution_reference_count": PANEL_COUNTS["distribution_references"],
            "human_floor_a_count": PANEL_COUNTS["human_floor_a"],
            "human_floor_b_count": PANEL_COUNTS["human_floor_b"],
            "prompt_distribution_pairing": "prohibited",
            "treatment_control_pairing": "within_prompt",
        },
        "panel_design_sha256": panel_design_sha,
        "embedding_families": [
            {
                "family_id": value["family_id"],
                "model_id": value["model_id"],
                "model_revision": value["model_revision"],
                "model_artifact_sha256": value["model_artifact_sha256"],
                "preprocessing_sha256": value["preprocessing_sha256"],
            }
            for value in family_values
        ],
        "bandwidths_sha256": bandwidth_sha,
        "tokenization_contract_sha256": copied["tokenization_contract"]["sha256"],
        "decision_rule_sha256": decision_value["decision_rule_sha256"],
        "decision_power_config_sha256": copied["decision_power_config"]["sha256"],
        "positive_controls_sha256": copied["positive_controls"]["sha256"],
        "positive_controls": {
            "status": controls_value["status"],
            "required": sorted(POSITIVE_CONTROLS),
        },
        "power": {
            "status": "qualified",
            "rule_id": decision_value["decision_rule"]["rule_id"],
            "trials_per_scenario": decision_value["power"]["trials_per_scenario"],
            "type_i_pass": True,
            "power_pass": True,
            "alternative_strictly_beyond_boundary": True,
        },
        "hashes": hashes,
        "artifact_bindings": {
            **copied,
            "panel_design": {
                "path": panel_design_path.relative_to(root).as_posix(),
                "sha256": panel_design_sha,
            },
            "bandwidths": {
                "path": bandwidth_path.relative_to(root).as_posix(),
                "sha256": bandwidth_sha,
            },
        },
    }
    protocol_path = root / "measurement_protocol_v3.json"
    protocol_sha = _write(protocol_path, protocol)
    status = {
        "artifact_schema": "dftr.measurement.operator_materialization.v3",
        "status": "protocol_ready_candidate_unopened",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "panel_design_sha256": panel_design_sha,
        "embedding_family_ids": sorted(family_ids),
        "candidate_outputs_opened": False,
    }
    _write(root / "materialization_status_v3.json", status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.measurement_v3_operator")
    for field in (
        "artifact-root",
        "prompt-manifest",
        "semantic-reference-manifest",
        "floor-a-manifest",
        "floor-b-manifest",
        "embedding-family-a",
        "embedding-family-b",
        "tokenization-contract",
        "decision-power-config",
        "positive-controls",
    ):
        parser.add_argument(f"--{field}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = freeze_measurement_v3(
            artifact_root=args.artifact_root,
            prompt_manifest=args.prompt_manifest,
            semantic_reference_manifest=args.semantic_reference_manifest,
            floor_a_manifest=args.floor_a_manifest,
            floor_b_manifest=args.floor_b_manifest,
            embedding_family_a=args.embedding_family_a,
            embedding_family_b=args.embedding_family_b,
            tokenization_contract=args.tokenization_contract,
            decision_power_config=args.decision_power_config,
            positive_controls=args.positive_controls,
        )
    except (MeasurementV3Error, OSError, ValueError) as error:
        print(f"measurement-v3-operator: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
