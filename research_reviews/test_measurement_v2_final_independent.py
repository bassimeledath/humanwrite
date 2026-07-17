"""Final independent checks for measurement-v2 hard-gate evidence binding."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from harness.measurement_v2 import validate_report_v2
from harness.metrics.distribution_v2 import MeasurementV2Error


ROOT = Path(__file__).resolve().parents[1]
PRIOR = runpy.run_path(
    str(ROOT / "research_reviews" / "test_measurement_v2_semantic_repair_independent.py")
)
synthetic_evidence = PRIOR["synthetic_evidence"]
build_promotion_report = PRIOR["build_promotion_report"]
resign = PRIOR["resign"]


def validate(root: Path, protocol: dict, report: dict, trusted: dict[str, str]):
    return validate_report_v2(
        report,
        protocol=protocol,
        artifact_root=root,
        trusted_public_keys=trusted,
    )


def replace_evidence(
    root: Path,
    report: dict,
    private_key,
    raw: str,
    *,
    gate_name: str = "factuality",
) -> None:
    path = root / f"gate-{gate_name}.json"
    path.write_text(raw)
    report["hard_gates"][gate_name]["evidence_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    resign(report, private_key)


def test_final_valid_signed_promotion_case_passes(tmp_path: Path) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=False
    )
    assert validate(tmp_path, protocol, report, trusted)["status"] == "pass"


@pytest.mark.parametrize(
    "raw",
    [
        '{"artifact_schema":"dftr.gate.factuality.v1","name":"factuality",'
        '"name":"brief_adherence","decision":"pass"}\n',
        '{"artifact_schema":"dftr.gate.factuality.v1","name":"factuality",'
        '"decision":"pass","extra":true}\n',
        '{"artifact_schema":"dftr.gate.factuality.v0","name":"factuality",'
        '"decision":"pass"}\n',
        '{"artifact_schema":"dftr.gate.factuality.v1","name":"brief_adherence",'
        '"decision":"pass"}\n',
        '{"artifact_schema":"dftr.gate.factuality.v1","name":"factuality",'
        '"decision":"fail"}\n',
    ],
)
def test_final_gate_json_is_exact_and_semantically_bound(
    tmp_path: Path, raw: str
) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=False
    )
    replace_evidence(tmp_path, report, private_key, raw)
    with pytest.raises(MeasurementV2Error, match="hard gate evidence"):
        validate(tmp_path, protocol, report, trusted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda gate: gate.update(version="dftr.gate.factuality.v0"),
        lambda gate: gate.update(decision="fail"),
        lambda gate: gate.update(unbound_extra="value"),
    ],
)
def test_final_signed_report_gate_entry_is_exact(
    tmp_path: Path, mutation
) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=False
    )
    mutation(report["hard_gates"]["factuality"])
    resign(report, private_key)
    with pytest.raises(MeasurementV2Error, match="hard gate"):
        validate(tmp_path, protocol, report, trusted)


def test_final_gate_paths_and_byte_identities_are_distinct(tmp_path: Path) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=False
    )
    report["hard_gates"]["brief_adherence"] = copy.deepcopy(
        report["hard_gates"]["factuality"]
    )
    report["hard_gates"]["brief_adherence"].update(
        version="dftr.gate.brief_adherence.v1"
    )
    resign(report, private_key)
    with pytest.raises(MeasurementV2Error, match="path is reused"):
        validate(tmp_path, protocol, report, trusted)

    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=False
    )
    source = tmp_path / report["hard_gates"]["factuality"]["evidence_path"]
    copied = tmp_path / "gate-brief-copy.json"
    copied.write_bytes(source.read_bytes())
    report["hard_gates"]["brief_adherence"].update(
        evidence_path=copied.name,
        evidence_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
    )
    resign(report, private_key)
    with pytest.raises(MeasurementV2Error, match="identity is reused"):
        validate(tmp_path, protocol, report, trusted)


def test_final_protocol_report_and_candidate_output_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    protocol, private_key, trusted = synthetic_evidence(tmp_path)
    report = build_promotion_report(
        tmp_path, protocol, private_key, placeholder_gate_evidence=False
    )

    protocol_changed = copy.deepcopy(protocol)
    protocol_changed["operator_approval"]["reviewed_at"] = "2026-07-18T00:00:00Z"
    resign(protocol_changed, private_key)
    with pytest.raises(MeasurementV2Error, match="protocol identity"):
        validate(tmp_path, protocol_changed, report, trusted)

    report_changed = copy.deepcopy(report)
    report_changed["hard_gates"]["factuality"]["decision"] = "fail"
    with pytest.raises(MeasurementV2Error, match="signature"):
        validate(tmp_path, protocol, report_changed, trusted)

    (tmp_path / "candidate-outputs.jsonl").write_text("post-signature swap\n")
    with pytest.raises(MeasurementV2Error, match="candidate output"):
        validate(tmp_path, protocol, report, trusted)
