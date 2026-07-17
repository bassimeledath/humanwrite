import hashlib
import json

import pytest

from harness.measurement_v2 import (
    REQUIRED_HARD_GATE_SCHEMAS,
    _validate_hard_gate_evidence_set,
)
from harness.metrics.distribution_v2 import MeasurementV2Error


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def write_evidence(root, name, value):
    path = root / f"gate-{name}.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def valid_gate_set(root):
    gates = {}
    for name, version in REQUIRED_HARD_GATE_SCHEMAS.items():
        path, digest = write_evidence(root, name, {
            "artifact_schema": version,
            "name": name,
            "decision": "pass",
        })
        gates[name] = {
            "version": version,
            "decision": "pass",
            "evidence_path": path.name,
            "evidence_sha256": digest,
        }
    return gates


def validate(root, gates, required_gates=REQUIRED_HARD_GATE_SCHEMAS):
    return _validate_hard_gate_evidence_set(
        required_gates=required_gates,
        hard_gates=gates,
        artifact_root=root,
        protocol_sha256=SHA_A,
        candidate_output_sha256=SHA_B,
        signed_report_payload_sha256=SHA_C,
    )


def test_exact_distinct_hard_gate_evidence_binds_signed_subjects(tmp_path):
    bindings = validate(tmp_path, valid_gate_set(tmp_path))
    assert set(bindings) == set(REQUIRED_HARD_GATE_SCHEMAS)
    assert len(set(bindings.values())) == len(REQUIRED_HARD_GATE_SCHEMAS)


def test_hard_gates_cannot_reuse_resolved_path_or_byte_identity(tmp_path):
    gates = valid_gate_set(tmp_path)
    factuality = gates["factuality"]
    gates["brief_adherence"].update({
        "evidence_path": factuality["evidence_path"],
        "evidence_sha256": factuality["evidence_sha256"],
    })
    with pytest.raises(MeasurementV2Error, match="hard gate evidence path is reused"):
        validate(tmp_path, gates)

    gates = valid_gate_set(tmp_path)
    source = tmp_path / gates["factuality"]["evidence_path"]
    copied = tmp_path / "gate-brief-copy.json"
    copied.write_bytes(source.read_bytes())
    gates["brief_adherence"].update({
        "evidence_path": copied.name,
        "evidence_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
    })
    with pytest.raises(MeasurementV2Error, match="hard gate evidence identity is reused"):
        validate(tmp_path, gates)


@pytest.mark.parametrize(
    "evidence",
    [
        "pass\n",
        '{"artifact_schema":"dftr.gate.factuality.v1","name":"factuality",'
        '"name":"brief_adherence","decision":"pass"}\n',
        {"name": "factuality", "decision": "pass"},
        {
            "artifact_schema": "dftr.gate.factuality.v0",
            "name": "factuality",
            "decision": "pass",
        },
        {
            "artifact_schema": "dftr.gate.factuality.v1",
            "name": "brief_adherence",
            "decision": "pass",
        },
        {
            "artifact_schema": "dftr.gate.factuality.v1",
            "name": "factuality",
            "decision": "fail",
        },
        {
            "artifact_schema": "dftr.gate.factuality.v1",
            "name": "factuality",
            "decision": "pass",
            "detail": "not part of the frozen schema",
        },
        ["dftr.gate.factuality.v1", "factuality", "pass"],
    ],
)
def test_hard_gate_evidence_requires_exact_json_schema(tmp_path, evidence):
    gates = valid_gate_set(tmp_path)
    path = tmp_path / "gate-factuality.json"
    if isinstance(evidence, str):
        path.write_text(evidence)
    else:
        path.write_text(json.dumps(evidence) + "\n")
    gates["factuality"].update({
        "evidence_path": path.name,
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    with pytest.raises(MeasurementV2Error, match="hard gate evidence"):
        validate(tmp_path, gates)


def test_hard_gate_report_entry_requires_exact_outer_schema(tmp_path):
    gates = valid_gate_set(tmp_path)
    gates["factuality"]["unbound_note"] = "extra"
    with pytest.raises(MeasurementV2Error, match="hard gate report entry schema"):
        validate(tmp_path, gates)
