"""Operator-owned measurement-v2 protocol, report, and attestation surfaces."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .metrics.distribution_v2 import MeasurementV2Error, bandwidth_hash as compute_bandwidth_hash
from .metrics.quality_v2 import validate_selection_firewall


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_BLIND_GROUPS = {
    "disjoint_floor",
    "small_matrix_oracle",
    "kernel_freeze",
    "cardinality_fail_closed",
    "matched_control",
    "self_bleu_cardinality",
    "repetition_resolution",
    "prompt_match",
    "auc_refit",
    "selection_firewall",
    "cluster_power",
    "historical_immutability",
    "no_sealed_imitation",
}

ARTIFACT_BINDINGS = {
    "dependency_lock": ("dependency_lock_sha256", None),
    "metric_code": ("metric_code_sha256", None),
    "prompt_panel": ("prompt_panel_sha256", "dftr.measurement.prompt_panel.v2"),
    "prompt_brief": ("prompt_brief_sha256", None),
    "human_panels": ("human_panels_sha256", "dftr.measurement.human_panels.v2"),
    "human_panel_contents": ("human_panel_contents_sha256", None),
    "bandwidths": ("bandwidths_sha256", "dftr.measurement.bandwidths.v2"),
    "power_plan": ("power_plan_sha256", "dftr.measurement.power_plan.v2"),
    "calibration": ("calibration_sha256", "dftr.measurement.calibration.v2"),
    "matched_baseline": ("matched_baseline_sha256", "dftr.measurement.matched_sft_baseline.v2"),
    "matched_baseline_outputs": ("matched_baseline_outputs_sha256", None),
    "selection_policy": ("selection_policy_sha256", "dftr.measurement.selection_policy.v2"),
}

REQUIRED_HARD_GATE_SCHEMAS = {
    "factuality": "dftr.gate.factuality.v1",
    "brief_adherence": "dftr.gate.brief_adherence.v1",
    "validity": "dftr.gate.validity.v1",
    "collapse": "dftr.gate.collapse.v1",
}
HARD_GATE_EVIDENCE_FIELDS = frozenset({"artifact_schema", "name", "decision"})
HARD_GATE_REPORT_FIELDS = frozenset(
    {"version", "decision", "evidence_path", "evidence_sha256"}
)


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MeasurementV2Error("measurement artifact must be a JSON object")
    return value


def _load_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise MeasurementV2Error(
                f"{field} contains invalid JSON on line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise MeasurementV2Error(f"{field} rows must be JSON objects")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, field: str) -> str:
    text = str(value or "")
    if not SHA256_RE.fullmatch(text):
        raise MeasurementV2Error(f"{field} must be a lowercase SHA-256")
    return text


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise MeasurementV2Error(f"{field} must be an integer >= {minimum}")
    return value


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _require_seed_list(value: Any, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in value)
    ):
        raise MeasurementV2Error(f"{field} must be a non-empty list of nonnegative integers")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_bytes_without(value: dict[str, Any], field: str) -> bytes:
    payload = {key: item for key, item in value.items() if key != field}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_hard_gate_evidence_set(
    *,
    required_gates: Any,
    hard_gates: Any,
    artifact_root: str | Path | None,
    protocol_sha256: str,
    candidate_output_sha256: str,
    signed_report_payload_sha256: str,
) -> dict[str, str]:
    """Validate exact gate artifacts and bind them to signed promotion subjects.

    The evidence JSON is deliberately small and frozen. Its byte identity and
    outer path/version/decision are covered by the report signature; that same
    signed payload contains the verified protocol and candidate-output hashes.
    The returned identities make that complete binding explicit and unique per
    gate without introducing a self-referential report hash into the artifact.
    """
    if required_gates != REQUIRED_HARD_GATE_SCHEMAS:
        raise MeasurementV2Error("frozen required hard gate schemas are invalid")
    if not isinstance(hard_gates, dict) or set(hard_gates) != set(required_gates):
        raise MeasurementV2Error(
            "promotion hard gate set does not equal frozen intersection"
        )
    if artifact_root is None:
        raise MeasurementV2Error("hard gate evidence requires an artifact root")
    root = Path(artifact_root).resolve()
    protocol_sha = _require_sha(protocol_sha256, "hard gate protocol identity")
    candidate_sha = _require_sha(
        candidate_output_sha256, "hard gate candidate output identity"
    )
    report_sha = _require_sha(
        signed_report_payload_sha256, "hard gate signed report identity"
    )
    seen_paths: set[Path] = set()
    seen_evidence_identities: set[str] = set()
    bindings: dict[str, str] = {}
    for name, version in required_gates.items():
        gate = hard_gates.get(name)
        if not isinstance(gate, dict) or set(gate) != HARD_GATE_REPORT_FIELDS:
            raise MeasurementV2Error(f"hard gate report entry schema mismatch: {name}")
        if (
            gate.get("version") != version
            or gate.get("decision") != "pass"
            or not isinstance(gate.get("evidence_path"), str)
            or not gate["evidence_path"]
        ):
            raise MeasurementV2Error(
                f"hard gate version or decision mismatch: {name}"
            )
        evidence_sha = _require_sha(
            gate.get("evidence_sha256"), f"hard gate evidence identity: {name}"
        )
        evidence_path = _resolve_bound_path(
            root, gate.get("evidence_path"), f"hard_gate:{name}"
        )
        if evidence_path in seen_paths:
            raise MeasurementV2Error("hard gate evidence path is reused across gates")
        if evidence_sha in seen_evidence_identities:
            raise MeasurementV2Error(
                "hard gate evidence identity is reused across gates"
            )
        seen_paths.add(evidence_path)
        seen_evidence_identities.add(evidence_sha)
        try:
            evidence_bytes = evidence_path.read_bytes()
        except OSError as error:
            raise MeasurementV2Error(
                f"hard gate evidence cannot be read: {name}"
            ) from error
        if hashlib.sha256(evidence_bytes).hexdigest() != evidence_sha:
            raise MeasurementV2Error(
                f"hard gate evidence byte hash mismatch: {name}"
            )
        try:
            evidence = json.loads(
                evidence_bytes.decode("utf-8"),
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise MeasurementV2Error(
                f"hard gate evidence is not valid JSON: {name}"
            ) from error
        if not isinstance(evidence, dict) or set(evidence) != HARD_GATE_EVIDENCE_FIELDS:
            raise MeasurementV2Error(
                f"hard gate evidence schema mismatch: {name}"
            )
        if (
            evidence.get("artifact_schema") != version
            or evidence.get("name") != name
            or evidence.get("decision") != "pass"
        ):
            raise MeasurementV2Error(
                f"hard gate evidence semantic binding mismatch: {name}"
            )
        bindings[name] = _canonical_sha256(
            {
                "gate_name": name,
                "gate_version": version,
                "gate_decision": "pass",
                "evidence_sha256": evidence_sha,
                "protocol_sha256": protocol_sha,
                "candidate_output_sha256": candidate_sha,
                "signed_report_payload_sha256": report_sha,
            }
        )
    if len(set(bindings.values())) != len(bindings):
        raise MeasurementV2Error("hard gate evaluated identity is reused across gates")
    return bindings


def _decode_public_key(value: str) -> bytes:
    try:
        raw = bytes.fromhex(value) if re.fullmatch(r"[0-9a-fA-F]{64}", value) else base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise MeasurementV2Error("trusted Ed25519 public key is malformed") from error
    if len(raw) != 32:
        raise MeasurementV2Error("trusted Ed25519 public key must contain 32 bytes")
    return raw


def decode_trusted_public_key(value: str) -> bytes:
    """Return canonical Ed25519 bytes for a hex- or base64-encoded trust entry."""
    return _decode_public_key(value)


def verify_signed_document(
    document: dict[str, Any],
    *,
    signature_field: str,
    trusted_public_keys: dict[str, str] | None,
) -> dict[str, str]:
    signature = document.get(signature_field) or {}
    if signature.get("algorithm") != "ed25519":
        raise MeasurementV2Error("verifiable Ed25519 signature is required")
    key_id = str(signature.get("key_id") or "")
    trusted = trusted_public_keys or {}
    if not key_id or key_id not in trusted:
        raise MeasurementV2Error("signature key is not independently trusted")
    encoded = str(signature.get("signature_base64") or "")
    try:
        signature_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise MeasurementV2Error("signature encoding is invalid") from error
    payload = _canonical_bytes_without(document, signature_field)
    expected_payload_sha = hashlib.sha256(payload).hexdigest()
    if signature.get("signed_payload_sha256") != expected_payload_sha:
        raise MeasurementV2Error("signature payload hash mismatch")
    try:
        Ed25519PublicKey.from_public_bytes(_decode_public_key(trusted[key_id])).verify(
            signature_bytes, payload
        )
    except (InvalidSignature, ValueError) as error:
        raise MeasurementV2Error("signature verification failed") from error
    return {"status": "verified", "key_id": key_id, "signed_payload_sha256": expected_payload_sha}


def _resolve_bound_path(root: Path, relative: Any, field: str) -> Path:
    text = str(relative or "")
    if not text or Path(text).is_absolute():
        raise MeasurementV2Error(f"artifact binding {field} requires a relative path")
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise MeasurementV2Error(f"artifact binding {field} escapes artifact root") from error
    if not path.is_file():
        raise MeasurementV2Error(f"artifact binding {field} does not resolve to a file")
    return path


def _verify_artifact_bindings(
    protocol: dict[str, Any], artifact_root: str | Path | None
) -> dict[str, Any]:
    if artifact_root is None:
        raise MeasurementV2Error("artifact root is required for content verification")
    root = Path(artifact_root).resolve()
    bindings = protocol.get("artifact_bindings") or {}
    hashes = protocol.get("hashes") or {}
    loaded: dict[str, Any] = {}
    for name, (hash_field, expected_schema) in ARTIFACT_BINDINGS.items():
        binding = bindings.get(name) or {}
        path = _resolve_bound_path(root, binding.get("path"), name)
        observed = _sha256(path)
        declared = _require_sha(binding.get("sha256"), f"artifact_bindings.{name}.sha256")
        top_level = _require_sha(hashes.get(hash_field), f"hashes.{hash_field}")
        if observed != declared or observed != top_level:
            raise MeasurementV2Error(f"artifact binding hash mismatch: {name}")
        if expected_schema is not None:
            value = _load(path)
            if value.get("artifact_schema") != expected_schema:
                raise MeasurementV2Error(f"artifact binding {name} has unexpected schema")
            loaded[name] = value
        else:
            loaded[name] = path
    return loaded


def _validate_bound_protocol_content(
    protocol: dict[str, Any], artifacts: dict[str, Any]
) -> None:
    n = protocol["design"]["documents_per_cell"]
    hashes = protocol["hashes"]
    panels = artifacts["human_panels"]
    if panels.get("status") != "materialized" or panels.get("frozen") is not True:
        raise MeasurementV2Error("human panel manifest is not frozen and materialized")
    if panels.get("content_bundle_sha256") != hashes["human_panel_contents_sha256"]:
        raise MeasurementV2Error("human panel manifest does not bind the content bundle bytes")
    if panels.get("target_n_per_panel") != n:
        raise MeasurementV2Error("human panel manifest n does not match protocol")
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    content_rows = _load_jsonl(artifacts["human_panel_contents"], "human panel contents")
    content_by_id: dict[str, str] = {}
    eligibility_rows: list[dict[str, Any]] = []
    for row in content_rows:
        document_id = str(row.get("document_id") or "")
        text = row.get("text")
        if not document_id or document_id in content_by_id or not isinstance(text, str):
            raise MeasurementV2Error(
                "human panel contents require unique document_id and string text"
            )
        content_by_id[document_id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        eligibility_basis = str(row.get("eligibility_basis") or "").strip()
        exclusion_flags = row.get("exclusion_flags")
        if (
            row.get("eligible") is not True
            or not eligibility_basis
            or not isinstance(exclusion_flags, list)
            or exclusion_flags
        ):
            raise MeasurementV2Error(
                "human panel content lacks passing eligibility and exclusion evidence"
            )
        eligibility_rows.append({
            "document_id": document_id,
            "eligible": True,
            "eligibility_basis": eligibility_basis,
            "exclusion_flags": [],
        })
    if len(content_by_id) != 3 * n:
        raise MeasurementV2Error("human panel content bundle cardinality must equal 3n")
    manifest_panels = panels.get("panels") or {}
    for name in ("human_eval", "human_floor_a", "human_floor_b"):
        rows = manifest_panels.get(name)
        if not isinstance(rows, list) or len(rows) != n:
            raise MeasurementV2Error(f"human panel content cardinality mismatch: {name}")
        ids = []
        for row in rows:
            if not isinstance(row, dict):
                raise MeasurementV2Error("human panel rows must be content-addressed objects")
            document_id = str(row.get("document_id") or "")
            content_sha = _require_sha(row.get("content_sha256"), "human panel content_sha256")
            if not document_id or document_id in seen_ids or content_sha in seen_content:
                raise MeasurementV2Error("human panel IDs and content hashes must be globally unique")
            if content_by_id.get(document_id) != content_sha:
                raise MeasurementV2Error(
                    "human panel content fingerprint does not match bound content bytes"
                )
            ids.append(document_id)
            seen_ids.add(document_id)
            seen_content.add(content_sha)
        if ids != (protocol["panels"].get(name) or {}).get("document_ids"):
            raise MeasurementV2Error(f"protocol panel IDs do not bind manifest order: {name}")
        if (protocol["panels"].get(name) or {}).get("content_manifest_sha256") != _canonical_sha256(rows):
            raise MeasurementV2Error(f"protocol panel content hash mismatch: {name}")
    if set(content_by_id) != seen_ids:
        raise MeasurementV2Error("human panel content bundle IDs do not match panel manifest")
    eligibility_sha = _require_sha(
        panels.get("eligibility_attestation_sha256"),
        "human panels eligibility attestation",
    )
    if eligibility_sha != _canonical_sha256(eligibility_rows):
        raise MeasurementV2Error(
            "human panel eligibility attestation does not match bound records"
        )

    bandwidths = artifacts["bandwidths"]
    if (
        bandwidths.get("status") != "frozen"
        or bandwidths.get("frozen") is not True
        or bandwidths.get("source") != "human_floor_a_union_human_floor_b_only"
        or bandwidths.get("values") != protocol["bandwidth_contract"].get("values")
        or bandwidths.get("panel_manifest_sha256") != hashes["human_panels_sha256"]
        or bandwidths.get("floor_a_content_manifest_sha256")
        != protocol["panels"]["human_floor_a"].get("content_manifest_sha256")
        or bandwidths.get("floor_b_content_manifest_sha256")
        != protocol["panels"]["human_floor_b"].get("content_manifest_sha256")
    ):
        raise MeasurementV2Error("frozen human-only bandwidth artifact is not cross-bound")
    _require_sha(bandwidths.get("embedder_sha256"), "bandwidth embedder hash")
    _require_sha(bandwidths.get("preprocessing_sha256"), "bandwidth preprocessing hash")
    if bandwidth_hash := bandwidths.get("bandwidth_sha256"):
        if bandwidth_hash != compute_bandwidth_hash(bandwidths["values"]):
            raise MeasurementV2Error("frozen bandwidth value hash mismatch")
    else:
        raise MeasurementV2Error("frozen bandwidth value hash is absent")
    if protocol["bandwidth_contract"].get("bandwidth_value_sha256") != bandwidth_hash:
        raise MeasurementV2Error("protocol bandwidth value hash mismatch")

    prompt_panel = artifacts["prompt_panel"]
    prompt_ids = prompt_panel.get("prompt_ids") or []
    if (
        prompt_panel.get("status") != "frozen"
        or prompt_panel.get("frozen") is not True
        or len(prompt_ids) != n
        or len(set(prompt_ids)) != n
    ):
        raise MeasurementV2Error("prompt panel is not a frozen unique n-sized panel")
    full_brief_sha = _require_sha(prompt_panel.get("full_brief_sha256"), "prompt panel full brief hash")
    if full_brief_sha != hashes["prompt_brief_sha256"]:
        raise MeasurementV2Error("prompt panel does not bind the complete brief bytes")

    baseline = artifacts["matched_baseline"]
    matched = protocol.get("matched_design") or {}
    if (
        matched.get("candidate_full_brief_sha256") != full_brief_sha
        or matched.get("control_full_brief_sha256") != full_brief_sha
    ):
        raise MeasurementV2Error("candidate/control full brief bytes are not prompt-panel bound")
    if (
        baseline.get("status") != "materialized"
        or baseline.get("frozen") is not True
        or baseline.get("documents_per_cell") != n
        or baseline.get("prompt_panel_sha256") != hashes["prompt_panel_sha256"]
        or baseline.get("full_brief_sha256") != full_brief_sha
        or baseline.get("sampling_grid_sha256") != matched.get("sampling_grid_sha256")
        or baseline.get("seed_grid") != matched.get("seed_grid")
        or baseline.get("output_manifest_sha256") != matched.get("control_output_manifest_sha256")
        or baseline.get("output_manifest_sha256") != hashes["matched_baseline_outputs_sha256"]
        or baseline.get("checkpoint_sha256") != matched.get("control_checkpoint_sha256")
        or baseline.get("decoding_policy_sha256") != matched.get("decoding_policy_sha256")
        or baseline.get("generation_contract_sha256")
        != matched.get("generation_contract_sha256")
    ):
        raise MeasurementV2Error("matched baseline does not bind the frozen prompt/sampling design")
    output_rows = _load_jsonl(
        artifacts["matched_baseline_outputs"], "matched baseline outputs"
    )
    expected_cells = {
        (str(prompt_id), training_seed, sampling_seed)
        for cell in matched.get("seed_grid") or []
        for training_seed in [cell.get("training_seed")]
        for sampling_seed in cell.get("sampling_seeds") or []
        for prompt_id in prompt_ids
    }
    if not expected_cells or len(output_rows) != len(expected_cells):
        raise MeasurementV2Error(
            "matched baseline output cardinality does not equal the frozen prompt-seed grid"
        )
    observed_cells: set[tuple[str, Any, Any]] = set()
    for row in output_rows:
        key = (
            str(row.get("prompt_id") or ""),
            row.get("training_seed"),
            row.get("sampling_seed"),
        )
        if key in observed_cells or key not in expected_cells:
            raise MeasurementV2Error(
                "matched baseline outputs contain duplicate or unregistered prompt-seed cells"
            )
        if not isinstance(row.get("text"), str):
            raise MeasurementV2Error("matched baseline output text must be a string")
        if (
            row.get("full_brief_sha256") != full_brief_sha
            or row.get("prompt_panel_sha256") != hashes["prompt_panel_sha256"]
            or row.get("sampling_grid_sha256") != matched.get("sampling_grid_sha256")
            or row.get("checkpoint_sha256") != matched.get("control_checkpoint_sha256")
            or row.get("decoding_policy_sha256") != matched.get("decoding_policy_sha256")
            or row.get("generation_contract_sha256")
            != matched.get("generation_contract_sha256")
        ):
            raise MeasurementV2Error(
                "matched baseline output content is not bound to brief, sampler, and checkpoint"
            )
        observed_cells.add(key)
    if observed_cells != expected_cells:
        raise MeasurementV2Error("matched baseline outputs do not cover the frozen grid")

    calibration = artifacts["calibration"]
    required_calibration_hashes = {
        "human_panels_sha256": hashes["human_panels_sha256"],
        "bandwidths_sha256": hashes["bandwidths_sha256"],
        "matched_baseline_sha256": hashes["matched_baseline_sha256"],
        "dependency_lock_sha256": hashes["dependency_lock_sha256"],
    }
    if (
        calibration.get("status") != "frozen"
        or calibration.get("frozen") is not True
        or calibration.get("documents_per_cell") != n
        or any(calibration.get(key) != value for key, value in required_calibration_hashes.items())
    ):
        raise MeasurementV2Error("calibration artifact is not matched and cross-bound")

    power = artifacts["power_plan"]
    results = power.get("results") or {}
    simulation = power.get("simulation_contract") or {}
    effects = simulation.get("minimally_important_effects") or {}
    trial_rows = power.get("simulation_results") or []
    required_scenarios = {
        "mmd_type_i": ("null", 0.0, 0.05, "maximum"),
        "mmd_power": ("alternative", effects.get("mmd"), 0.8, "minimum"),
        "auc_power": ("alternative", effects.get("auc"), 0.8, "minimum"),
        "repetition_power": ("alternative", effects.get("repetition"), 0.8, "minimum"),
        "coverage": ("coverage", effects.get("coverage"), 0.93, "coverage"),
    }
    parsed_trials: dict[str, dict[str, Any]] = {}
    if isinstance(trial_rows, list):
        for row in trial_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("endpoint") or "")
            trials = row.get("trials")
            successes = row.get("successes")
            if (
                name in parsed_trials
                or not isinstance(trials, int)
                or isinstance(trials, bool)
                or trials < 1000
                or not isinstance(successes, int)
                or isinstance(successes, bool)
                or successes < 0
                or successes > trials
            ):
                continue
            parsed_trials[name] = row
    semantic_power_valid = (
        simulation.get("prospective") is True
        and simulation.get("prompt_clusters") == n
        and simulation.get("documents_per_cell") == n
        and simulation.get("seed_grid") == matched.get("seed_grid")
        and _require_sha(simulation.get("null_generator_sha256"), "power null generator hash")
        and _require_sha(
            simulation.get("alternative_generator_sha256"),
            "power alternative generator hash",
        )
        and _require_sha(
            simulation.get("analysis_code_sha256"), "power analysis code hash"
        ) == hashes["metric_code_sha256"]
        and _require_sha(
            power.get("trial_manifest_sha256"), "power trial manifest hash"
        ) == _canonical_sha256(simulation)
        and _require_sha(
            power.get("simulation_results_sha256"), "power simulation results hash"
        ) == _canonical_sha256(trial_rows)
        and set(parsed_trials) == set(required_scenarios)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in effects.values()
        )
    )
    if semantic_power_valid:
        for name, (scenario, effect, threshold, rule) in required_scenarios.items():
            row = parsed_trials[name]
            rate = row["successes"] / row["trials"]
            if row.get("scenario") != scenario or row.get("effect") != effect:
                semantic_power_valid = False
                break
            reported = results.get("mmd_type_i_rate" if name == "mmd_type_i" else name)
            if not isinstance(reported, (int, float)) or abs(float(reported) - rate) > 1e-12:
                semantic_power_valid = False
                break
            if rule == "maximum" and rate > threshold:
                semantic_power_valid = False
                break
            if rule == "minimum" and rate < threshold:
                semantic_power_valid = False
                break
            if rule == "coverage" and not 0.93 <= rate <= 0.97:
                semantic_power_valid = False
                break
    if (
        power.get("status") != "frozen"
        or power.get("frozen") is not True
        or power.get("all_targets_pass") is not True
        or power.get("documents_per_cell") != n
        or results.get("mmd_type_i_max", 1.0) > 0.05
        or results.get("mmd_type_i_rate", 1.0) > results.get("mmd_type_i_max", 0.05)
        or results.get("mmd_power", 0.0) < 0.8
        or results.get("auc_power", 0.0) < 0.8
        or results.get("repetition_power", 0.0) < 0.8
        or not 0.93 <= results.get("coverage", 0.0) <= 0.97
        or not str((power.get("multiplicity") or {}).get("method") or "")
        or semantic_power_valid is not True
        or any(power.get(key) != hashes[value] for key, value in {
            "human_panels_sha256": "human_panels_sha256",
            "bandwidths_sha256": "bandwidths_sha256",
            "calibration_sha256": "calibration_sha256",
            "matched_baseline_sha256": "matched_baseline_sha256",
            "dependency_lock_sha256": "dependency_lock_sha256",
        }.items())
    ):
        raise MeasurementV2Error("power evidence is incomplete, underpowered, or not cross-bound")

    selection = artifacts["selection_policy"]
    if selection.get("status") != "frozen" or selection.get("selection") != protocol.get("selection_policy", {}).get("selection"):
        raise MeasurementV2Error("selection policy artifact is not cross-bound")
    validate_selection_firewall(selection)


def _manifest_digest(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_historical_inventory(
    inventory: dict[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    if inventory.get("artifact_schema") != "dftr.measurement.historical_inventory.v1":
        raise MeasurementV2Error("unexpected historical inventory schema")
    root = Path(repo_root).resolve()
    results = []
    for artifact_set in inventory.get("artifact_sets") or []:
        patterns = artifact_set.get("include_globs") or []
        paths = sorted(
            {
                path.resolve()
                for pattern in patterns
                for path in root.glob(str(pattern))
                if path.is_file()
            }
        )
        expected_count = _require_int(
            artifact_set.get("file_count"),
            f"artifact_sets.{artifact_set.get('name')}.file_count",
        )
        expected_digest = _require_sha(
            artifact_set.get("manifest_sha256"),
            f"artifact_sets.{artifact_set.get('name')}.manifest_sha256",
        )
        observed_digest = _manifest_digest(root, paths)
        passed = len(paths) == expected_count and observed_digest == expected_digest
        results.append(
            {
                "name": str(artifact_set.get("name")),
                "file_count": len(paths),
                "expected_file_count": expected_count,
                "manifest_sha256": observed_digest,
                "expected_manifest_sha256": expected_digest,
                "status": "pass" if passed else "fail",
            }
        )
    if not results:
        raise MeasurementV2Error("historical inventory has no artifact sets")
    return {
        "artifact_schema": "dftr.measurement.historical_inventory_check.v1",
        "status": "pass" if all(row["status"] == "pass" for row in results) else "fail",
        "inventory_sha256": _canonical_sha256(inventory),
        "inventory": inventory,
        "artifact_sets": results,
    }


def _verify_historical_inventory_check(
    inventory_check: dict[str, Any], *, repo_root: str | Path | None,
    trusted_public_keys: dict[str, str] | None
) -> dict[str, str]:
    if (
        inventory_check.get("artifact_schema")
        != "dftr.measurement.historical_inventory_check.v1"
        or inventory_check.get("status") != "pass"
    ):
        raise MeasurementV2Error("historical inventory check schema or status is invalid")
    inventory = inventory_check.get("inventory")
    if not isinstance(inventory, dict):
        raise MeasurementV2Error("historical inventory check does not bind its source inventory")
    inventory_sha = _require_sha(
        inventory_check.get("inventory_sha256"), "historical inventory hash"
    )
    if inventory_sha != _canonical_sha256(inventory):
        raise MeasurementV2Error("historical inventory source hash mismatch")
    if repo_root is None:
        raise MeasurementV2Error("historical inventory verification requires a repository root")
    rows = inventory_check.get("artifact_sets")
    if not isinstance(rows, list) or not rows:
        raise MeasurementV2Error("historical inventory check has no verified artifact sets")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise MeasurementV2Error("historical inventory rows must be objects")
        name = str(row.get("name") or "")
        if not name or name in names or row.get("status") != "pass":
            raise MeasurementV2Error("historical inventory rows are duplicate or failed")
        names.add(name)
        observed_count = _require_int(row.get("file_count"), "historical file count")
        expected_count = _require_int(
            row.get("expected_file_count"), "historical expected file count"
        )
        observed_sha = _require_sha(
            row.get("manifest_sha256"), "historical observed manifest hash"
        )
        expected_sha = _require_sha(
            row.get("expected_manifest_sha256"), "historical expected manifest hash"
        )
        if observed_count != expected_count or observed_sha != expected_sha:
            raise MeasurementV2Error("historical inventory row evidence does not match")
    recomputed = verify_historical_inventory(inventory, repo_root=repo_root)
    if (
        recomputed.get("status") != "pass"
        or recomputed.get("inventory_sha256") != inventory_sha
        or recomputed.get("artifact_sets") != rows
    ):
        raise MeasurementV2Error(
            "historical inventory check does not match repository bytes"
        )
    return verify_signed_document(
        inventory_check,
        signature_field="operator_signature",
        trusted_public_keys=trusted_public_keys,
    )


def protocol_readiness(
    protocol: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
    trusted_public_keys: dict[str, str] | None = None,
) -> dict[str, Any]:
    reasons = []
    if protocol.get("artifact_schema") != "dftr.measurement.protocol.v2":
        reasons.append("unexpected_schema")
    if protocol.get("frozen") is not True:
        reasons.append("protocol_not_frozen")
    if protocol.get("status") != "ready":
        reasons.append("protocol_not_ready")
    design = protocol.get("design") or {}
    n = _int_or_zero(design.get("documents_per_cell"))
    if n < 64:
        reasons.append("documents_per_cell_below_64")
    if _int_or_zero(design.get("human_pool_documents")) < 3 * n:
        reasons.append("human_pool_below_3n")
    if design.get("replacement_sampling") is not False:
        reasons.append("replacement_sampling_not_prohibited")
    hashes = protocol.get("hashes") or {}
    for field in (
        "dependency_lock_sha256",
        "metric_code_sha256",
        "prompt_panel_sha256",
        "prompt_brief_sha256",
        "human_panels_sha256",
        "human_panel_contents_sha256",
        "bandwidths_sha256",
        "power_plan_sha256",
        "calibration_sha256",
        "matched_baseline_sha256",
        "matched_baseline_outputs_sha256",
        "selection_policy_sha256",
    ):
        if not SHA256_RE.fullmatch(str(hashes.get(field) or "")):
            reasons.append(f"missing_hash:{field}")
    panels = protocol.get("panels") or {}
    panel_ids = []
    for name in ("human_eval", "human_floor_a", "human_floor_b"):
        panel = panels.get(name) or {}
        if panel.get("status") != "materialized" or panel.get("frozen") is not True:
            reasons.append(f"panel_unavailable:{name}")
        if _int_or_zero(panel.get("document_count")) != n:
            reasons.append(f"panel_cardinality:{name}")
        ids = panel.get("document_ids")
        if not isinstance(ids, list) or len(ids) != n:
            reasons.append(f"panel_id_cardinality:{name}")
        else:
            panel_ids.extend(ids)
    if len(panel_ids) != 3 * n or len(panel_ids) != len(set(panel_ids)):
        reasons.append("human_panels_not_disjoint")
    bandwidth = protocol.get("bandwidth_contract") or {}
    bandwidth_values = bandwidth.get("values") or []
    if (
        bandwidth.get("status") != "frozen"
        or bandwidth.get("source") != "human_floor_a_union_human_floor_b"
        or not bandwidth_values
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in bandwidth_values
        )
    ):
        reasons.append("bandwidths_not_frozen_human_only")
    power = protocol.get("power") or {}
    required_power = ("mmd_type_i", "mmd_power", "auc_power", "repetition_power", "coverage")
    for field in required_power:
        if power.get(field) != "pass":
            reasons.append(f"power_not_passed:{field}")
    required_gates = protocol.get("required_hard_gates")
    if required_gates != REQUIRED_HARD_GATE_SCHEMAS:
        reasons.append("required_hard_gates_not_frozen")
    seeds = protocol.get("seeds") or {}
    for field in ("permutation", "bootstrap", "authorship_split"):
        value = seeds.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            reasons.append(f"seed_not_frozen:{field}")
    try:
        validate_selection_firewall(protocol.get("selection_policy") or {})
    except MeasurementV2Error:
        reasons.append("selection_policy_not_endpoint_independent")
    approval = protocol.get("operator_approval") or {}
    if (
        approval.get("approved") is not True
        or not str(approval.get("reviewer") or "").strip()
        or not str(approval.get("reviewed_at") or "").strip()
    ):
        reasons.append("operator_approval_not_bound")
    try:
        artifacts = _verify_artifact_bindings(protocol, artifact_root)
        _validate_bound_protocol_content(protocol, artifacts)
    except (OSError, json.JSONDecodeError, MeasurementV2Error, TypeError, KeyError) as error:
        reasons.append(f"artifact_evidence_invalid:{error}")
    try:
        verify_signed_document(
            protocol,
            signature_field="operator_signature",
            trusted_public_keys=trusted_public_keys,
        )
    except MeasurementV2Error as error:
        reasons.append(f"signature_invalid:{error}")
    return {
        "artifact_schema": "dftr.measurement.protocol_readiness.v2",
        "status": "ready" if not reasons else "fail_closed",
        "reasons": sorted(set(reasons)),
    }


def validate_protocol(
    protocol: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
    trusted_public_keys: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = protocol_readiness(
        protocol, artifact_root=artifact_root, trusted_public_keys=trusted_public_keys
    )
    if result["status"] != "ready":
        raise MeasurementV2Error(
            "measurement v2 protocol is not ready: " + ", ".join(result["reasons"])
        )
    return result


def prepare_protocol_transfer(
    candidate_path: str | Path,
    expected_sha256: str,
    *,
    artifact_root: str | Path | None = None,
    trusted_public_keys: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = Path(candidate_path)
    if _sha256(path) != _require_sha(expected_sha256, "expected_sha256"):
        raise MeasurementV2Error("protocol candidate SHA-256 mismatch")
    candidate = _load(path)
    if candidate.get("artifact_schema") != "dftr.measurement.protocol_candidate.v2":
        raise MeasurementV2Error("unexpected protocol candidate schema")
    if (candidate.get("operator_review") or {}).get("approved") is not True:
        raise MeasurementV2Error("protocol candidate lacks operator approval")
    transferred = candidate.get("frozen_protocol")
    if not isinstance(transferred, dict):
        raise MeasurementV2Error("candidate lacks a separately signed frozen protocol")
    if (
        transferred.get("artifact_schema") != "dftr.measurement.protocol.v2"
        or transferred.get("status") != "ready"
        or transferred.get("frozen") is not True
    ):
        raise MeasurementV2Error("embedded frozen protocol has invalid transfer state")
    validate_protocol(
        transferred, artifact_root=artifact_root, trusted_public_keys=trusted_public_keys
    )
    return dict(transferred)


def validate_report_v2(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any] | None = None,
    artifact_root: str | Path | None = None,
    trusted_public_keys: dict[str, str] | None = None,
) -> dict[str, Any]:
    if report.get("artifact_schema") != "dftr.measurement.report.v2":
        raise MeasurementV2Error("unexpected measurement report schema")
    if report.get("evidence_class") not in {"prospective_screen", "post_hoc_shadow"}:
        raise MeasurementV2Error("v2 report has invalid evidence_class")
    promotion = report.get("promotion") or {}
    if not isinstance(promotion.get("eligible"), bool):
        raise MeasurementV2Error("report promotion eligibility must be boolean")
    if report.get("evidence_class") == "post_hoc_shadow" and promotion.get("eligible") is True:
        raise MeasurementV2Error("post-hoc shadow evidence can never be promotion eligible")
    authorship = report.get("authorship") or {}
    if authorship.get("status") == "underpowered" and promotion.get("eligible") is True:
        raise MeasurementV2Error("underpowered authorship can never be promotion eligible")
    counts = report.get("counts") or {}
    n = _require_int(counts.get("documents_per_cell"), "report.counts.documents_per_cell", minimum=64)
    if _require_int(
        counts.get("human_documents_per_panel"),
        "report.counts.human_documents_per_panel",
        minimum=1,
    ) != n:
        raise MeasurementV2Error("v2 report requires matched n >= 64")
    if _require_int(
        counts.get("effective_prompt_clusters"),
        "report.counts.effective_prompt_clusters",
        minimum=1,
    ) != n:
        raise MeasurementV2Error("sampling-seed rows cannot inflate effective prompt n")
    hashes = report.get("hashes") or {}
    for field in (
        "protocol_sha256",
        "prompt_panel_sha256",
        "human_eval_sha256",
        "human_floor_a_sha256",
        "human_floor_b_sha256",
        "bandwidths_sha256",
        "power_plan_sha256",
        "dependency_lock_sha256",
        "evaluator_commit_sha256",
    ):
        _require_sha(hashes.get(field), f"report.hashes.{field}")
    seeds = report.get("seeds") or {}
    _require_seed_list(seeds.get("training"), "report.seeds.training")
    _require_seed_list(seeds.get("sampling"), "report.seeds.sampling")
    distribution = report.get("distribution") or {}
    if (
        _require_int(distribution.get("documents_per_cell"), "report.distribution.documents_per_cell", minimum=1) != n
        or _require_int(distribution.get("human_documents_per_panel"), "report.distribution.human_documents_per_panel", minimum=1) != n
    ):
        raise MeasurementV2Error("distribution cells must all use report n")
    distribution_bandwidth_hash = _require_sha(
        distribution.get("bandwidth_sha256"), "report.distribution.bandwidth_sha256"
    )
    if distribution_bandwidth_hash != hashes["bandwidths_sha256"]:
        raise MeasurementV2Error("distribution bandwidth hash does not match frozen bandwidth artifact")
    for field in (
        "candidate_mmd2_unbiased",
        "control_mmd2_unbiased",
        "human_floor_mmd2_unbiased",
    ):
        value = distribution.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise MeasurementV2Error(f"report.distribution.{field} must be finite")
    _require_int(distribution.get("permutation_seed"), "report.distribution.permutation_seed")
    checkpoint_manifest = report.get("checkpoint_manifest") or {}
    validate_selection_firewall(checkpoint_manifest)
    selection = checkpoint_manifest["selection"]
    selected_seeds = (
        selection.get("seeds")
        if selection.get("rule_type") == "all_preregistered_seeds"
        else [selection.get("seed")]
    )
    if selection.get("rule_type") != "training_only" and selected_seeds != seeds["training"]:
        raise MeasurementV2Error("checkpoint selection seed does not match report training seeds")
    for field in (
        "calibration_sha256",
        "matched_baseline_sha256",
        "selection_policy_sha256",
        "candidate_full_brief_sha256",
        "control_full_brief_sha256",
        "sampling_grid_sha256",
        "control_output_manifest_sha256",
    ):
        _require_sha(hashes.get(field), f"report.hashes.{field}")
    if (report.get("quality") or {}).get("status") == "measured":
        linkage = report.get("quality_linkage") or {}
        if linkage.get("status") != "verified" or _int_or_zero(linkage.get("matched_pairs")) != n:
            raise MeasurementV2Error("measured quality lacks one-to-one prompt linkage")
    if (report.get("repetition") or {}).get("status") == "underpowered":
        if promotion.get("eligible") is True:
            raise MeasurementV2Error("underpowered repetition cannot promote")
    repetition = report.get("repetition") or {}
    if _require_int(repetition.get("documents_per_panel"), "report.repetition.documents_per_panel", minimum=1) != n:
        raise MeasurementV2Error("repetition must use matched report n")
    if repetition.get("power_plan_passed") is not True and repetition.get("status") != "underpowered":
        raise MeasurementV2Error("repetition without power evidence must be underpowered")
    if authorship.get("grouped") is not True or _require_int(
        authorship.get("fit_count"), "report.authorship.fit_count", minimum=1
    ) < 1:
        raise MeasurementV2Error("authorship uncertainty must refit a grouped pipeline")
    if _require_int(
        authorship.get("effective_clusters"), "report.authorship.effective_clusters", minimum=1
    ) < 64 and authorship.get("status") != "underpowered":
        raise MeasurementV2Error("authorship with fewer than 64 clusters is underpowered")
    if protocol is None:
        raise MeasurementV2Error("report must be bound to a verified frozen protocol")
    if promotion.get("eligible") is True:
        promotion_binding_errors: list[str] = []
        required_gates = protocol.get("required_hard_gates")
        hard_gates = report.get("hard_gates")
        candidate_output_sha: str | None = None
        try:
            candidate_output_sha = _require_sha(
                hashes.get("candidate_output_manifest_sha256"),
                "report.hashes.candidate_output_manifest_sha256",
            )
            output_binding = report.get("candidate_output_binding") or {}
            if artifact_root is None:
                raise MeasurementV2Error("candidate output binding requires an artifact root")
            output_path = _resolve_bound_path(
                Path(artifact_root).resolve(), output_binding.get("path"),
                "candidate_output",
            )
            if (
                output_binding.get("sha256") != candidate_output_sha
                or _sha256(output_path) != candidate_output_sha
            ):
                raise MeasurementV2Error("candidate output binding byte hash mismatch")
        except MeasurementV2Error as error:
            promotion_binding_errors.append(str(error))
        signature_result: dict[str, str] | None = None
        try:
            signature_result = verify_signed_document(
                report,
                signature_field="operator_signature",
                trusted_public_keys=trusted_public_keys,
            )
        except MeasurementV2Error as error:
            promotion_binding_errors.append(f"promotion report signature invalid: {error}")
        try:
            protocol_sha = _require_sha(
                hashes.get("protocol_sha256"), "hard gate protocol identity"
            )
            if protocol_sha != _canonical_sha256(protocol):
                raise MeasurementV2Error("hard gate protocol identity mismatch")
            if candidate_output_sha is None or signature_result is None:
                raise MeasurementV2Error(
                    "hard gate evaluated identities are not verified"
                )
            _validate_hard_gate_evidence_set(
                required_gates=required_gates,
                hard_gates=hard_gates,
                artifact_root=artifact_root,
                protocol_sha256=protocol_sha,
                candidate_output_sha256=candidate_output_sha,
                signed_report_payload_sha256=signature_result[
                    "signed_payload_sha256"
                ],
            )
        except MeasurementV2Error as error:
            promotion_binding_errors.append(str(error))
        if promotion_binding_errors:
            raise MeasurementV2Error("; ".join(promotion_binding_errors))
    validate_protocol(
        protocol, artifact_root=artifact_root, trusted_public_keys=trusted_public_keys
    )
    if hashes["protocol_sha256"] != _canonical_sha256(protocol):
        raise MeasurementV2Error("report protocol hash does not match verified protocol")
    protocol_hashes = protocol.get("hashes") or {}
    for report_field, protocol_field in (
        ("prompt_panel_sha256", "prompt_panel_sha256"),
        ("control_output_manifest_sha256", "matched_baseline_outputs_sha256"),
        ("bandwidths_sha256", "bandwidths_sha256"),
        ("power_plan_sha256", "power_plan_sha256"),
        ("calibration_sha256", "calibration_sha256"),
        ("matched_baseline_sha256", "matched_baseline_sha256"),
        ("selection_policy_sha256", "selection_policy_sha256"),
        ("dependency_lock_sha256", "dependency_lock_sha256"),
        ("evaluator_commit_sha256", "metric_code_sha256"),
    ):
        if hashes[report_field] != protocol_hashes.get(protocol_field):
            raise MeasurementV2Error(f"report hash is not protocol-bound: {report_field}")
    for report_field, panel_name in (
        ("human_eval_sha256", "human_eval"),
        ("human_floor_a_sha256", "human_floor_a"),
        ("human_floor_b_sha256", "human_floor_b"),
    ):
        if hashes[report_field] != (protocol.get("panels", {}).get(panel_name) or {}).get(
            "content_manifest_sha256"
        ):
            raise MeasurementV2Error(f"report human panel content hash mismatch: {report_field}")
    matched = protocol.get("matched_design") or {}
    for field in (
        "candidate_full_brief_sha256",
        "control_full_brief_sha256",
        "sampling_grid_sha256",
        "control_output_manifest_sha256",
    ):
        if hashes[field] != matched.get(field):
            raise MeasurementV2Error(f"report matched-design hash mismatch: {field}")
    cells = seeds.get("cells")
    if not isinstance(cells, list) or cells != matched.get("seed_grid"):
        raise MeasurementV2Error("report training/sampling seed nesting is not protocol-bound")
    if promotion.get("eligible") is True:
        quality = report.get("quality") or {}
        if (
            report.get("evidence_class") != "prospective_screen"
            or distribution.get("decision") != "pass"
            or distribution.get("power_plan_passed") is not True
            or quality.get("status") != "measured"
            or quality.get("decision") != "pass"
            or repetition.get("status") != "ready"
            or repetition.get("decision") != "pass"
            or repetition.get("power_plan_passed") is not True
            or authorship.get("status") != "ready"
            or authorship.get("decision") != "pass"
        ):
            raise MeasurementV2Error("promotion intersection rule is not satisfied")
    return {"status": "pass", "artifact_schema": "dftr.measurement.report_check.v2"}


def validate_blind_qualification(
    *,
    protocol: dict[str, Any],
    blind_test_manifest: dict[str, Any],
    operator: str,
    artifact_root: str | Path | None = None,
    trusted_public_keys: dict[str, str] | None = None,
    require_distinct_signer: bool = True,
) -> dict[str, Any]:
    """Verify the independently signed, protocol-bound 13-group qualification."""
    blind_signature = verify_signed_document(
        blind_test_manifest,
        signature_field="operator_signature",
        trusted_public_keys=trusted_public_keys,
    )
    protocol_signature = verify_signed_document(
        protocol,
        signature_field="operator_signature",
        trusted_public_keys=trusted_public_keys,
    )
    if require_distinct_signer:
        trusted = trusted_public_keys or {}
        if (
            blind_signature["key_id"] == protocol_signature["key_id"]
            or decode_trusted_public_key(trusted[blind_signature["key_id"]])
            == decode_trusted_public_key(trusted[protocol_signature["key_id"]])
        ):
            raise MeasurementV2Error(
                "blind qualification signer must be distinct from protocol operator"
            )
    validate_protocol(
        protocol, artifact_root=artifact_root, trusted_public_keys=trusted_public_keys
    )
    tests = blind_test_manifest.get("tests") or []
    if (
        blind_test_manifest.get("artifact_schema") != "dftr.measurement.blind_test_manifest.v2"
        or blind_test_manifest.get("status") != "qualified"
        or not str(blind_test_manifest.get("tested_at") or "").strip()
        or not isinstance(blind_test_manifest.get("runtime_versions"), dict)
        or not blind_test_manifest.get("runtime_versions")
        or any(
            not str(key).strip() or not str(value).strip()
            for key, value in (blind_test_manifest.get("runtime_versions") or {}).items()
        )
    ):
        raise MeasurementV2Error("signed blind manifest metadata is incomplete")
    if (
        not isinstance(tests, list)
        or len(tests) != len(REQUIRED_BLIND_GROUPS)
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "status"}
            or row.get("status") != "pass"
            for row in tests
        )
    ):
        raise MeasurementV2Error("blind qualification requires the exact passing group rows")
    observed = [str(row["name"]) for row in tests]
    if len(observed) != len(set(observed)) or set(observed) != REQUIRED_BLIND_GROUPS:
        raise MeasurementV2Error("blind qualification does not equal the frozen group set")
    for field in ("evaluator_commit", "dependency_lock_sha256", "fixture_pack_sha256"):
        _require_sha(blind_test_manifest.get(field), f"blind_test_manifest.{field}")
    if blind_test_manifest.get("no_sealed_imitation") is not True:
        raise MeasurementV2Error("no-sealed-imitation attestation is absent")
    if not str(operator).strip():
        raise MeasurementV2Error("operator is required")
    if blind_test_manifest.get("dependency_lock_sha256") != protocol["hashes"].get(
        "dependency_lock_sha256"
    ):
        raise MeasurementV2Error("signed blind manifest dependency lock is not protocol-bound")
    if blind_test_manifest.get("evaluator_commit") != protocol["hashes"].get(
        "metric_code_sha256"
    ):
        raise MeasurementV2Error("signed blind manifest evaluator image is not protocol-bound")
    if blind_test_manifest.get("protocol_sha256") != _canonical_sha256(protocol):
        raise MeasurementV2Error("signed blind manifest protocol hash mismatch")
    if str(blind_test_manifest.get("signer_identity") or "").strip() != str(operator).strip():
        raise MeasurementV2Error("signed blind manifest operator identity mismatch")
    return {
        "status": "qualified",
        "signature_verification": blind_signature,
        "protocol_signature_verification": protocol_signature,
        "blind_test_groups": sorted(REQUIRED_BLIND_GROUPS),
        "no_sealed_imitation": True,
    }


def build_attestation(
    *,
    protocol: dict[str, Any],
    inventory_check: dict[str, Any],
    blind_test_manifest: dict[str, Any],
    operator: str,
    attested_at: str,
    artifact_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    trusted_public_keys: dict[str, str] | None = None,
) -> dict[str, Any]:
    # Authenticate caller-supplied qualification evidence before using any of
    # its claims or diagnosing downstream inventory/protocol bindings.
    qualification = validate_blind_qualification(
        protocol=protocol,
        blind_test_manifest=blind_test_manifest,
        operator=operator,
        artifact_root=artifact_root,
        trusted_public_keys=trusted_public_keys,
        require_distinct_signer=False,
    )
    inventory_signature = _verify_historical_inventory_check(
        inventory_check,
        repo_root=repo_root if repo_root is not None else artifact_root,
        trusted_public_keys=trusted_public_keys,
    )
    if not str(operator).strip() or not str(attested_at).strip():
        raise MeasurementV2Error("operator and attested_at are required")
    return {
        "artifact_schema": "dftr.measurement.operator_attestation.v2",
        "status": "qualified",
        "protocol_sha256": _canonical_sha256(protocol),
        "operator": str(operator).strip(),
        "attested_at": str(attested_at).strip(),
        "evaluator_commit": blind_test_manifest["evaluator_commit"],
        "dependency_lock_sha256": blind_test_manifest["dependency_lock_sha256"],
        "fixture_pack_sha256": blind_test_manifest["fixture_pack_sha256"],
        "blind_test_manifest_sha256": _canonical_sha256(blind_test_manifest),
        "blind_manifest_signature_verification": qualification["signature_verification"],
        "historical_inventory_verified": True,
        "historical_inventory_sha256": _canonical_sha256(inventory_check),
        "historical_inventory_signature_verification": inventory_signature,
        "blind_test_groups": sorted(REQUIRED_BLIND_GROUPS),
        "no_sealed_imitation": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.measurement_v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("verify-inventory")
    inventory_parser.add_argument("inventory")
    inventory_parser.add_argument("--repo-root", required=True)
    protocol_parser = subparsers.add_parser("validate-protocol")
    protocol_parser.add_argument("protocol")
    protocol_parser.add_argument("--artifact-root", required=True)
    protocol_parser.add_argument("--trusted-keys", required=True)
    transfer_parser = subparsers.add_parser("prepare-protocol-transfer")
    transfer_parser.add_argument("candidate")
    transfer_parser.add_argument("--expected-sha256", required=True)
    transfer_parser.add_argument("--artifact-root", required=True)
    transfer_parser.add_argument("--trusted-keys", required=True)
    report_parser = subparsers.add_parser("validate-report")
    report_parser.add_argument("report")
    report_parser.add_argument("--protocol", required=True)
    report_parser.add_argument("--artifact-root", required=True)
    report_parser.add_argument("--trusted-keys", required=True)
    attestation_parser = subparsers.add_parser("attest")
    attestation_parser.add_argument("protocol")
    attestation_parser.add_argument("inventory_check")
    attestation_parser.add_argument("blind_test_manifest")
    attestation_parser.add_argument("--operator", required=True)
    attestation_parser.add_argument("--attested-at", required=True)
    attestation_parser.add_argument("--artifact-root", required=True)
    attestation_parser.add_argument("--repo-root")
    attestation_parser.add_argument("--trusted-keys", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-inventory":
            result = verify_historical_inventory(_load(args.inventory), repo_root=args.repo_root)
        elif args.command == "validate-protocol":
            result = validate_protocol(
                _load(args.protocol),
                artifact_root=args.artifact_root,
                trusted_public_keys=_load(args.trusted_keys),
            )
        elif args.command == "prepare-protocol-transfer":
            result = prepare_protocol_transfer(
                args.candidate,
                args.expected_sha256,
                artifact_root=args.artifact_root,
                trusted_public_keys=_load(args.trusted_keys),
            )
        elif args.command == "validate-report":
            result = validate_report_v2(
                _load(args.report),
                protocol=_load(args.protocol),
                artifact_root=args.artifact_root,
                trusted_public_keys=_load(args.trusted_keys),
            )
        else:
            result = build_attestation(
                protocol=_load(args.protocol),
                inventory_check=_load(args.inventory_check),
                blind_test_manifest=_load(args.blind_test_manifest),
                operator=args.operator,
                attested_at=args.attested_at,
                artifact_root=args.artifact_root,
                repo_root=args.repo_root,
                trusted_public_keys=_load(args.trusted_keys),
            )
    except (OSError, json.JSONDecodeError, MeasurementV2Error) as error:
        print(f"measurement-v2: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
