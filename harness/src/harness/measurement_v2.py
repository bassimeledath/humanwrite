"""Operator-owned measurement-v2 protocol, report, and attestation surfaces."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

from .metrics.distribution_v2 import MeasurementV2Error
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


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MeasurementV2Error("measurement artifact must be a JSON object")
    return value


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
        "artifact_sets": results,
    }


def protocol_readiness(protocol: dict[str, Any]) -> dict[str, Any]:
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
        "human_panels_sha256",
        "bandwidths_sha256",
        "power_plan_sha256",
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
        panel_ids.extend(panel.get("document_ids") or [])
    if panel_ids and len(panel_ids) != len(set(panel_ids)):
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
    return {
        "artifact_schema": "dftr.measurement.protocol_readiness.v2",
        "status": "ready" if not reasons else "fail_closed",
        "reasons": sorted(set(reasons)),
    }


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    result = protocol_readiness(protocol)
    if result["status"] != "ready":
        raise MeasurementV2Error(
            "measurement v2 protocol is not ready: " + ", ".join(result["reasons"])
        )
    return result


def prepare_protocol_transfer(
    candidate_path: str | Path, expected_sha256: str
) -> dict[str, Any]:
    path = Path(candidate_path)
    if _sha256(path) != _require_sha(expected_sha256, "expected_sha256"):
        raise MeasurementV2Error("protocol candidate SHA-256 mismatch")
    candidate = _load(path)
    if candidate.get("artifact_schema") != "dftr.measurement.protocol_candidate.v2":
        raise MeasurementV2Error("unexpected protocol candidate schema")
    if (candidate.get("operator_review") or {}).get("approved") is not True:
        raise MeasurementV2Error("protocol candidate lacks operator approval")
    transferred = dict(candidate)
    transferred["artifact_schema"] = "dftr.measurement.protocol.v2"
    transferred["frozen"] = True
    transferred["status"] = "ready"
    transferred["operator_approval"] = dict(candidate["operator_review"])
    transferred.pop("operator_review", None)
    validate_protocol(transferred)
    return transferred


def validate_report_v2(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("artifact_schema") != "dftr.measurement.report.v2":
        raise MeasurementV2Error("unexpected measurement report schema")
    if report.get("evidence_class") not in {"prospective_screen", "post_hoc_shadow"}:
        raise MeasurementV2Error("v2 report has invalid evidence_class")
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
    validate_selection_firewall(report.get("checkpoint_manifest") or {})
    distribution = report.get("distribution") or {}
    if (
        _require_int(distribution.get("documents_per_cell"), "report.distribution.documents_per_cell", minimum=1) != n
        or _require_int(distribution.get("human_documents_per_panel"), "report.distribution.human_documents_per_panel", minimum=1) != n
    ):
        raise MeasurementV2Error("distribution cells must all use report n")
    _require_sha(distribution.get("bandwidth_sha256"), "report.distribution.bandwidth_sha256")
    for field in (
        "candidate_mmd2_unbiased",
        "control_mmd2_unbiased",
        "human_floor_mmd2_unbiased",
    ):
        value = distribution.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise MeasurementV2Error(f"report.distribution.{field} must be finite")
    _require_int(distribution.get("permutation_seed"), "report.distribution.permutation_seed")
    if (report.get("quality") or {}).get("status") == "measured":
        linkage = report.get("quality_linkage") or {}
        if linkage.get("status") != "verified" or _int_or_zero(linkage.get("matched_pairs")) != n:
            raise MeasurementV2Error("measured quality lacks one-to-one prompt linkage")
    if (report.get("repetition") or {}).get("status") == "underpowered":
        if (report.get("promotion") or {}).get("eligible") is True:
            raise MeasurementV2Error("underpowered repetition cannot promote")
    repetition = report.get("repetition") or {}
    if _require_int(repetition.get("documents_per_panel"), "report.repetition.documents_per_panel", minimum=1) != n:
        raise MeasurementV2Error("repetition must use matched report n")
    if repetition.get("power_plan_passed") is not True and repetition.get("status") != "underpowered":
        raise MeasurementV2Error("repetition without power evidence must be underpowered")
    authorship = report.get("authorship") or {}
    if authorship.get("grouped") is not True or _require_int(
        authorship.get("fit_count"), "report.authorship.fit_count", minimum=1
    ) < 1:
        raise MeasurementV2Error("authorship uncertainty must refit a grouped pipeline")
    if _require_int(
        authorship.get("effective_clusters"), "report.authorship.effective_clusters", minimum=1
    ) < 64 and authorship.get("status") != "underpowered":
        raise MeasurementV2Error("authorship with fewer than 64 clusters is underpowered")
    return {"status": "pass", "artifact_schema": "dftr.measurement.report_check.v2"}


def build_attestation(
    *,
    protocol: dict[str, Any],
    inventory_check: dict[str, Any],
    blind_test_manifest: dict[str, Any],
    operator: str,
    attested_at: str,
) -> dict[str, Any]:
    validate_protocol(protocol)
    if inventory_check.get("status") != "pass":
        raise MeasurementV2Error("historical inventory check did not pass")
    tests = blind_test_manifest.get("tests") or []
    observed = {str(row.get("name")) for row in tests if row.get("status") == "pass"}
    missing = sorted(REQUIRED_BLIND_GROUPS - observed)
    if missing:
        raise MeasurementV2Error("blind qualification is incomplete: " + ", ".join(missing))
    for field in ("evaluator_commit", "dependency_lock_sha256", "fixture_pack_sha256"):
        _require_sha(blind_test_manifest.get(field), f"blind_test_manifest.{field}")
    if blind_test_manifest.get("no_sealed_imitation") is not True:
        raise MeasurementV2Error("no-sealed-imitation attestation is absent")
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
        "historical_inventory_verified": True,
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
    transfer_parser = subparsers.add_parser("prepare-protocol-transfer")
    transfer_parser.add_argument("candidate")
    transfer_parser.add_argument("--expected-sha256", required=True)
    report_parser = subparsers.add_parser("validate-report")
    report_parser.add_argument("report")
    attestation_parser = subparsers.add_parser("attest")
    attestation_parser.add_argument("protocol")
    attestation_parser.add_argument("inventory_check")
    attestation_parser.add_argument("blind_test_manifest")
    attestation_parser.add_argument("--operator", required=True)
    attestation_parser.add_argument("--attested-at", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-inventory":
            result = verify_historical_inventory(_load(args.inventory), repo_root=args.repo_root)
        elif args.command == "validate-protocol":
            result = validate_protocol(_load(args.protocol))
        elif args.command == "prepare-protocol-transfer":
            result = prepare_protocol_transfer(args.candidate, args.expected_sha256)
        elif args.command == "validate-report":
            result = validate_report_v2(_load(args.report))
        else:
            result = build_attestation(
                protocol=_load(args.protocol),
                inventory_check=_load(args.inventory_check),
                blind_test_manifest=_load(args.blind_test_manifest),
                operator=args.operator,
                attested_at=args.attested_at,
            )
    except (OSError, json.JSONDecodeError, MeasurementV2Error) as error:
        print(f"measurement-v2: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
