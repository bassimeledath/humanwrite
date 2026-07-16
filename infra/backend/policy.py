"""Pure policy and append-only state helpers used by gateway and reaper."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any


MONTHLY_GPU_CAP_USD = 40.0
MONTHLY_API_CAP_USD = 100.0
BUDGET_CLASSES = {
    "smoke": {"max_seconds": 20 * 60, "max_gpus": 1},
    "screen": {"max_seconds": 2 * 60 * 60, "max_gpus": 1},
    "promo": {"max_seconds": 8 * 60 * 60, "max_gpus": 1},
}

# Modal public list prices retrieved 2026-07-15.  The 20% reserve covers CPU,
# memory, startup, and price drift.  The Modal dashboard hard cap is still the
# authoritative outer boundary.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
ALLOWED_COMMAND_PREFIX = ["python", "-m", "experiments.runner"]
TERMINAL = {"completed", "failed", "cancelled", "reaped", "launch_failed"}
UNRESOLVED_REVISION_PREFIX = "__M1_RESOLVE_"


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class LaunchPolicy:
    comparison_id: str
    budget_class: str
    timeout_seconds: int
    gpu: str
    worst_case_cost_usd: float
    task_kind: str = "experiment"
    api_reserved_cost_usd: float = 0.0


def canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def revision_is_unresolved(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.startswith(UNRESOLVED_REVISION_PREFIX)


def validate_launch(payload: dict[str, Any]) -> LaunchPolicy:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise PolicyError("config must be an object")
    if payload.get("config_hash") != canonical_hash(config):
        raise PolicyError("config hash mismatch")
    run = config.get("run") or {}
    comparison_id = str(run.get("comparison_id", ""))
    if not comparison_id:
        raise PolicyError("config.run.comparison_id is required")
    prereg = payload.get("preregistration") or {}
    if (
        prereg.get("kind") != "prereg"
        or prereg.get("status") != "open"
        or prereg.get("comparison") != comparison_id
    ):
        raise PolicyError("open matching preregistration required")
    budget_class = str(payload.get("budget_class", ""))
    if budget_class not in BUDGET_CLASSES:
        raise PolicyError("unknown budget class")
    if run.get("budget_class") not in (None, budget_class):
        raise PolicyError("budget class mismatch")
    compute = config.get("compute") or {}
    if int(compute.get("gpus", 1)) != 1:
        raise PolicyError("single-GPU rule violated")
    timeout_seconds = int(
        compute.get("timeout_min", BUDGET_CLASSES[budget_class]["max_seconds"] // 60)
    ) * 60
    if timeout_seconds <= 0 or timeout_seconds > BUDGET_CLASSES[budget_class]["max_seconds"]:
        raise PolicyError("requested timeout exceeds budget class")
    task_kind = str(run.get("task_kind", "experiment"))
    if task_kind not in {"experiment", "brief_synthesis", "source_materialization"}:
        raise PolicyError("unsupported task_kind")
    gpu = str(compute.get("gpu", "L40S")).upper() if task_kind == "experiment" else "CPU"
    if task_kind == "experiment" and gpu not in GPU_USD_PER_SECOND:
        raise PolicyError(f"unsupported GPU: {gpu}")
    base_model = str((config.get("model") or {}).get("base", ""))
    workflow_step = str((config.get("workflow") or {}).get("step", "")).casefold()
    if "14B" in base_model.upper() and not payload.get("human_scaleup_approved"):
        raise PolicyError("14B scale-up lacks human approval")
    if workflow_step in {"train_sft", "sample_sweep"} and revision_is_unresolved(
        (config.get("model") or {}).get("revision")
    ):
        raise PolicyError(
            "M1 evidentiary experiment jobs require model.revision to be a resolved immutable revision"
        )
    api_reserved = 0.0
    if task_kind == "experiment":
        command = run.get("command", ALLOWED_COMMAND_PREFIX)
        if not isinstance(command, list) or command[:3] != ALLOWED_COMMAND_PREFIX:
            raise PolicyError("command is outside the allowlist")
        worst = round(timeout_seconds * GPU_USD_PER_SECOND[gpu] * 1.20, 6)
    elif task_kind == "brief_synthesis":
        data = config.get("data") or {}
        api = config.get("api") or {}
        for field in ("input_uri", "output_uri"):
            if not str(data.get(field, "")).startswith("modal-volume://humanwrite-checkpoints/"):
                raise PolicyError(f"brief_synthesis data.{field} must use the checkpoint volume")
        input_sha = str(data.get("input_sha256", ""))
        if len(input_sha) != 64 or any(character not in "0123456789abcdef" for character in input_sha):
            raise PolicyError("brief_synthesis requires lowercase data.input_sha256")
        max_records = int(data.get("max_records", 0))
        if max_records <= 0 or max_records > 50_000:
            raise PolicyError("brief_synthesis data.max_records must be between 1 and 50000")
        if not str(api.get("model", "")):
            raise PolicyError("brief_synthesis requires a frozen api.model")
        if api.get("force_empty_quotations") is True:
            max_missing = int((config.get("recovery") or {}).get("max_missing_records", 0))
            if budget_class != "smoke" or not 1 <= max_missing <= 16:
                raise PolicyError(
                    "quote-free recovery requires smoke budget and 1..16 max missing records"
                )
        api_reserved = float(api.get("max_cost_usd", 0.0))
        if api_reserved <= 0 or api_reserved > MONTHLY_API_CAP_USD:
            raise PolicyError("brief_synthesis requires api.max_cost_usd within the monthly cap")
        worst = 0.0
    else:
        source = config.get("source") or {}
        data = config.get("data") or {}
        for field in ("train_output_uri", "dev_output_uri", "manifest_output_uri"):
            if not str(data.get(field, "")).startswith("modal-volume://humanwrite-checkpoints/"):
                raise PolicyError(f"source_materialization data.{field} must use the checkpoint volume")
        required_source = ("dataset_id", "dataset_config", "revision", "split", "files")
        if any(not source.get(field) for field in required_source):
            raise PolicyError("source_materialization requires a fully pinned source")
        selection = config.get("selection") or {}
        if int(selection.get("corpus_size", 0)) <= 0 or int(selection.get("corpus_size", 0)) > 5000:
            raise PolicyError("source_materialization corpus_size must be between 1 and 5000")
        worst = 0.0
    return LaunchPolicy(
        comparison_id, budget_class, timeout_seconds, gpu, worst,
        task_kind=task_kind, api_reserved_cost_usd=api_reserved,
    )


def utc_month(timestamp: float | None = None) -> str:
    instant = datetime.fromtimestamp(timestamp, timezone.utc) if timestamp else datetime.now(timezone.utc)
    return instant.strftime("%Y-%m")


def read_events(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    events = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).timestamp())
    serialized = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return row


def run_snapshot(events: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    relevant = [event for event in events if event.get("run_id") == run_id]
    if not relevant:
        return None
    result: dict[str, Any] = {}
    for event in relevant:
        result.update({key: value for key, value in event.items() if value is not None})
    return result


def budget_snapshot(events: list[dict[str, Any]], month: str | None = None) -> dict[str, float]:
    month = month or utc_month()
    run_ids = {str(event["run_id"]) for event in events if event.get("run_id")}
    gpu_committed = 0.0
    for run_id in run_ids:
        launch = next(
            (event for event in events if event.get("run_id") == run_id and event.get("kind") == "run"),
            {},
        )
        state = run_snapshot(events, run_id) or {}
        billing_month = str(launch.get("billing_month") or utc_month(float(launch.get("ts", 0))))
        if billing_month != month:
            continue
        if state.get("status") in TERMINAL:
            gpu_committed += float(state.get("actual_cost_usd") or 0.0)
        else:
            gpu_committed += float(state.get("reserved_cost_usd") or 0.0)
    api_spend = sum(
        float(event.get("cost_usd") or 0.0)
        for event in events
        if event.get("kind") == "api_cost" and utc_month(float(event.get("ts", 0))) == month
    )
    for run_id in run_ids:
        launch = next(
            (event for event in events if event.get("run_id") == run_id and event.get("kind") == "run"),
            {},
        )
        if str(launch.get("billing_month") or utc_month(float(launch.get("ts", 0)))) != month:
            continue
        state = run_snapshot(events, run_id) or {}
        if state.get("task_kind") != "brief_synthesis":
            continue
        if state.get("status") not in TERMINAL:
            already_reported = sum(
                float(event.get("cost_usd") or 0.0)
                for event in events
                if event.get("kind") == "api_cost" and event.get("run_id") == run_id
            )
            reservation = float(state.get("api_reserved_cost_usd") or 0.0)
            api_spend += max(0.0, reservation - already_reported)
    return {
        "gpu_cap_usd": MONTHLY_GPU_CAP_USD,
        "gpu_committed_usd": round(gpu_committed, 6),
        "gpu_remaining_usd": round(max(0.0, MONTHLY_GPU_CAP_USD - gpu_committed), 6),
        "api_cap_usd": MONTHLY_API_CAP_USD,
        "api_spend_usd": round(api_spend, 6),
        "api_remaining_usd": round(max(0.0, MONTHLY_API_CAP_USD - api_spend), 6),
    }


def has_capacity(events: list[dict[str, Any]], requested_usd: float) -> bool:
    return budget_snapshot(events)["gpu_remaining_usd"] >= requested_usd


def has_api_capacity(events: list[dict[str, Any]], requested_usd: float) -> bool:
    return budget_snapshot(events)["api_remaining_usd"] >= requested_usd


def accrued_gpu_spend(events: list[dict[str, Any]], now: float) -> float:
    """Conservative accrued GPU spend for independent reaper decisions."""
    total = 0.0
    run_ids = {str(event["run_id"]) for event in events if event.get("run_id")}
    for run_id in run_ids:
        launch = next(
            (event for event in events if event.get("run_id") == run_id and event.get("kind") == "run"),
            {},
        )
        state = run_snapshot(events, run_id) or {}
        if state.get("status") in TERMINAL:
            total += float(state.get("actual_cost_usd") or 0.0)
            continue
        reserved = float(state.get("reserved_cost_usd") or 0.0)
        timeout = max(1.0, float(state.get("timeout_seconds") or 1.0))
        elapsed = max(0.0, now - float(state.get("started_at") or now))
        total += reserved * min(1.0, elapsed / timeout)
    return round(total, 6)


def authorized(header: str | None, expected_token: str) -> bool:
    if not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:], expected_token)
