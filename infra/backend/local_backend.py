from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from .policy import (
    append_event,
    budget_snapshot,
    has_api_capacity,
    has_capacity,
    read_events,
    run_snapshot,
    validate_launch,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = ROOT / ".swarmy" / "local_gpu"
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _state_dir(path: str | Path | None = None) -> Path:
    base = Path(path or os.environ.get("DFTR_GPU_LOCAL_STATE_DIR", DEFAULT_STATE_DIR)).resolve()
    base.mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)
    (base / "payloads").mkdir(exist_ok=True)
    (base / "pids").mkdir(exist_ok=True)
    return base


def _events_path(state_dir: Path) -> Path:
    return state_dir / "events.jsonl"


def _events(state_dir: Path) -> list[dict[str, Any]]:
    return read_events(_events_path(state_dir))


def _record(state_dir: Path, event: dict[str, Any]) -> dict[str, Any]:
    return append_event(_events_path(state_dir), event)


def _pid_path(state_dir: Path, run_id: str) -> Path:
    return state_dir / "pids" / f"{run_id}.pid"


def _log_path(state_dir: Path, run_id: str) -> Path:
    return state_dir / "logs" / f"{run_id}.log"


def _artifact_dir(state_dir: Path, run_id: str) -> Path:
    return state_dir / "artifacts" / run_id


def _count_generated_tokens(state_dir: Path, run_id: str) -> int:
    samples_path = _artifact_dir(state_dir, run_id) / "samples.jsonl"
    if not samples_path.exists():
        return 0
    total = 0
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(
            row.get("output")
            or row.get("generated_completion")
            or row.get("completion")
            or ""
        )
        total += len(TOKEN_RE.findall(text))
    return total


def submit_local(payload: dict[str, Any], state_dir: str | Path | None = None) -> dict[str, Any]:
    state = _state_dir(state_dir)
    policy = validate_launch(payload)
    run_id = str(payload["run_id"])
    events = _events(state)
    if run_snapshot(events, run_id):
        raise ValueError("run_id already exists")
    if not has_capacity(events, policy.worst_case_cost_usd):
        raise ValueError("monthly GPU budget exhausted")
    if not has_api_capacity(events, policy.api_reserved_cost_usd):
        raise ValueError("monthly API budget exhausted")
    config_path = state / "payloads" / f"{run_id}.config.json"
    payload_path = state / "payloads" / f"{run_id}.payload.json"
    config_path.write_text(json.dumps(payload["config"], sort_keys=True), encoding="utf-8")
    worker_payload = dict(payload)
    worker_payload.update(
        {
            "timeout_seconds": policy.timeout_seconds,
            "reserved_cost_usd": policy.worst_case_cost_usd,
            "api_reserved_cost_usd": policy.api_reserved_cost_usd,
            "state_dir": str(state),
            "config_path": str(config_path),
        }
    )
    payload_path.write_text(json.dumps(worker_payload, sort_keys=True), encoding="utf-8")
    _record(
        state,
        {
            "kind": "run",
            "run_id": run_id,
            "comparison": policy.comparison_id,
            "status": "reserved",
            "budget_class": policy.budget_class,
            "gpu": policy.gpu,
            "task_kind": policy.task_kind,
            "timeout_seconds": policy.timeout_seconds,
            "reserved_cost_usd": policy.worst_case_cost_usd,
            "api_reserved_cost_usd": policy.api_reserved_cost_usd,
            "config_hash": payload["config_hash"],
            "git_sha": payload["git_sha"],
            "started_at": time.time(),
        },
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "infra.backend.local_worker", "--payload", str(payload_path)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        _record(
            state,
            {
                "kind": "run_update",
                "run_id": run_id,
                "status": "launch_failed",
            },
        )
        raise
    _pid_path(state, run_id).write_text(str(process.pid), encoding="utf-8")
    _record(
        state,
        {
            "kind": "run_update",
            "run_id": run_id,
            "status": "running",
            "function_call_id": f"local-{process.pid}",
            "pid": process.pid,
            "log_path": str(_log_path(state, run_id)),
        },
    )
    return {
        "run_id": run_id,
        "status": "running",
        "budget_class": policy.budget_class,
        "reserved_cost_usd": policy.worst_case_cost_usd,
        "api_reserved_cost_usd": policy.api_reserved_cost_usd,
        "backend": "local",
    }


def status_local(run_id: str, state_dir: str | Path | None = None) -> dict[str, Any]:
    state = run_snapshot(_events(_state_dir(state_dir)), run_id)
    if not state:
        raise FileNotFoundError("run not found")
    return state


def logs_local(run_id: str, tail: int = 200, state_dir: str | Path | None = None) -> dict[str, Any]:
    state = _state_dir(state_dir)
    if not run_snapshot(_events(state), run_id):
        raise FileNotFoundError("run not found")
    path = _log_path(state, run_id)
    if not path.exists():
        return {"run_id": run_id, "logs": ""}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"run_id": run_id, "logs": "\n".join(lines[-tail:])}


def cancel_local(run_id: str, state_dir: str | Path | None = None) -> dict[str, Any]:
    state = _state_dir(state_dir)
    snapshot = run_snapshot(_events(state), run_id)
    if not snapshot:
        raise FileNotFoundError("run not found")
    if snapshot.get("status") in {"completed", "failed", "cancelled", "reaped", "launch_failed"}:
        return {"run_id": run_id, "status": snapshot["status"]}
    pid_path = _pid_path(state, run_id)
    if pid_path.exists():
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
    elapsed = max(0.0, time.time() - float(snapshot.get("started_at") or time.time()))
    timeout = max(1.0, float(snapshot.get("timeout_seconds") or 1.0))
    reserved = float(snapshot.get("reserved_cost_usd") or 0.0)
    actual = min(reserved, reserved * elapsed / timeout)
    _record(
        state,
        {
            "kind": "run_update",
            "run_id": run_id,
            "status": "cancelled",
            "finished_at": time.time(),
            "accel_seconds": round(elapsed, 3),
            "tokens": _count_generated_tokens(state, run_id),
            "actual_cost_usd": round(actual, 6),
        },
    )
    return {"run_id": run_id, "status": "cancelled"}


def budget_local(state_dir: str | Path | None = None) -> dict[str, float]:
    return budget_snapshot(_events(_state_dir(state_dir)))
