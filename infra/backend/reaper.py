"""Independent five-minute Modal reaper deployment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from urllib import request as urlrequest

import modal

from .policy import (
    BUDGET_CLASSES,
    MONTHLY_GPU_CAP_USD,
    accrued_gpu_spend,
    append_event,
    read_events,
    run_snapshot,
)


STATE_PATH = "/state/events.jsonl"
state_volume = modal.Volume.from_name("humanwrite-gateway-state", create_if_missing=True)
reaper_secret = modal.Secret.from_name("humanwrite-reaper-auth")
source_root = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(source_root, remote_path="/root/infra_backend", copy=True)
)
app = modal.App("humanwrite-gpu-reaper")


def _notify(details: dict) -> None:
    webhook = os.environ.get("DFTR_ALERT_WEBHOOK_URL")
    if not webhook:
        return
    request = urlrequest.Request(
        webhook,
        data=json.dumps({"event": "reaper_kill", "project": "humanwrite", **details}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urlrequest.urlopen(request, timeout=10).read()
    except Exception as exc:
        print(f"alert delivery failed: {type(exc).__name__}")


@app.function(
    image=image,
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[reaper_secret],
    volumes={"/state": state_volume},
    timeout=240,
)
def reap() -> dict:
    state_volume.reload()
    events = read_events(STATE_PATH)
    now = time.time()
    killed = []
    run_ids = {str(event["run_id"]) for event in events if event.get("run_id")}
    global_exhausted = accrued_gpu_spend(events, now) >= MONTHLY_GPU_CAP_USD
    for run_id in sorted(run_ids):
        state = run_snapshot(events, run_id) or {}
        if state.get("status") not in {"reserved", "running"}:
            continue
        limit = BUDGET_CLASSES.get(str(state.get("budget_class")), {}).get("max_seconds", 0)
        elapsed = now - float(state.get("started_at") or now)
        reason = None
        if limit and elapsed > limit:
            reason = "wall_clock"
        elif global_exhausted:
            reason = "global_cap"
        if not reason:
            continue
        call_id = state.get("function_call_id")
        if call_id:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        timeout = max(1.0, float(state.get("timeout_seconds") or 1.0))
        reserved = float(state.get("reserved_cost_usd") or 0.0)
        actual = min(reserved, reserved * max(0.0, elapsed) / timeout)
        append_event(STATE_PATH, {
            "kind": "run_update",
            "run_id": run_id,
            "status": "reaped",
            "reaper_reason": reason,
            "finished_at": now,
            "actual_cost_usd": round(actual, 6),
        })
        killed.append({"run_id": run_id, "reason": reason})
        _notify(killed[-1])
    if killed:
        state_volume.commit()
    return {"checked": len(run_ids), "killed": killed}
