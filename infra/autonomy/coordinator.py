#!/usr/bin/env python3
"""Event-driven Humanwrite continuation coordinator.

The ten-minute polling path uses only stdlib HTTP and macOS Keychain access.
Codex is invoked only for a newly observed terminal transition or a configured
coarse progress milestone, never on ordinary polls. Persistent state lives
under the gitignored .operator tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[2]
CONTROL_PATH = ROOT / "progress" / "autonomy.json"
STATE_DIR = ROOT / ".operator" / "autonomy"
RUNTIME_PATH = STATE_DIR / "runtime.json"
LIVE_PATH = STATE_DIR / "live.json"
AUDIT_PATH = STATE_DIR / "audit.json"
LOCK_PATH = STATE_DIR / "codex.lock"
NATIVE_GOAL_LEASE_PATH = STATE_DIR / "native-goal.lease"
GATEWAY_URL = "https://bassimfaizal--humanwrite-gpu-gateway-gateway.modal.run"
KEYCHAIN_SERVICE = "humanwrite-gateway-token"
TERMINAL = {"completed", "failed", "cancelled", "reaped", "launch_failed"}
CODEX = Path("/Applications/Codex.app/Contents/Resources/codex")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _gateway_token() -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("gateway token is missing from Keychain")
    return token


def _run_status(run_id: str, token: str) -> dict[str, Any]:
    call = request.Request(
        f"{GATEWAY_URL}/status/{run_id}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "humanwrite-coordinator/1"},
    )
    try:
        with request.urlopen(call, timeout=30) as response:
            value = json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"gateway status {run_id} failed with HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"gateway status {run_id} was not an object")
    return value


def terminal_signature(generation: int, statuses: dict[str, dict[str, Any]]) -> str | None:
    terminal = sorted(
        (run_id, str(state.get("status")))
        for run_id, state in statuses.items()
        if str(state.get("status")) in TERMINAL
    )
    if not terminal:
        return None
    payload = {"generation": generation, "terminal": terminal}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def milestone_signature(
    generation: int,
    monitored_runs: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
) -> str | None:
    """Return a stable signature for newly reached coarse progress milestones.

    Milestones are opt-in per run through ``progress_target``. This keeps the
    zero-token poll cheap and avoids waking Codex for every counter change.
    """

    reached: list[tuple[str, int]] = []
    for item in monitored_runs:
        run_id = str((item or {}).get("run_id") or "")
        target = int((item or {}).get("progress_target") or 0)
        if not run_id or target <= 0:
            continue
        state = statuses.get(run_id, {})
        if str(state.get("status")) in TERMINAL:
            continue
        processed = int(state.get("records_processed") or 0)
        for fraction in item.get("progress_milestones", [0.5]):
            percentage = int(round(float(fraction) * 100))
            if percentage <= 0 or percentage >= 100:
                continue
            if processed >= target * float(fraction):
                reached.append((run_id, percentage))
    if not reached:
        return None
    # Unlike terminal transitions, a reached running milestone can remain in
    # the monitored set after a continuation increments the control generation.
    # Keep this signature generation-independent so it cannot wake repeatedly.
    _ = generation
    payload = {"milestones": sorted(reached)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def within_wake_budget(wake_times: list[float], now: float, maximum: int) -> bool:
    recent = [value for value in wake_times if now - float(value) < 86_400]
    return len(recent) < maximum


def successful_wake_times(runtime: dict[str, Any]) -> list[float]:
    """Return only completed, evidence-bearing Codex continuations."""

    return [
        float(item["started_at"])
        for item in runtime.get("wake_history", [])
        if item.get("succeeded") is True and item.get("started_at") is not None
    ]


def native_goal_lease_active(control: dict[str, Any], now: float) -> bool:
    """Prevent a fallback Codex launch while the native goal owns handoffs."""

    seconds = int(control.get("native_goal_lease_seconds", 1800))
    if seconds <= 0:
        return False
    try:
        modified = NATIVE_GOAL_LEASE_PATH.stat().st_mtime
    except OSError:
        return False
    return 0 <= now - modified < seconds


def should_wake(
    *,
    control: dict[str, Any],
    runtime: dict[str, Any],
    statuses: dict[str, dict[str, Any]],
    now: float,
) -> tuple[bool, str | None, str]:
    if control.get("enabled") is not True:
        return False, None, "disabled"
    if native_goal_lease_active(control, now):
        return False, None, "native_goal_lease_active"
    runs = control.get("monitored_runs")
    if not isinstance(runs, list) or not runs:
        return False, None, "no_monitored_runs"
    generation = int(control.get("generation", 0))
    signature = terminal_signature(generation, statuses)
    transition = "terminal"
    if signature is None:
        signature = milestone_signature(generation, runs, statuses)
        transition = "milestone"
    if signature is None:
        return False, None, "waiting_for_event_transition"
    handled = set(str(value) for value in runtime.get("handled_signatures", []))
    if signature in handled:
        return False, signature, f"{transition}_transition_already_handled"
    maximum = int(control.get("max_codex_wakes_per_24h", 4))
    wake_times = successful_wake_times(runtime)
    if not within_wake_budget(wake_times, now, maximum):
        return False, signature, "daily_codex_wake_budget_exhausted"
    return True, signature, f"new_{transition}_transition"


def should_run_scheduled_audit(
    *, control: dict[str, Any], runtime: dict[str, Any], now: float
) -> tuple[bool, str]:
    """Decide whether the 90-minute model-backed safety audit should run.

    The LaunchAgent supplies the coarse 90-minute cadence.  This additional
    recent-wake guard prevents a scheduled audit from duplicating useful work
    just performed by an event-triggered continuation.
    """

    if control.get("enabled") is not True:
        return False, "disabled"
    if native_goal_lease_active(control, now):
        return False, "native_goal_lease_active"
    maximum = int(control.get("max_codex_wakes_per_24h", 8))
    wake_times = successful_wake_times(runtime)
    if not within_wake_budget(wake_times, now, maximum):
        return False, "daily_codex_wake_budget_exhausted"
    recent_guard = int(control.get("audit_recent_continuation_seconds", 3600))
    successful_wakes = wake_times
    if successful_wakes and now - max(successful_wakes) < recent_guard:
        return False, "recent_continuation_already_checked_pipeline"
    return True, "scheduled_90m_audit_due"


def _continuation_prompt(
    control: dict[str, Any],
    statuses: dict[str, dict[str, Any]],
    *,
    invocation: str = "terminal_transition",
) -> str:
    compact_status = {
        run_id: {
            key: state.get(key)
            for key in (
                "status",
                "comparison",
                "records_processed",
                "records_failed",
                "actual_api_cost_usd",
                "cost_usd",
            )
            if state.get(key) is not None
        }
        for run_id, state in statuses.items()
    }
    return f"""Continue the Humanwrite research pipeline autonomously.

Core objective: obtain a Qwen3-14B model that rewrites AI-styled prose more humanly than its matched rewrite-SFT control while preserving facts, or reach a defensible negative conclusion under the frozen M3 protocol.

Read CLAUDE.md, RESEARCH_CONTEXT.md, research_reviews/m3_rewriting_14b_preregistration_2026-07-19.md, progress/autonomy.json, progress/status.json, FINDINGS.md, and recent git history. The user explicitly waived routine intermediate sign-offs and authorized the complete gate-controlled 4K/16K/46K M3 cycle using Qwen3-14B, subject to the frozen scientific gates and the existing $100 Modal plus $100 API caps. The 46K stage is authorized only if the 16K gate passes. Tier 3 detectors remain human-triggered only; budgets may never be raised automatically.

The deterministic coordinator observed this run state transition:
{json.dumps(compact_status, indent=2, sort_keys=True)}

Invocation type: {invocation}. If this is the scheduled 90-minute audit, inspect
the full pipeline for a missed handoff, stale monitor target, silent failure, or
completed job whose artifacts have not been validated. It is a real safety
review, not a request to poll repeatedly or manufacture work when the pipeline
is healthy.

Do meaningful next work now: validate completed artifacts, repair recoverable infrastructure failures, implement any missing fixed pipeline step, and launch the next already-authorized asynchronous job when ready. Do not sleep, idle-poll, or repeatedly check unchanged jobs. Preserve prospective controls, require matched SFT14/HUMANWRITE14 exposure, and never weaken data-quality, content-preservation, evaluation, or spend gates merely to advance.

Before exiting, update progress/status.json and progress/autonomy.json. Increment autonomy generation by exactly one and set monitored_runs to only the asynchronous run IDs whose transition should wake the next continuation. If no remote job is active but local work remains, complete that work during this invocation. If the core scientific answer is complete or genuinely requires user authority, set enabled=false and state the reason. Commit and push coherent changes.
"""


def _invoke_codex(
    control: dict[str, Any],
    statuses: dict[str, dict[str, Any]],
    *,
    invocation: str = "terminal_transition",
) -> dict[str, Any]:
    if not CODEX.is_file():
        raise RuntimeError(f"Codex CLI not found at {CODEX}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_log = STATE_DIR / f"codex-{stamp}.jsonl"
    last_message = STATE_DIR / f"codex-{stamp}-last.txt"
    command = [
        str(CODEX),
        "exec",
        "--json",
        "--output-last-message",
        str(last_message),
        "-C",
        str(ROOT),
        "-s",
        "danger-full-access",
        "-m",
        "gpt-5.4",
        "-c",
        'approval_policy="never"',
        "-c",
        'model_reasoning_effort="high"',
        _continuation_prompt(control, statuses, invocation=invocation),
    ]
    environment = dict(os.environ)
    environment["PATH"] = (
        "/opt/homebrew/bin:/Users/bassime/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    )
    with event_log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=int(control.get("codex_timeout_seconds", 7200)),
            check=False,
        )
    succeeded = (
        result.returncode == 0
        and last_message.is_file()
        and bool(last_message.read_text(encoding="utf-8").strip())
    )
    return {
        "started_at": started,
        "finished_at": time.time(),
        "return_code": result.returncode,
        "succeeded": succeeded,
        "event_log": str(event_log),
        "last_message": str(last_message),
    }


def tick(*, dry_run: bool = False) -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    control = _read_json(CONTROL_PATH, {})
    runtime = _read_json(
        RUNTIME_PATH,
        {"handled_signatures": [], "codex_wake_times": [], "wake_history": []},
    )
    now = time.time()
    monitored = control.get("monitored_runs") if isinstance(control, dict) else []
    statuses: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    if isinstance(monitored, list) and monitored:
        try:
            token = _gateway_token()
            for item in monitored:
                run_id = str((item or {}).get("run_id") or "")
                if not run_id:
                    continue
                try:
                    statuses[run_id] = _run_status(run_id, token)
                except Exception as exc:
                    errors[run_id] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            errors["gateway"] = f"{type(exc).__name__}: {exc}"
    wake, signature, reason = should_wake(
        control=control if isinstance(control, dict) else {},
        runtime=runtime if isinstance(runtime, dict) else {},
        statuses=statuses,
        now=now,
    )
    live = {
        "schema": "humanwrite.autonomy_live.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "generation": control.get("generation"),
        "enabled": control.get("enabled"),
        "decision": reason,
        "dry_run": dry_run,
        "statuses": statuses,
        "errors": errors,
        "token_use": "zero for polling; Codex only on a terminal or configured coarse milestone transition",
    }
    _atomic_json(LIVE_PATH, live)
    if not wake or dry_run or errors:
        return live
    if LOCK_PATH.exists():
        live["decision"] = "codex_lock_exists"
        _atomic_json(LIVE_PATH, live)
        return live
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    try:
        runtime.setdefault("codex_wake_times", []).append(now)
        runtime["codex_wake_times"] = [
            value for value in runtime["codex_wake_times"] if now - float(value) < 86_400
        ]
        _atomic_json(RUNTIME_PATH, runtime)
        invocation = (
            "progress_milestone"
            if reason == "new_milestone_transition"
            else "terminal_transition"
        )
        result = _invoke_codex(control, statuses, invocation=invocation)
        runtime = _read_json(RUNTIME_PATH, runtime)
        if result["succeeded"]:
            runtime.setdefault("handled_signatures", []).append(signature)
        runtime.setdefault("wake_history", []).append({"signature": signature, **result})
        _atomic_json(RUNTIME_PATH, runtime)
        live["decision"] = (
            "codex_continuation_finished"
            if result["succeeded"]
            else "codex_continuation_failed"
        )
        live["codex_result"] = result
        _atomic_json(LIVE_PATH, live)
        return live
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def audit(*, dry_run: bool = False) -> dict[str, Any]:
    """Run one bounded model-backed pipeline audit on the 90-minute cadence."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    control = _read_json(CONTROL_PATH, {})
    runtime = _read_json(
        RUNTIME_PATH,
        {"handled_signatures": [], "codex_wake_times": [], "wake_history": []},
    )
    now = time.time()
    statuses: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    monitored = control.get("monitored_runs") if isinstance(control, dict) else []
    if isinstance(monitored, list) and monitored:
        try:
            token = _gateway_token()
            for item in monitored:
                run_id = str((item or {}).get("run_id") or "")
                if run_id:
                    try:
                        statuses[run_id] = _run_status(run_id, token)
                    except Exception as exc:
                        errors[run_id] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            errors["gateway"] = f"{type(exc).__name__}: {exc}"
    due, reason = should_run_scheduled_audit(
        control=control if isinstance(control, dict) else {},
        runtime=runtime if isinstance(runtime, dict) else {},
        now=now,
    )
    report = {
        "schema": "humanwrite.autonomy_audit.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "generation": control.get("generation"),
        "enabled": control.get("enabled"),
        "decision": reason,
        "dry_run": dry_run,
        "statuses": statuses,
        "errors": errors,
        "token_use": "one bounded Codex turn only when the 90-minute audit is due",
    }
    _atomic_json(AUDIT_PATH, report)
    if not due or dry_run or errors:
        return report
    if LOCK_PATH.exists():
        report["decision"] = "codex_lock_exists"
        _atomic_json(AUDIT_PATH, report)
        return report
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    try:
        current_signature = terminal_signature(
            int(control.get("generation", 0)), statuses
        )
        runtime.setdefault("codex_wake_times", []).append(now)
        runtime["codex_wake_times"] = [
            value for value in runtime["codex_wake_times"]
            if now - float(value) < 86_400
        ]
        _atomic_json(RUNTIME_PATH, runtime)
        result = _invoke_codex(
            control, statuses, invocation="scheduled_90m_safety_audit"
        )
        runtime = _read_json(RUNTIME_PATH, runtime)
        if current_signature is not None and result["succeeded"]:
            runtime.setdefault("handled_signatures", []).append(current_signature)
        runtime.setdefault("wake_history", []).append(
            {"signature": f"scheduled-audit-{int(now)}", **result}
        )
        _atomic_json(RUNTIME_PATH, runtime)
        report["decision"] = (
            "scheduled_audit_finished"
            if result["succeeded"]
            else "scheduled_audit_failed"
        )
        report["codex_result"] = result
        _atomic_json(AUDIT_PATH, report)
        return report
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Event-driven Humanwrite autonomy coordinator")
    parser.add_argument(
        "command", choices=("tick", "audit", "status"), nargs="?", default="tick"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        value = _read_json(LIVE_PATH, {"status": "never_run"})
    elif args.command == "audit":
        value = audit(dry_run=args.dry_run)
    else:
        value = tick(dry_run=args.dry_run)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
