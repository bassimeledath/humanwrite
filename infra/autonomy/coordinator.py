#!/usr/bin/env python3
"""Event-driven Humanwrite continuation coordinator.

The ten-minute polling path uses only stdlib HTTP and macOS Keychain access.
Codex is invoked once per newly observed terminal run-state transition, never
on ordinary polls. Persistent state lives under the gitignored .operator tree.
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
LOCK_PATH = STATE_DIR / "codex.lock"
GATEWAY_URL = "https://bassimfaizal--humanwrite-gpu-gateway-gateway.modal.run"
KEYCHAIN_SERVICE = "humanwrite-gateway-token"
TERMINAL = {"completed", "failed", "cancelled", "reaped", "launch_failed"}
CODEX = Path("/opt/homebrew/bin/codex")


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


def within_wake_budget(wake_times: list[float], now: float, maximum: int) -> bool:
    recent = [value for value in wake_times if now - float(value) < 86_400]
    return len(recent) < maximum


def should_wake(
    *,
    control: dict[str, Any],
    runtime: dict[str, Any],
    statuses: dict[str, dict[str, Any]],
    now: float,
) -> tuple[bool, str | None, str]:
    if control.get("enabled") is not True:
        return False, None, "disabled"
    runs = control.get("monitored_runs")
    if not isinstance(runs, list) or not runs:
        return False, None, "no_monitored_runs"
    signature = terminal_signature(int(control.get("generation", 0)), statuses)
    if signature is None:
        return False, None, "waiting_for_terminal_transition"
    handled = set(str(value) for value in runtime.get("handled_signatures", []))
    if signature in handled:
        return False, signature, "terminal_transition_already_handled"
    maximum = int(control.get("max_codex_wakes_per_24h", 4))
    wake_times = [float(value) for value in runtime.get("codex_wake_times", [])]
    if not within_wake_budget(wake_times, now, maximum):
        return False, signature, "daily_codex_wake_budget_exhausted"
    return True, signature, "new_terminal_transition"


def _continuation_prompt(control: dict[str, Any], statuses: dict[str, dict[str, Any]]) -> str:
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

Core objective: obtain a model that demonstrably writes more like a human than its matched SFT control, or reach a defensible negative conclusion about the current method family.

Read CLAUDE.md, RESEARCH_CONTEXT.md, progress/autonomy.json, progress/status.json, FINDINGS.md, and recent git history. The user explicitly waived intermediate milestone sign-offs for this bounded cycle and authorized the preregistered 4K/16K data-scale ladder, but not the gated 46K cell. Tier 3 detectors remain human-triggered only. Respect the existing $100 Modal and $100 API caps.

The deterministic coordinator observed this run state transition:
{json.dumps(compact_status, indent=2, sort_keys=True)}

Do meaningful next work now: validate completed artifacts, repair recoverable failures, implement any missing fixed pipeline step, and launch the next safe asynchronous jobs when ready. Do not sleep, idle-poll, or repeatedly check unchanged jobs. Preserve prospective controls and never weaken quality gates merely to advance.

Before exiting, update progress/status.json and progress/autonomy.json. Increment autonomy generation by exactly one and set monitored_runs to only the asynchronous run IDs whose transition should wake the next continuation. If no remote job is active but local work remains, complete that work during this invocation. If the core scientific answer is complete or genuinely requires user authority, set enabled=false and state the reason. Commit and push coherent changes.
"""


def _invoke_codex(control: dict[str, Any], statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
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
        "-c",
        'approval_policy="never"',
        "-c",
        'mcp_servers.braintrust.command="true"',
        _continuation_prompt(control, statuses),
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
    return {
        "started_at": started,
        "finished_at": time.time(),
        "return_code": result.returncode,
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
        "token_use": "zero for polling; Codex only on a new terminal transition",
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
        runtime.setdefault("handled_signatures", []).append(signature)
        runtime.setdefault("codex_wake_times", []).append(now)
        runtime["codex_wake_times"] = [
            value for value in runtime["codex_wake_times"] if now - float(value) < 86_400
        ]
        _atomic_json(RUNTIME_PATH, runtime)
        result = _invoke_codex(control, statuses)
        runtime = _read_json(RUNTIME_PATH, runtime)
        runtime.setdefault("wake_history", []).append({"signature": signature, **result})
        _atomic_json(RUNTIME_PATH, runtime)
        live["decision"] = "codex_continuation_finished"
        live["codex_result"] = result
        _atomic_json(LIVE_PATH, live)
        return live
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Event-driven Humanwrite autonomy coordinator")
    parser.add_argument("command", choices=("tick", "status"), nargs="?", default="tick")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        value = _read_json(LIVE_PATH, {"status": "never_run"})
    else:
        value = tick(dry_run=args.dry_run)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
