from __future__ import annotations

from infra.autonomy.coordinator import (
    milestone_signature,
    should_run_scheduled_audit,
    should_wake,
    successful_wake_times,
    terminal_signature,
    within_wake_budget,
)
import infra.autonomy.coordinator as coordinator


def control(**overrides):
    value = {
        "enabled": True,
        "generation": 3,
        "max_codex_wakes_per_24h": 4,
        "monitored_runs": [{"run_id": "run-a"}, {"run_id": "run-b"}],
    }
    value.update(overrides)
    return value


def test_running_poll_never_wakes_codex():
    wake, signature, reason = should_wake(
        control=control(),
        runtime={},
        statuses={"run-a": {"status": "running"}, "run-b": {"status": "running"}},
        now=1000.0,
    )
    assert wake is False
    assert signature is None
    assert reason == "waiting_for_event_transition"


def test_configured_progress_milestone_wakes_once():
    monitored = [
        {
            "run_id": "run-a",
            "progress_target": 200,
            "progress_milestones": [0.5],
        }
    ]
    statuses = {"run-a": {"status": "running", "records_processed": 100}}
    signature = milestone_signature(3, monitored, statuses)
    wake, observed, reason = should_wake(
        control=control(monitored_runs=monitored),
        runtime={},
        statuses=statuses,
        now=1000.0,
    )
    assert wake is True
    assert observed == signature
    assert reason == "new_milestone_transition"

    wake, _, reason = should_wake(
        control=control(monitored_runs=monitored),
        runtime={"handled_signatures": [signature]},
        statuses=statuses,
        now=1010.0,
    )
    assert wake is False
    assert reason == "milestone_transition_already_handled"


def test_progress_below_milestone_does_not_wake():
    monitored = [{"run_id": "run-a", "progress_target": 200}]
    wake, signature, reason = should_wake(
        control=control(monitored_runs=monitored),
        runtime={},
        statuses={"run-a": {"status": "running", "records_processed": 99}},
        now=1000.0,
    )
    assert wake is False
    assert signature is None
    assert reason == "waiting_for_event_transition"


def test_reached_progress_milestone_is_stable_across_generation_changes():
    monitored = [{"run_id": "run-a", "progress_target": 200}]
    statuses = {"run-a": {"status": "running", "records_processed": 100}}
    assert milestone_signature(3, monitored, statuses) == milestone_signature(
        4, monitored, statuses
    )


def test_terminal_transition_wakes_once():
    statuses = {"run-a": {"status": "completed"}, "run-b": {"status": "running"}}
    signature = terminal_signature(3, statuses)
    wake, observed, reason = should_wake(
        control=control(), runtime={}, statuses=statuses, now=1000.0
    )
    assert wake is True
    assert observed == signature
    assert reason == "new_terminal_transition"

    wake, _, reason = should_wake(
        control=control(),
        runtime={"handled_signatures": [signature]},
        statuses=statuses,
        now=1010.0,
    )
    assert wake is False
    assert reason == "terminal_transition_already_handled"


def test_generation_change_rearms_same_run_state():
    statuses = {"run-a": {"status": "failed"}, "run-b": {"status": "running"}}
    old = terminal_signature(3, statuses)
    new = terminal_signature(4, statuses)
    assert old != new


def test_daily_budget_is_rolling_and_fail_closed():
    now = 100_000.0
    assert within_wake_budget([now - 90_000, now - 10], now, 1) is False
    assert within_wake_budget([now - 90_000], now, 1) is True
    wake, _, reason = should_wake(
        control=control(max_codex_wakes_per_24h=1),
        runtime={
            "wake_history": [
                {"started_at": now - 10, "return_code": 0, "succeeded": True}
            ]
        },
        statuses={"run-a": {"status": "completed"}},
        now=now,
    )
    assert wake is False
    assert reason == "daily_codex_wake_budget_exhausted"


def test_disabled_or_empty_control_never_wakes():
    assert should_wake(
        control=control(enabled=False),
        runtime={},
        statuses={"run-a": {"status": "completed"}},
        now=0.0,
    )[2] == "disabled"
    assert should_wake(
        control=control(monitored_runs=[]),
        runtime={},
        statuses={},
        now=0.0,
    )[2] == "no_monitored_runs"


def test_native_goal_lease_suppresses_event_and_scheduled_wakes(tmp_path, monkeypatch):
    lease = tmp_path / "native-goal.lease"
    lease.write_text("native thread owns handoffs\n")
    monkeypatch.setattr(coordinator, "NATIVE_GOAL_LEASE_PATH", lease)
    now = lease.stat().st_mtime + 10
    assert should_wake(
        control=control(native_goal_lease_seconds=30),
        runtime={},
        statuses={"run-a": {"status": "completed"}},
        now=now,
    )[2] == "native_goal_lease_active"
    assert should_run_scheduled_audit(
        control=control(native_goal_lease_seconds=30), runtime={}, now=now
    )[1] == "native_goal_lease_active"

    later = lease.stat().st_mtime + 31
    assert should_wake(
        control=control(native_goal_lease_seconds=30),
        runtime={},
        statuses={"run-a": {"status": "completed"}},
        now=later,
    )[2] == "new_terminal_transition"


def test_scheduled_audit_runs_without_recent_continuation():
    due, reason = should_run_scheduled_audit(
        control=control(audit_recent_continuation_seconds=3600),
        runtime={"codex_wake_times": [1000.0]},
        now=5000.0,
    )
    assert due is True
    assert reason == "scheduled_90m_audit_due"


def test_scheduled_audit_skips_recent_or_over_budget_continuation():
    due, reason = should_run_scheduled_audit(
        control=control(audit_recent_continuation_seconds=3600),
        runtime={
            "codex_wake_times": [4500.0],
            "wake_history": [{"started_at": 4500.0, "return_code": 0, "succeeded": True}],
        },
        now=5000.0,
    )
    assert due is False
    assert reason == "recent_continuation_already_checked_pipeline"

    due, reason = should_run_scheduled_audit(
        control=control(max_codex_wakes_per_24h=1),
        runtime={
            "codex_wake_times": [4500.0],
            "wake_history": [
                {"started_at": 4500.0, "return_code": 0, "succeeded": True}
            ],
        },
        now=5000.0,
    )
    assert due is False
    assert reason == "daily_codex_wake_budget_exhausted"


def test_failed_launch_does_not_suppress_scheduled_audit():
    due, reason = should_run_scheduled_audit(
        control=control(audit_recent_continuation_seconds=3600),
        runtime={
            "codex_wake_times": [4500.0],
            "wake_history": [{"started_at": 4500.0, "return_code": 1, "succeeded": False}],
        },
        now=5000.0,
    )
    assert due is True
    assert reason == "scheduled_90m_audit_due"


def test_successful_wake_times_excludes_failed_or_legacy_false_successes():
    assert successful_wake_times(
        {
            "wake_history": [
                {"started_at": 10.0, "return_code": 1, "succeeded": False},
                {"started_at": 20.0, "return_code": 0},
                {"started_at": 30.0, "return_code": 0, "succeeded": True},
            ]
        }
    ) == [30.0]
