from __future__ import annotations

from pathlib import Path

from backend import status_progress
from backend.status_progress import (
    enrich_running_api_state,
    running_api_progress,
    running_sidecar_progress,
)


def test_running_api_progress_reads_latest_cumulative_snapshot(tmp_path: Path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "record failure id=a error=ValueError detail=cleaned document violates word-count bounds",
                "processed=50 total_completed=50 api_cost_usd=0.123456 concurrency=128",
                "record failure id=b error=RuntimeError detail=provider finish_reason=length",
                "processed=75 total_completed=75 api_cost_usd=0.456789 concurrency=128",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert running_api_progress(log_path) == {
        "records_processed": 75,
        "records_failed": 2,
        "actual_api_cost_usd": 0.456789,
    }


def test_running_api_progress_handles_missing_or_failure_only_logs(tmp_path: Path):
    missing = tmp_path / "missing.log"
    assert running_api_progress(missing) == {}

    failure_only = tmp_path / "failure-only.log"
    failure_only.write_text(
        "record failure id=a error=RuntimeError detail=provider finish_reason=length\n",
        encoding="utf-8",
    )
    assert running_api_progress(failure_only) == {"records_failed": 1}


def test_running_sidecar_progress_reads_records_completed(tmp_path: Path):
    progress_path = tmp_path / "baseline.progress.json"
    progress_path.write_text(
        '{"run_id":"dftr-test","records_completed":64}\n',
        encoding="utf-8",
    )
    assert running_sidecar_progress(progress_path) == {"records_processed": 64}


def test_enrich_running_api_state_only_mutates_running_api_tasks(tmp_path: Path):
    log_path = tmp_path / "brief.log"
    log_path.write_text("processed=24 api_cost_usd=0.222222 concurrency=8\n", encoding="utf-8")

    state = {
        "run_id": "dftr-test",
        "status": "running",
        "task_kind": "brief_synthesis",
        "cost_usd": 0.02,
    }
    assert enrich_running_api_state(state, log_path) == {
        "run_id": "dftr-test",
        "status": "running",
        "task_kind": "brief_synthesis",
        "cost_usd": 0.222222,
        "records_processed": 24,
        "actual_api_cost_usd": 0.222222,
    }

    terminal = {"run_id": "dftr-test", "status": "completed", "task_kind": "brief_synthesis"}
    assert enrich_running_api_state(terminal, log_path) == terminal

    experiment = {"run_id": "dftr-test", "status": "running", "task_kind": "experiment"}
    assert enrich_running_api_state(experiment, log_path) == experiment


def test_enrich_running_rewrite_synthesis_state(tmp_path: Path):
    log_path = tmp_path / "rewrite.log"
    log_path.write_text(
        "processed=12 total_completed=108 api_cost_usd=1.250000 concurrency=16\n",
        encoding="utf-8",
    )
    state = {"status": "running", "task_kind": "rewrite_synthesis"}
    enriched = enrich_running_api_state(state, log_path)
    assert enriched["records_processed"] == 108
    assert enriched["actual_api_cost_usd"] == 1.25


def test_enrich_running_baseline_draft_state_reads_progress_sidecar(
    tmp_path: Path,
    monkeypatch,
):
    progress_path = tmp_path / "baseline.progress.json"
    progress_path.write_text(
        '{"run_id":"dftr-test","records_completed":96}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(status_progress, "M3_BASELINE_DRAFT_PROGRESS_PATH", progress_path)
    state = {
        "status": "running",
        "task_kind": "experiment",
        "workflow_step": "generate_m3_baseline_drafts",
    }
    enriched = enrich_running_api_state(state, tmp_path / "missing.log")
    assert enriched["records_processed"] == 96
