from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


API_PROGRESS_TASK_KINDS = {"brief_synthesis", "document_cleaning"}
_PROGRESS_RE = re.compile(
    r"processed=(?P<processed>\d+)"
    r"(?: total_completed=(?P<total_completed>\d+))?"
    r" api_cost_usd=(?P<api_cost_usd>\d+(?:\.\d+)?)"
)


def running_api_progress(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {}
    processed: int | None = None
    spent: float | None = None
    failed = 0
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("record failure "):
            failed += 1
        match = _PROGRESS_RE.search(line)
        if match:
            processed = int(match.group("processed"))
            spent = round(float(match.group("api_cost_usd")), 6)
    progress: dict[str, Any] = {}
    if processed is not None:
        progress["records_processed"] = processed
    if failed:
        progress["records_failed"] = failed
    if spent is not None:
        progress["actual_api_cost_usd"] = spent
    return progress


def enrich_running_api_state(state: Mapping[str, Any], log_path: Path) -> dict[str, Any]:
    enriched = dict(state)
    if (
        enriched.get("status") != "running"
        or enriched.get("task_kind") not in API_PROGRESS_TASK_KINDS
    ):
        return enriched
    progress = running_api_progress(log_path)
    enriched.update(progress)
    if progress.get("actual_api_cost_usd") is not None:
        enriched["cost_usd"] = progress["actual_api_cost_usd"]
    return enriched
