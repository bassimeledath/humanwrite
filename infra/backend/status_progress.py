from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping


API_PROGRESS_TASK_KINDS = {
    "brief_synthesis",
    "document_cleaning",
    "rewrite_synthesis",
    "rewrite_judging",
}
M3_BASELINE_DRAFT_PROGRESS_PATH = Path(
    "/checkpoints/data/m3-rewriting-14b-v1/baseline-draft-candidates-4096-v1.progress.json"
)
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
            processed = int(match.group("total_completed") or match.group("processed"))
            spent = round(float(match.group("api_cost_usd")), 6)
    progress: dict[str, Any] = {}
    if processed is not None:
        progress["records_processed"] = processed
    if failed:
        progress["records_failed"] = failed
    if spent is not None:
        progress["actual_api_cost_usd"] = spent
    return progress


def running_sidecar_progress(progress_path: Path) -> dict[str, Any]:
    if not progress_path.is_file():
        return {}
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    processed = payload.get("records_completed", payload.get("records_processed"))
    if isinstance(processed, int) and processed >= 0:
        return {"records_processed": processed}
    return {}


def _default_sidecar_progress_path(state: Mapping[str, Any]) -> Path | None:
    if str(state.get("workflow_step") or "") == "generate_m3_baseline_drafts":
        return M3_BASELINE_DRAFT_PROGRESS_PATH
    return None


def enrich_running_api_state(state: Mapping[str, Any], log_path: Path) -> dict[str, Any]:
    enriched = dict(state)
    if enriched.get("status") != "running":
        return enriched
    if enriched.get("task_kind") in API_PROGRESS_TASK_KINDS:
        progress = running_api_progress(log_path)
    else:
        progress_path_value = str(enriched.get("progress_path") or "").strip()
        progress_path = (
            Path(progress_path_value)
            if progress_path_value
            else _default_sidecar_progress_path(enriched)
        )
        progress = running_sidecar_progress(progress_path) if progress_path else {}
    enriched.update(progress)
    if progress.get("actual_api_cost_usd") is not None:
        enriched["cost_usd"] = progress["actual_api_cost_usd"]
    return enriched
