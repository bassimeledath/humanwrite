from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "ledger.py"


def run_ledger(tmp_path: Path, *args: str) -> str:
    env = dict(os.environ)
    env["DFTR_LEDGER_PATH"] = str(tmp_path / "ledger.jsonl")
    return subprocess.check_output([sys.executable, str(LEDGER), *args], cwd=ROOT, env=env, text=True)


def test_ledger_appends_and_keeps_backup(tmp_path):
    run_ledger(tmp_path, "add", "--comparison", "M0-offline-smoke", "--hypothesis", "smoke path works")
    run_ledger(
        tmp_path,
        "add",
        "--comparison",
        "M0-offline-smoke",
        "--run-id",
        "dftr-test",
        "--config-hash",
        "abc",
        "--git-sha",
        "f" * 40,
        "--budget-class",
        "smoke",
        "--data-split-hash",
        "123",
    )
    run_ledger(tmp_path, "update", "--run-id", "dftr-test", "--status", "completed", "--tokens", "10")
    ledger_path = tmp_path / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [row["kind"] for row in rows] == ["prereg", "run", "run_update"]
    assert (tmp_path / "ledger.jsonl.bak").read_text(encoding="utf-8") == ledger_path.read_text(encoding="utf-8")


def test_query_open_respects_env_override(tmp_path):
    run_ledger(tmp_path, "add", "--comparison", "cmp-a", "--hypothesis", "a")
    run_ledger(tmp_path, "add", "--comparison", "cmp-b", "--hypothesis", "b")
    output = run_ledger(tmp_path, "query", "--comparison", "cmp-b", "--open")
    rows = json.loads(output)
    assert len(rows) == 1
    assert rows[0]["comparison"] == "cmp-b"

