from __future__ import annotations

from pathlib import Path

from infra.backend.local_backend import budget_local, cancel_local, logs_local, status_local, submit_local
from infra.backend.policy import canonical_hash


class DummyProcess:
    pid = 4242


def payload():
    config = {
        "run": {
            "comparison_id": "M0-offline-smoke",
            "budget_class": "smoke",
            "command": ["python", "-m", "experiments.runner"],
        },
        "model": {"base": "Qwen/Qwen3-1.7B"},
        "compute": {"gpu": "L4", "gpus": 1, "timeout_min": 10},
        "data": {"train_split_hash": "c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6"},
    }
    return {
        "run_id": "dftr-local-test",
        "config": config,
        "config_hash": canonical_hash(config),
        "git_sha": "a" * 40,
        "budget_class": "smoke",
        "preregistration": {"kind": "prereg", "comparison": "M0-offline-smoke", "status": "open"},
        "human_scaleup_approved": False,
    }


def test_local_backend_submit_status_logs_and_cancel(monkeypatch, tmp_path):
    monkeypatch.setattr("infra.backend.local_backend.subprocess.Popen", lambda *args, **kwargs: DummyProcess())
    monkeypatch.setattr("infra.backend.local_backend.os.kill", lambda pid, sig: None)
    result = submit_local(payload(), state_dir=tmp_path)
    assert result["backend"] == "local"
    status = status_local("dftr-local-test", state_dir=tmp_path)
    assert status["status"] == "running"
    log_path = Path(status["log_path"])
    log_path.write_text("line1\nline2\n", encoding="utf-8")
    assert "line2" in logs_local("dftr-local-test", state_dir=tmp_path)["logs"]
    cancelled = cancel_local("dftr-local-test", state_dir=tmp_path)
    assert cancelled["status"] == "cancelled"
    assert budget_local(state_dir=tmp_path)["gpu_remaining_usd"] < 40.0
