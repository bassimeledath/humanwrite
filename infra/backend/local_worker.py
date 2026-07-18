from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .local_backend import _artifact_dir, _count_generated_tokens, _log_path, _pid_path, _record, _state_dir
from .volume_paths import run_artifact_metadata


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_worker(payload_path: Path) -> dict[str, object]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    run_id = str(payload["run_id"])
    state = _state_dir(payload["state_dir"])
    log_path = _log_path(state, run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = _pid_path(state, run_id)
    status = "failed"
    return_code = None
    started = time.time()
    try:
        command = list((payload["config"].get("run") or {}).get("command", ["python", "-m", "experiments.runner"]))
        command += ["--config", str(payload["config_path"]), "--run-id", run_id]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(ROOT)),
            "PYTHONPATH": str(ROOT),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DFTR_RUN_ID": run_id,
            "DFTR_CHECKPOINT_DIR": str(state / "artifacts" / run_id),
        }
        readiness = payload.get("dft_a64_readiness")
        if readiness is not None:
            readiness_path = Path(str(readiness["manifest_path"]))
            if (
                not readiness_path.is_file()
                or readiness_path.is_symlink()
                or _sha256(readiness_path) != readiness["manifest_sha256"]
            ):
                raise ValueError("A64 readiness manifest wrapper verification failed")
            env["DFTR_M2_A64_READINESS_MANIFEST"] = str(readiness_path)
            env["DFTR_M2_A64_READINESS_SHA256"] = str(readiness["manifest_sha256"])
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("[dftr] local backend worker start\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=int(payload["timeout_seconds"]),
                check=False,
            )
        return_code = result.returncode
        status = "completed" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "reaped"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("[dftr] local backend timeout\n")
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[dftr] local backend worker failure: {type(exc).__name__}: {exc}\n")
    finally:
        elapsed = max(0.0, time.time() - started)
        reserved = float(payload["reserved_cost_usd"])
        actual = min(reserved, reserved * elapsed / max(1.0, float(payload["timeout_seconds"])))
        artifact_dir = _artifact_dir(state, run_id)
        _record(
            state,
            {
                "kind": "run_update",
                "run_id": run_id,
                "status": status,
                "return_code": return_code,
                "finished_at": time.time(),
                "accel_seconds": round(elapsed, 3),
                "tokens": _count_generated_tokens(state, run_id),
                "actual_cost_usd": round(actual, 6),
                "artifact_dir": str(artifact_dir.resolve()),
                **run_artifact_metadata(artifact_dir, mount_path=str(state / "artifacts")),
            },
        )
        if pid_path.exists():
            pid_path.unlink()
    return {"run_id": run_id, "status": status, "return_code": return_code}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local offline GPU smoke worker.")
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_worker(Path(args.payload).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
