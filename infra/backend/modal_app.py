"""Modal deployment for the constrained DFT-R compute gateway.

Deploy with: modal deploy -m infra.backend.modal_app
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib import request as urlrequest

import modal

from .policy import (
    MONTHLY_API_CAP_USD,
    MONTHLY_GPU_CAP_USD,
    PolicyError,
    append_event,
    authorized,
    budget_snapshot,
    has_api_capacity,
    has_capacity,
    read_events,
    revision_is_unresolved,
    run_snapshot,
    validate_launch,
)


APP_NAME = "humanwrite-gpu-gateway"
STATE_PATH = "/state/events.jsonl"
CHECKPOINT_PATH = "/checkpoints"
REPO_URL = "https://github.com/bassimeledath/humanwrite.git"

state_volume = modal.Volume.from_name("humanwrite-gateway-state", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints", create_if_missing=True)
gateway_secret = modal.Secret.from_name("humanwrite-gateway-auth")
provider_secret = modal.Secret.from_name("the-other-ones")

source_root = Path(__file__).resolve().parent
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("fastapi[standard]", "pyyaml", "requests")
    .add_local_dir(source_root, remote_path="/root/infra_backend", copy=True)
)
worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pyyaml",
        "huggingface-hub>=0.33,<1",
        "torch>=2.7,<3",
        "transformers>=4.53,<5",
        "datasets>=3.6,<5",
        "accelerate>=1.8,<2",
        "peft>=0.16,<1",
        "requests>=2.32,<3",
    )
    .add_local_dir(source_root, remote_path="/root/infra_backend", copy=True)
)

app = modal.App(APP_NAME)


def _events() -> list[dict]:
    try:
        state_volume.reload()
    except Exception:
        pass
    return read_events(STATE_PATH)


def _record(event: dict) -> None:
    append_event(STATE_PATH, event)
    state_volume.commit()


def _notify(event: str, details: dict) -> None:
    webhook = os.environ.get("DFTR_ALERT_WEBHOOK_URL")
    if not webhook:
        return
    body = json.dumps({"event": event, "project": "humanwrite", **details}).encode()
    req = urlrequest.Request(webhook, data=body, headers={"Content-Type": "application/json"})
    try:
        urlrequest.urlopen(req, timeout=10).read()
    except Exception as exc:
        print(f"alert delivery failed: {type(exc).__name__}")


def _artifact_dir(run_id: str) -> Path:
    return Path(CHECKPOINT_PATH) / "runs" / run_id


def _resolved_model_manifest_path(run_id: str) -> Path:
    return _artifact_dir(run_id) / "resolved_model.json"


def _count_artifact_tokens(artifact_dir: Path) -> int:
    manifest_path = artifact_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        accounting = manifest.get("token_accounting") or {}
        if accounting.get("total_tokens") is not None:
            return int(accounting["total_tokens"])
    total = 0
    for sample_path in sorted(artifact_dir.rglob("*.jsonl")):
        for line in sample_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(
                row.get("output")
                or row.get("generated_completion")
                or row.get("completion")
                or ""
            )
            total += len(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text, re.UNICODE))
    return total


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=8 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def training_worker(run_id: str, payload: dict) -> dict:
    """Run only the allowlisted experiment module at an immutable git SHA."""
    started = time.time()
    config = payload["config"]
    worktree = Path("/tmp") / run_id
    log_path = Path("/state/logs") / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = "failed"
    return_code = None
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", REPO_URL, str(worktree)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "-C", str(worktree), "checkout", "--detach", payload["git_sha"]],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        # The privileged wrapper may use HF_TOKEN to populate a shared cache,
        # then deletes it before agent-authored experiment code starts.
        hf_token = os.environ.pop("HF_TOKEN", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        base_model = str((config.get("model") or {}).get("base", ""))
        model_revision = (config.get("model") or {}).get("revision")
        workflow_step = str((config.get("workflow") or {}).get("step", "")).casefold()
        artifact_dir = _artifact_dir(run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        resolved_manifest_path = _resolved_model_manifest_path(run_id)
        if hf_token and base_model:
            from huggingface_hub import snapshot_download
            if workflow_step == "resolve_revision":
                from huggingface_hub import HfApi

                requested_revision = str((config.get("model") or {}).get("requested_revision") or "main")
                model_info = HfApi(token=hf_token).model_info(base_model, revision=requested_revision)
                resolved_revision = str(model_info.sha or requested_revision)
                snapshot_path = snapshot_download(
                    repo_id=base_model,
                    revision=resolved_revision,
                    token=hf_token,
                    cache_dir="/checkpoints/hf-cache",
                    allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
                )
                resolved_manifest_path.write_text(
                    json.dumps(
                        {
                            "base_model": base_model,
                            "requested_revision": requested_revision,
                            "resolved_revision": resolved_revision,
                            "resolved_at": time.time(),
                            "snapshot_path": snapshot_path,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            elif not revision_is_unresolved(model_revision):
                snapshot_download(
                    repo_id=base_model,
                    revision=str(model_revision),
                    token=hf_token,
                    cache_dir="/checkpoints/hf-cache",
                    allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
                )
            checkpoint_volume.commit()

        config_path = worktree / ".dftr-run-config.json"
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        command = list((config.get("run") or {}).get(
            "command", ["python", "-m", "experiments.runner"]
        )) + ["--config", str(config_path), "--run-id", run_id]
        clean_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": "/tmp",
            "PYTHONPATH": str(worktree),
            "HF_HOME": "/checkpoints/hf-cache",
            "HF_HUB_CACHE": "/checkpoints/hf-cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DFTR_RUN_ID": run_id,
            "DFTR_CHECKPOINT_DIR": f"/checkpoints/runs/{run_id}",
        }
        if resolved_manifest_path.is_file():
            clean_env["DFTR_RESOLVED_MODEL_MANIFEST"] = str(resolved_manifest_path)
        with log_path.open("w", encoding="utf-8") as logs:
            result = subprocess.run(
                command,
                cwd=worktree,
                env=clean_env,
                stdout=logs,
                stderr=subprocess.STDOUT,
                timeout=int(payload["timeout_seconds"]),
                check=False,
            )
        return_code = result.returncode
        status = "completed" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "reaped"
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write("\n[dftr] hard timeout reached\n")
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(f"\n[dftr] wrapper failure: {type(exc).__name__}: {exc}\n")
    finally:
        elapsed = max(0.0, time.time() - started)
        reserved = float(payload["reserved_cost_usd"])
        actual = min(reserved, reserved * elapsed / float(payload["timeout_seconds"]))
        artifact_dir = _artifact_dir(run_id)
        _record({
            "kind": "run_update",
            "run_id": run_id,
            "status": status,
            "return_code": return_code,
            "finished_at": time.time(),
            "accel_seconds": round(elapsed, 3),
            "tokens": _count_artifact_tokens(artifact_dir),
            "actual_cost_usd": round(actual, 6),
            "artifact_dir": str(artifact_dir.resolve()),
        })
        checkpoint_volume.commit()
    return {"run_id": run_id, "status": status, "return_code": return_code}


def _volume_path(uri: str) -> Path:
    prefix = "modal-volume://humanwrite-checkpoints/"
    if not uri.startswith(prefix):
        raise ValueError("URI is outside the checkpoint volume")
    relative = uri[len(prefix):]
    path = (Path(CHECKPOINT_PATH) / relative).resolve()
    if not path.is_relative_to(Path(CHECKPOINT_PATH)):
        raise ValueError("unsafe volume URI")
    return path


def _brief_prompt(source_text: str) -> str:
    return (
        "Convert the supplied human web document into the disclosed DFT training brief. "
        "Return one JSON object only with keys user_prompt, use_case, style_kind, style, "
        "detail_mode, target_length, em_dashes_allowed, and outline. outline is a list of "
        "objects with section, supported_facts, and quotations. Every fact and quote must "
        "be traceable to the document; do not invent facts. target_length is an integer token "
        "estimate. detail_mode must be strict or creative.\n\nDOCUMENT:\n" + source_text
    )


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=8 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def brief_synthesis_worker(run_id: str, payload: dict) -> dict:
    """Privileged fixed-code brief synthesis; never executes repository code."""
    import requests

    config = payload["config"]
    data_config = config.get("data") or {}
    input_path = _volume_path(str(data_config["input_uri"]))
    output_path = _volume_path(str(data_config["output_uri"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_records = int(data_config.get("max_records", 50_000))
    model = os.environ["DFTR_OPENROUTER_MODEL"]
    api_key = os.environ["OPENROUTER_API_KEY"]
    cost_cap = float(payload["api_reserved_cost_usd"])
    spent = 0.0
    reported_spent = 0.0
    processed = 0
    status = "failed"
    log_path = Path("/state/logs") / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint_volume.reload()
        if not input_path.is_file():
            raise FileNotFoundError(f"input artifact not found: {data_config['input_uri']}")
        completed_ids = set()
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    completed_ids.add(str(row.get("fingerprint") or row.get("fineweb_id")))
        with input_path.open(encoding="utf-8") as source, output_path.open("a", encoding="utf-8") as sink:
            for line in source:
                if processed >= max_records or spent >= cost_cap:
                    break
                record = json.loads(line)
                record_id = str(record.get("fingerprint") or record.get("fineweb_id"))
                if not record_id or record_id in completed_ids:
                    continue
                text = str(record.get("completion") or record.get("text") or "")
                if not text.strip():
                    continue
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                        "X-OpenRouter-Title": "Humanwrite DFT-R brief synthesis",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": _brief_prompt(text[:120_000])}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.2,
                        "max_tokens": 1800,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                body = response.json()
                usage_cost = (body.get("usage") or {}).get("cost")
                if usage_cost is None:
                    raise RuntimeError("OpenRouter response omitted usage.cost; refusing unmetered call")
                spent += float(usage_cost)
                content = body["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                brief = json.loads(content)
                required = {
                    "user_prompt", "use_case", "style_kind", "style", "detail_mode",
                    "target_length", "em_dashes_allowed", "outline",
                }
                if not required.issubset(brief):
                    raise ValueError(f"brief response missing fields: {sorted(required - set(brief))}")
                # The disclosed 25% empty-outline arm is deterministic by document identity.
                empty_outline = int.from_bytes(
                    __import__("hashlib").sha256(record_id.encode()).digest()[:4], "big"
                ) % 4 == 0
                if empty_outline:
                    brief["outline"] = []
                emitted = dict(record)
                emitted.update(brief)
                emitted["generation_mode"] = "generate"
                emitted["completion"] = text
                sink.write(json.dumps(emitted, ensure_ascii=False, sort_keys=True) + "\n")
                sink.flush()
                processed += 1
                completed_ids.add(record_id)
                if processed % 50 == 0:
                    checkpoint_volume.commit()
                    _record({
                        "kind": "api_cost", "run_id": run_id,
                        "cost_usd": round(spent - reported_spent, 6),
                    })
                    reported_spent = spent
                    with log_path.open("a", encoding="utf-8") as logs:
                        logs.write(f"processed={processed} api_cost_usd={spent:.6f}\n")
        checkpoint_volume.commit()
        status = "completed" if processed or completed_ids else "failed"
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(f"brief synthesis failure: {type(exc).__name__}: {exc}\n")
    finally:
        if spent > reported_spent:
            _record({
                "kind": "api_cost", "run_id": run_id,
                "cost_usd": round(spent - reported_spent, 6),
            })
        _record({
            "kind": "run_update",
            "run_id": run_id,
            "status": status,
            "finished_at": time.time(),
            "actual_api_cost_usd": round(min(cost_cap, spent), 6),
            "records_processed": processed,
        })
    return {"run_id": run_id, "status": status, "records_processed": processed,
            "actual_api_cost_usd": round(spent, 6)}


@app.function(
    image=base_image,
    secrets=[gateway_secret, provider_secret],
    volumes={"/state": state_volume},
    max_containers=1,
    timeout=120,
)
@modal.asgi_app()
def gateway():
    from fastapi import FastAPI, Header, HTTPException, Query

    api = FastAPI(title="Humanwrite constrained GPU gateway", docs_url=None, redoc_url=None)

    def require_auth(authorization: str | None) -> None:
        if not authorized(authorization, os.environ["DFTR_GPU_GATEWAY_TOKEN"]):
            raise HTTPException(status_code=401, detail="unauthorized")

    @api.post("/submit")
    def submit(payload: dict, authorization: str | None = Header(default=None)):
        require_auth(authorization)
        try:
            policy = validate_launch(payload)
        except PolicyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        run_id = str(payload.get("run_id", ""))
        if not run_id.startswith("dftr-"):
            raise HTTPException(status_code=400, detail="invalid run_id")
        events = _events()
        if run_snapshot(events, run_id):
            raise HTTPException(status_code=409, detail="run_id already exists")
        if not has_capacity(events, policy.worst_case_cost_usd):
            raise HTTPException(status_code=402, detail="monthly GPU budget exhausted")
        if not has_api_capacity(events, policy.api_reserved_cost_usd):
            raise HTTPException(status_code=402, detail="monthly API budget exhausted")
        launch = {
            "kind": "run",
            "run_id": run_id,
            "comparison": policy.comparison_id,
            "status": "reserved",
            "budget_class": policy.budget_class,
            "gpu": policy.gpu,
            "task_kind": policy.task_kind,
            "timeout_seconds": policy.timeout_seconds,
            "reserved_cost_usd": policy.worst_case_cost_usd,
            "api_reserved_cost_usd": policy.api_reserved_cost_usd,
            "config_hash": payload["config_hash"],
            "git_sha": payload["git_sha"],
            "billing_month": datetime.now(timezone.utc).strftime("%Y-%m"),
            "started_at": time.time(),
        }
        _record(launch)
        worker_payload = dict(payload)
        worker_payload.update({
            "timeout_seconds": policy.timeout_seconds,
            "reserved_cost_usd": policy.worst_case_cost_usd,
            "api_reserved_cost_usd": policy.api_reserved_cost_usd,
        })
        try:
            if policy.task_kind == "brief_synthesis":
                call = brief_synthesis_worker.with_options(
                    timeout=policy.timeout_seconds + 120,
                ).spawn(run_id, worker_payload)
            else:
                call = training_worker.with_options(
                    gpu=policy.gpu,
                    timeout=policy.timeout_seconds + 120,
                ).spawn(run_id, worker_payload)
        except Exception as exc:
            _record({"kind": "run_update", "run_id": run_id, "status": "launch_failed"})
            raise HTTPException(status_code=503, detail="Modal launch failed") from exc
        _record({
            "kind": "run_update",
            "run_id": run_id,
            "status": "running",
            "function_call_id": call.object_id,
        })
        if policy.budget_class == "promo":
            _notify("promo_launch", {"run_id": run_id, "comparison": policy.comparison_id})
        return {
            "run_id": run_id,
            "status": "running",
            "budget_class": policy.budget_class,
            "reserved_cost_usd": policy.worst_case_cost_usd,
            "api_reserved_cost_usd": policy.api_reserved_cost_usd,
        }

    @api.get("/status/{run_id}")
    def status(run_id: str, authorization: str | None = Header(default=None)):
        require_auth(authorization)
        state = run_snapshot(_events(), run_id)
        if not state:
            raise HTTPException(status_code=404, detail="run not found")
        return {key: value for key, value in state.items() if key != "function_call_id"}

    @api.get("/logs/{run_id}")
    def logs(run_id: str, tail: int = Query(default=200, ge=1, le=5000),
             authorization: str | None = Header(default=None)):
        require_auth(authorization)
        if not run_snapshot(_events(), run_id):
            raise HTTPException(status_code=404, detail="run not found")
        path = Path("/state/logs") / f"{run_id}.log"
        if not path.exists():
            return {"run_id": run_id, "logs": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"run_id": run_id, "logs": "\n".join(lines[-tail:])}

    @api.post("/cancel/{run_id}")
    def cancel(run_id: str, authorization: str | None = Header(default=None)):
        require_auth(authorization)
        state = run_snapshot(_events(), run_id)
        if not state:
            raise HTTPException(status_code=404, detail="run not found")
        if state.get("status") in {"completed", "failed", "cancelled", "reaped"}:
            return {"run_id": run_id, "status": state["status"]}
        call_id = state.get("function_call_id")
        if call_id:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        elapsed = max(0.0, time.time() - float(state.get("started_at") or time.time()))
        timeout = max(1.0, float(state.get("timeout_seconds") or 1.0))
        reserved = float(state.get("reserved_cost_usd") or 0.0)
        actual = min(reserved, reserved * elapsed / timeout)
        artifact_dir = _artifact_dir(run_id)
        _record({"kind": "run_update", "run_id": run_id, "status": "cancelled",
                 "finished_at": time.time(), "accel_seconds": round(elapsed, 3),
                 "tokens": _count_artifact_tokens(artifact_dir),
                 "actual_cost_usd": round(actual, 6),
                 "artifact_dir": str(artifact_dir.resolve())})
        return {"run_id": run_id, "status": "cancelled"}

    @api.get("/budget")
    def budget(authorization: str | None = Header(default=None)):
        require_auth(authorization)
        return budget_snapshot(_events())

    @api.post("/judge")
    def judge(payload: dict, authorization: str | None = Header(default=None)):
        """Fixed Tier 1 pairwise judge; returns only A, B, or TIE."""
        require_auth(authorization)
        current_budget = budget_snapshot(_events())
        if current_budget["api_remaining_usd"] <= 0.02:
            raise HTTPException(status_code=402, detail="monthly API budget exhausted")
        prompt = str(payload.get("prompt", ""))
        candidate_a = str(payload.get("candidate_a", ""))
        candidate_b = str(payload.get("candidate_b", ""))
        if not prompt or not candidate_a or not candidate_b:
            raise HTTPException(status_code=400, detail="prompt and both candidates are required")
        if max(map(len, (prompt, candidate_a, candidate_b))) > 200_000:
            raise HTTPException(status_code=413, detail="judge input too large")
        import requests

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                    "X-OpenRouter-Title": "Humanwrite DFT-R Tier 1 judge",
                },
                json={
                    "model": os.environ["DFTR_JUDGE_MODEL"],
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Compare A and B for correctness, relevance, clarity, document quality, "
                                "and faithfulness to the prompt. Return exactly A, B, or TIE."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"PROMPT:\n{prompt}\n\nA:\n{candidate_a}\n\nB:\n{candidate_b}",
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": 8,
                },
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            usage_cost = (body.get("usage") or {}).get("cost")
            if usage_cost is None:
                raise RuntimeError("judge response omitted usage cost")
            choice = str(body["choices"][0]["message"]["content"]).strip().upper()
            match = re.search(r"\b(TIE|A|B)\b", choice)
            if not match:
                raise RuntimeError("judge returned an invalid choice")
            _record({"kind": "api_cost", "cost_usd": float(usage_cost), "purpose": "tier1_judge"})
            return {"winner": match.group(1)}
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail="judge provider unavailable") from exc
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=502, detail="invalid judge provider response") from exc

    return api
