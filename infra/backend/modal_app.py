"""Modal deployment for the constrained DFT-R compute gateway.

Deploy with: modal deploy -m infra.backend.modal_app
"""
from __future__ import annotations

from datetime import datetime, timezone
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib import request as urlrequest

import modal

from .policy import (
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
from .source_materializer import materialize_rows
from .status_progress import enrich_running_api_state
from .brief_contract import (
    brief_response_format,
    empty_brief_quotations,
    exact_empty_outline_ids,
    prompt_repair_response_format,
    record_id,
    validate_brief,
    validate_repaired_user_prompt,
)
from .cleaning_contract import (
    apply_line_selection,
    cleaning_response_format,
    numbered_cleaning_prompt,
)
from .openrouter_contract import (
    chat_request,
    schema_transport_fallback_allowed,
    structured_chat_request,
)
from .volume_paths import (
    checkpoint_volume_path,
    missing_run_artifact_metadata,
    run_artifact_metadata,
    run_worker_log_path,
)


APP_NAME = "humanwrite-gpu-gateway"
STATE_PATH = "/state/events.jsonl"
CHECKPOINT_PATH = "/checkpoints"
REPO_URL = "https://github.com/bassimeledath/humanwrite.git"
BRIEF_FALLBACK_MODEL = "anthropic/claude-haiku-4.5"
CLEANING_MODEL = "qwen/qwen3-32b"
LOWER_VARIANCE_BRIEF_PROTOCOL = "dftr.lower_variance_briefs.two_provider.v1"
LOWER_VARIANCE_OUTLINE_MODEL = "openai/gpt-5-mini"
LOWER_VARIANCE_BRIEF_CONCURRENCY = 8
DOCUMENT_CLEANING_CONCURRENCY = 128
M3_REWRITE_TASK_PROTOCOL = "humanwrite.m3.rewrite_tasks.v1"
M3_SCIENTIFIC_REWRITE_PROTOCOL = "humanwrite.m3.scientific_api_rewrites.v2"
M3_BASELINE_VERIFY_PROTOCOL = "humanwrite.m3.baseline_draft_verification.v1"
M3_EVAL_REWRITE_PROTOCOL = "humanwrite.m3.eval_rewrite_inputs.v1"
M3_REWRITE_JUDGE_PROTOCOL = "humanwrite.m3.rewrite_judge.v2"

state_volume = modal.Volume.from_name("humanwrite-gateway-state", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints", create_if_missing=True)
gateway_secret = modal.Secret.from_name("humanwrite-gateway-auth")
provider_secret = modal.Secret.from_name("the-other-ones")
receipt_signing_secret = modal.Secret.from_name("humanwrite-wrapper-receipt-signing")

source_root = Path(__file__).resolve().parent
data_root = source_root.parents[1] / "data"
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("fastapi[standard]", "pyyaml", "requests")
    .add_local_dir(source_root, remote_path="/root/infra_backend", copy=True)
    .add_local_dir(data_root, remote_path="/root/data", copy=True)
)
worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pyyaml",
        "huggingface-hub>=0.33,<1",
        "torch==2.13.0",
        "transformers==4.57.6",
        "datasets>=3.6,<5",
        "accelerate>=1.8,<2",
        "peft==0.19.1",
        "requests>=2.32,<3",
        "cryptography>=42,<47",
        "scikit-learn>=1.4,<2",
    )
    .add_local_dir(source_root, remote_path="/root/infra_backend", copy=True)
    .add_local_dir(data_root, remote_path="/root/data", copy=True)
)

app = modal.App(APP_NAME)


@app.function(
    image=base_image,
    volumes={"/state": state_volume},
    max_containers=1,
    timeout=60,
)
def record_event(event: dict) -> None:
    """Serialize gateway event writes across independently running workers."""
    state_volume.reload()
    append_event(STATE_PATH, event)
    state_volume.commit()


def _events() -> list[dict]:
    try:
        state_volume.reload()
    except Exception:
        pass
    return read_events(STATE_PATH)


def _record(event: dict) -> None:
    record_event.remote(event)


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


def _log_path(run_id: str) -> Path:
    return run_worker_log_path(run_id, CHECKPOINT_PATH)


def _run_logged_subprocess(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> int:
    stop_flush = threading.Event()

    def flush_loop() -> None:
        while not stop_flush.wait(15):
            try:
                checkpoint_volume.commit()
            except Exception:
                pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logs:
        flusher = threading.Thread(target=flush_loop, daemon=True)
        flusher.start()
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=logs,
                stderr=subprocess.STDOUT,
            )
            try:
                return process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
        finally:
            logs.flush()
            try:
                os.fsync(logs.fileno())
            except OSError:
                pass
            stop_flush.set()
            flusher.join(timeout=5)
            checkpoint_volume.commit()


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


def _file_sha256(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_generation_receipt(run_id: str, payload: dict, config: dict) -> Path:
    """Sign immutable generation bytes outside the experiment subprocess."""
    artifact_dir = _artifact_dir(run_id)
    manifest_path, output_path = artifact_dir / "run_manifest.json", artifact_dir / "outputs.jsonl"
    if not manifest_path.is_file() or not output_path.is_file():
        raise ValueError("completed generation lacks manifest or output bytes")
    receipt = {
        "artifact_schema": "dftr.wrapper.generation_receipt.v1",
        "status": "completed",
        "key_id": "humanwrite-modal-wrapper-receipt-v1",
        "run_id": run_id,
        "comparison_id": str((config.get("run") or {}).get("comparison_id") or ""),
        "config_sha256": str(payload["config_hash"]),
        "git_sha": str(payload["git_sha"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
    }
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    private_raw = base64.b64decode(
        os.environ["DFTR_RECEIPT_SIGNING_KEY_BASE64"], validate=True
    )
    private_key = Ed25519PrivateKey.from_private_bytes(private_raw)
    receipt["signature"] = {
        "algorithm": "ed25519",
        "signed_payload_sha256": __import__("hashlib").sha256(canonical).hexdigest(),
        "signature_base64": base64.b64encode(private_key.sign(canonical)).decode("ascii"),
    }
    path = artifact_dir / "wrapper_generation_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@app.function(
    image=worker_image,
    secrets=[receipt_signing_secret],
    volumes={CHECKPOINT_PATH: checkpoint_volume},
    timeout=60,
    retries=0,
    single_use_containers=True,
)
def finalize_generation_receipt(run_id: str, expected: dict) -> dict:
    """Read canonical volume bytes and sign them after the restricted worker exits."""
    checkpoint_volume.reload()
    manifest_path = _artifact_dir(run_id) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        set(expected) != {"config_hash", "git_sha", "comparison"}
        or manifest.get("run_id") != run_id
        or manifest.get("config_sha256") != expected.get("config_hash")
        or manifest.get("git_sha") != expected.get("git_sha")
        or manifest.get("comparison_id") != expected.get("comparison")
        or manifest.get("status") != "completed"
    ):
        raise ValueError("generation finalizer identity mismatch")
    receipt_path = _write_generation_receipt(
        run_id,
        {"config_hash": expected["config_hash"], "git_sha": expected["git_sha"]},
        {"run": {"comparison_id": expected["comparison"]}},
    )
    checkpoint_volume.commit()
    return {"path": str(receipt_path), "sha256": _file_sha256(receipt_path)}


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={CHECKPOINT_PATH: checkpoint_volume},
    timeout=8 * 60 * 60,
    retries=0,
    single_use_containers=True,
    restrict_modal_access=True,
)
def training_worker(run_id: str, payload: dict) -> dict:
    """Run only the allowlisted experiment module at an immutable git SHA."""
    started = time.time()
    config = payload["config"]
    worktree = Path("/tmp") / run_id
    log_path = _log_path(run_id)
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
                representation = config.get("representation") or {}
                embedder = str(representation.get("embedder") or "")
                embedder_revision = str(representation.get("revision") or "")
                if embedder and embedder_revision:
                    snapshot_download(
                        repo_id=embedder,
                        revision=embedder_revision,
                        token=hf_token,
                        cache_dir="/checkpoints/hf-cache",
                        allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
                    )

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
        readiness = payload.get("dft_a64_readiness")
        if readiness is not None:
            readiness_path = Path(str(readiness["manifest_path"]))
            try:
                readiness_path.resolve().relative_to(Path(CHECKPOINT_PATH).resolve())
            except (OSError, ValueError) as exc:
                raise ValueError("A64 readiness manifest is outside the checkpoint volume") from exc
            if (
                not readiness_path.is_file()
                or readiness_path.is_symlink()
                or _file_sha256(readiness_path) != readiness["manifest_sha256"]
            ):
                raise ValueError("A64 readiness manifest wrapper verification failed")
            readiness_manifest = json.loads(readiness_path.read_text(encoding="utf-8"))
            scoped_paths = [
                readiness_path,
                Path(str((readiness_manifest.get("a0_checkpoint_manifest") or {}).get("path") or "")),
                Path(str((readiness_manifest.get("a0_generation_manifest") or {}).get("path") or "")),
                Path(str((readiness_manifest.get("a0_generation_manifest") or {}).get("output_path") or "")),
                Path(str((readiness_manifest.get("measurement_protocol") or {}).get("path") or "")),
                Path(str((readiness_manifest.get("measurement_protocol") or {}).get("artifact_root") or "")),
                Path(str((readiness_manifest.get("blind_qualification") or {}).get("path") or "")),
                Path(str((readiness_manifest.get("blind_qualification") or {}).get("fixture_pack_path") or "")),
                Path(str((readiness_manifest.get("trusted_public_keys") or {}).get("path") or "")),
            ]
            try:
                for scoped_path in scoped_paths:
                    scoped_path.resolve().relative_to(Path(CHECKPOINT_PATH).resolve())
            except (OSError, ValueError) as exc:
                raise ValueError("A64 readiness evidence escapes the checkpoint volume") from exc
            clean_env["DFTR_M2_A64_READINESS_MANIFEST"] = str(readiness_path)
            clean_env["DFTR_M2_A64_READINESS_SHA256"] = str(readiness["manifest_sha256"])
        return_code = _run_logged_subprocess(
            command=command,
            cwd=worktree,
            env=clean_env,
            log_path=log_path,
            timeout_seconds=int(payload["timeout_seconds"]),
        )
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
        result_payload = {
            "run_id": run_id,
            "status": status,
            "return_code": return_code,
            "finished_at": time.time(),
            "accel_seconds": round(elapsed, 3),
            "tokens": _count_artifact_tokens(artifact_dir),
            "actual_cost_usd": round(actual, 6),
            "artifact_dir": str(artifact_dir.resolve()),
        }
        result_payload.update(run_artifact_metadata(artifact_dir, mount_path=CHECKPOINT_PATH))
        # Volume commits on container shutdown are not sufficiently prompt for
        # scientific handoffs.  Make the final checkpoint/manifest visible
        # before the successful FunctionCall result can be consumed.
        checkpoint_volume.commit()
    return result_payload


def _volume_path(uri: str) -> Path:
    return checkpoint_volume_path(uri, CHECKPOINT_PATH)


def _volume_artifact(uri: str, path: Path, *, sha_key: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {sha_key: _file_sha256(path), "metrics_ptr": uri}


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=2 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def source_materialization_worker(run_id: str, payload: dict) -> dict:
    """Privileged fixed-code FineWeb selection with no token in research code."""
    config = json.loads(json.dumps(payload["config"]))
    source = config["source"]
    data = config["data"]
    log_path = _log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = "failed"
    try:
        token = os.environ.pop("HF_TOKEN", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
        from datasets import DownloadConfig, load_dataset

        checkpoint_volume.reload()
        excluded_fingerprints: set[str] = set()
        excluded_domains: set[str] = set()
        for uri in config.get("exclusion_input_uris") or []:
            exclusion_path = _volume_path(str(uri))
            if not exclusion_path.is_file():
                raise FileNotFoundError(f"exclusion artifact not found: {uri}")
            raw = exclusion_path.read_text(encoding="utf-8")
            try:
                values = [json.loads(raw)]
            except json.JSONDecodeError:
                values = [json.loads(line) for line in raw.splitlines() if line.strip()]
            for value in values:
                excluded_fingerprints.update(str(item) for item in value.get("fingerprints") or [])
                excluded_domains.update(str(item).casefold() for item in value.get("domains") or [])
                if value.get("fingerprint"):
                    excluded_fingerprints.add(str(value["fingerprint"]))
                if value.get("domain"):
                    excluded_domains.add(str(value["domain"]).casefold())
        inline = config.get("exclusions") or {}
        config["exclusions"] = {
            "fingerprints": sorted(excluded_fingerprints | {str(v) for v in inline.get("fingerprints") or []}),
            "domains": sorted(excluded_domains | {str(v).casefold() for v in inline.get("domains") or []}),
        }
        files = source.get("files") or []
        base = f"https://huggingface.co/datasets/{source['dataset_id']}/resolve/{source['revision']}"
        urls = [f"{base}/{str(path).lstrip('/')}" for path in files]
        rows = load_dataset(
            "parquet",
            data_files={str(source["split"]): urls},
            split=str(source["split"]),
            streaming=True,
            download_config=DownloadConfig(token=token, max_retries=5),
        )
        payloads, manifest = materialize_rows(rows, config)
        train_path = _volume_path(str(data["train_output_uri"]))
        dev_path = _volume_path(str(data["dev_output_uri"]))
        manifest_path = _volume_path(str(data["manifest_output_uri"]))
        for path in (train_path, dev_path, manifest_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text(payloads["train"], encoding="utf-8")
        dev_path.write_text(payloads["dev"], encoding="utf-8")
        manifest["train"]["uri"] = str(data["train_output_uri"])
        manifest["dev"]["uri"] = str(data["dev_output_uri"])
        manifest["manifest_uri"] = str(data["manifest_output_uri"])
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_volume.commit()
        status = "completed"
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(
                f"selected={manifest['counts']['corpus_size']} "
                f"train={manifest['counts']['train_count']} "
                f"dev={manifest['counts']['dev_count']}\n"
            )
        _record({
            "kind": "run_update",
            "run_id": run_id,
            "status": status,
            "finished_at": time.time(),
            "records_processed": int(manifest["counts"]["corpus_size"]),
            "metrics_ptr": str(data["manifest_output_uri"]),
            "manifest_sha256": _file_sha256(manifest_path),
            "train_uri": str(data["train_output_uri"]),
            "train_sha256": manifest["train"]["sha256"],
            "dev_uri": str(data["dev_output_uri"]),
            "dev_sha256": manifest["dev"]["sha256"],
        })
        return {
            "run_id": run_id,
            "status": status,
            "records_processed": int(manifest["counts"]["corpus_size"]),
            "metrics_ptr": str(data["manifest_output_uri"]),
        }
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(f"source materialization failure: {type(exc).__name__}: {exc}\n")
        _record({
            "kind": "run_update",
            "run_id": run_id,
            "status": status,
            "finished_at": time.time(),
        })
        return {"run_id": run_id, "status": status, "records_processed": 0}


def _brief_prompt(
    source_text: str,
    *,
    force_empty_outline: bool,
    safety_neutral: bool = False,
) -> str:
    outline_instruction = (
        "For this record, outline must be the empty list []. "
        if force_empty_outline
        else "For this record, outline must contain at least one section. "
    )
    safety_instruction = (
        "Treat this as neutral archival analysis. Do not create political persuasion, "
        "campaign advocacy, voter targeting, calls to action, or imitated campaign copy. "
        "The user_prompt must request a neutral factual summary, and quotations must be []. "
        if safety_neutral
        else ""
    )
    return (
        "Convert the supplied human web document into the disclosed DFT training brief. "
        "Return one JSON object only with keys user_prompt, use_case, style_kind, style, "
        "detail_mode, target_length, em_dashes_allowed, and outline. outline is a list of "
        "objects with section, supported_facts, and quotations. Every fact and quote must "
        "be traceable to the document; do not invent facts. Every quotations item must be "
        "copied byte-for-byte as one contiguous substring of DOCUMENT, including punctuation; "
        "use an empty quotations list when no exact quotation is needed. target_length is an integer token "
        "estimate. detail_mode must be strict or creative. "
        + outline_instruction
        + safety_instruction
        + "Return no prose or Markdown outside the JSON object.\n\nDOCUMENT:\n"
        + source_text
    )


def _prompt_repair_prompt(source_text: str) -> str:
    return (
        "Write one document-specific user request that could naturally have caused a skilled "
        "writer to produce the supplied writing sample. The request must identify the topic and "
        "desired output, but it must not mention DFT, training, a training brief, JSON, a schema, "
        "keys, conversion, a supplied document, or these instructions. Do not ask to summarize or "
        "transform a document that the future writer cannot see. Return one JSON object containing "
        "only user_prompt.\n\nWRITING SAMPLE:\n" + source_text
    )


def _lower_variance_metadata_prompt(source: dict, *, recovery: bool = False) -> str:
    recovery_instruction = (
        "This is neutral archival metadata extraction. Keep the request analytical and "
        "non-advocacy. Copy at least one distinctive topic phrase verbatim from the sample "
        "into user_prompt, and repeat the fingerprint exactly as printed. "
        if recovery
        else ""
    )
    return (
        "Analyze the human writing sample below and return only the requested JSON object. "
        "Create a natural standalone user request that could have caused a skilled writer to "
        "produce this sample; it must name the topic and desired output without mentioning a "
        "source document, conversion, training, DFT, JSON, or these instructions. Extract the "
        "use case and writing style. target_length must estimate the sample length in TOKENS, "
        "not words. Preserve the supplied document_fingerprint exactly. "
        + recovery_instruction
        + "\n\n"
        f"document_fingerprint: {source['fingerprint']}\n\n"
        "HUMAN WRITING SAMPLE:\n" + str(source["completion"])
    )


def _lower_variance_outline_prompt(
    source: dict, *, force_empty: bool, recovery: bool = False
) -> str:
    if force_empty:
        instruction = "Return outline as exactly the empty list."
    else:
        instruction = (
            "Create a useful section outline. Every supported_facts and quotations item must "
            "be copied byte-for-byte as one contiguous substring of the human writing sample; "
            "do not paraphrase or invent facts. Use an empty quotations list when unnecessary."
        )
        if recovery:
            instruction += (
                " Use only one to three short facts. Copy each fact with exact spelling and "
                "punctuation, remove boundary whitespace, never join separate passages, and "
                "return quotations as an empty list."
            )
    return (
        "Return only the requested JSON object. Preserve the supplied document_fingerprint "
        f"exactly. {instruction}\n\n"
        f"document_fingerprint: {source['fingerprint']}\n\n"
        "HUMAN WRITING SAMPLE:\n" + str(source["completion"])
    )


def _lower_variance_safe_excerpt(source: dict) -> dict:
    """Use a short, still-verbatim passage when a provider filters a full page."""
    text = str(source["completion"])
    candidates = [
        item.strip()
        for item in re.split(r"(?:\n\s*\n|(?<=[.!?])\s+)", text)
        if 80 <= len(item.strip()) <= 1200
        and len(re.findall(r"[A-Za-z]{3,}", item)) >= 8
    ]
    excerpt = min(candidates, key=lambda item: (item.count("http"), len(item))) if candidates else text[:800].strip()
    recovered = dict(source)
    recovered["completion"] = excerpt
    return recovered


def _json_schema_response_format(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=8 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def lower_variance_brief_synthesis_worker(run_id: str, payload: dict) -> dict:
    """Privileged two-provider synthesis for the faithful lower-variance corpus."""
    import requests
    from data.lower_variance_briefs import (
        QWEN_MODEL,
        OUTLINE_MODEL,
        deterministic_empty_outline_ids,
        merge_brief,
        outline_response_schema,
        qwen_metadata_response_schema,
        validate_assembled_brief,
    )

    config = payload["config"]
    data = config["data"]
    api = config["api"]
    input_path = _volume_path(str(data["input_uri"]))
    output_path = _volume_path(str(data["output_uri"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cost_cap = float(payload["api_reserved_cost_usd"])
    spent = 0.0
    reported = 0.0
    processed = 0
    failed = 0
    status = "failed"
    log_path = _log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def provider_json(*, model: str, prompt: str, schema_name: str, schema: dict) -> tuple[dict, float]:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                "X-OpenRouter-Title": "Humanwrite faithful two-provider briefs",
            },
            json=structured_chat_request(
                model=model,
                prompt=prompt,
                response_format=_json_schema_response_format(schema_name, schema),
                max_completion_tokens=3000,
                reasoning={"effort": "minimal", "exclude": True},
            ),
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        usage_cost = (body.get("usage") or {}).get("cost")
        if usage_cost is None:
            raise RuntimeError("OpenRouter response omitted usage.cost")
        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(f"provider finish_reason={choice.get('finish_reason')}")
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise TypeError("provider message content was not a string")
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(content)
        if not isinstance(value, dict):
            raise TypeError("provider response was not a JSON object")
        return value, float(usage_cost)

    def synthesize_one(source: dict) -> tuple[dict | None, Exception | None, float]:
        fingerprint = str(source["fingerprint"])
        emitted = None
        last_error: Exception | None = None
        record_spent = 0.0
        for _attempt in range(6):
            recovery = _attempt >= 2
            prompt_source = (
                _lower_variance_safe_excerpt(source) if _attempt >= 4 else source
            )
            try:
                metadata, metadata_cost = provider_json(
                    model=QWEN_MODEL,
                    prompt=_lower_variance_metadata_prompt(
                        prompt_source, recovery=recovery
                    ),
                    schema_name="lower_variance_brief_metadata",
                    schema=qwen_metadata_response_schema(),
                )
                record_spent += metadata_cost
                outline, outline_cost = provider_json(
                    model=OUTLINE_MODEL,
                    prompt=_lower_variance_outline_prompt(
                        prompt_source,
                        force_empty=fingerprint in empty_ids,
                        recovery=recovery,
                    ),
                    schema_name="lower_variance_outline",
                    schema=outline_response_schema(
                        force_empty_outline=fingerprint in empty_ids
                    ),
                )
                record_spent += outline_cost
                emitted = merge_brief(
                    source=source,
                    qwen_metadata=metadata,
                    outline_response=outline,
                    force_empty_outline=fingerprint in empty_ids,
                    qwen_model=QWEN_MODEL,
                    outline_model=OUTLINE_MODEL,
                )
                break
            except Exception as exc:
                last_error = exc
        return emitted, last_error, record_spent

    try:
        if api["protocol"] != LOWER_VARIANCE_BRIEF_PROTOCOL:
            raise ValueError("lower-variance brief protocol mismatch")
        if api["metadata_model"] != QWEN_MODEL or api["outline_model"] != OUTLINE_MODEL:
            raise ValueError("lower-variance provider model mismatch")
        checkpoint_volume.reload()
        if not input_path.is_file() or _file_sha256(input_path) != data["input_sha256"]:
            raise ValueError("lower-variance brief input hash mismatch")
        records = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][: int(data["max_records"])]
        if len(records) != int(data["max_records"]):
            raise ValueError("lower-variance brief input cardinality mismatch")
        source_index = {str(row.get("fingerprint") or ""): row for row in records}
        if "" in source_index or len(source_index) != len(records):
            raise ValueError("lower-variance sources require unique fingerprints")
        empty_ids = deterministic_empty_outline_ids(records)
        completed: set[str] = set()
        if output_path.exists():
            for raw in output_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                fingerprint = str(row.get("fingerprint") or "")
                if fingerprint in completed or fingerprint not in source_index:
                    raise ValueError("existing lower-variance output has invalid identity")
                validate_assembled_brief(
                    row,
                    source=source_index[fingerprint],
                    force_empty_outline=fingerprint in empty_ids,
                )
                completed.add(fingerprint)
        pending = [
            source for source in records if str(source["fingerprint"]) not in completed
        ]
        with (
            output_path.open("a", encoding="utf-8") as sink,
            ThreadPoolExecutor(max_workers=LOWER_VARIANCE_BRIEF_CONCURRENCY) as pool,
        ):
            source_iterator = iter(pending)
            futures = {}
            for _ in range(LOWER_VARIANCE_BRIEF_CONCURRENCY):
                source = next(source_iterator, None)
                if source is None:
                    break
                futures[pool.submit(synthesize_one, source)] = source
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    source = futures.pop(future)
                    emitted, last_error, record_spent = future.result()
                    fingerprint = str(source["fingerprint"])
                    spent += record_spent
                    if emitted is None:
                        failed += 1
                        detail = re.sub(r"\s+", " ", str(last_error or "unknown"))[:200]
                        with log_path.open("a", encoding="utf-8") as logs:
                            logs.write(
                                f"record failure id={fingerprint} "
                                f"error={type(last_error).__name__} detail={detail}\n"
                            )
                    else:
                        sink.write(
                            json.dumps(emitted, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        sink.flush()
                        completed.add(fingerprint)
                        processed += 1
                    if spent < cost_cap:
                        next_source = next(source_iterator, None)
                        if next_source is not None:
                            futures[pool.submit(synthesize_one, next_source)] = next_source
                checkpoint_volume.commit()
                if processed and (processed % 24 == 0 or spent - reported >= 0.02):
                    _record({
                        "kind": "api_cost",
                        "run_id": run_id,
                        "cost_usd": round(spent - reported, 6),
                    })
                    reported = spent
                    with log_path.open("a", encoding="utf-8") as logs:
                        logs.write(
                            f"processed={processed} api_cost_usd={spent:.6f} "
                            f"concurrency={LOWER_VARIANCE_BRIEF_CONCURRENCY}\n"
                        )
        checkpoint_volume.commit()
        status = "completed" if len(completed) == len(records) and not failed else "failed"
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(f"lower-variance brief failure: {type(exc).__name__}: {exc}\n")
    finally:
        if spent > reported:
            _record({
                "kind": "api_cost",
                "run_id": run_id,
                "cost_usd": round(spent - reported, 6),
            })
        _record({
            "kind": "run_update",
            "run_id": run_id,
            "status": status,
            "finished_at": time.time(),
            "actual_api_cost_usd": round(spent, 6),
            "records_processed": processed,
            "records_failed": failed,
            "output_uri": str(config["data"]["output_uri"]),
            **_volume_artifact(str(config["data"]["output_uri"]), output_path, sha_key="output_sha256"),
        })
    return {
        "run_id": run_id,
        "status": status,
        "records_processed": processed,
        "records_failed": failed,
        "actual_api_cost_usd": round(spent, 6),
    }


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=8 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def rewrite_synthesis_worker(run_id: str, payload: dict) -> dict:
    """Construct cross-provider-verified M3 rewriting tasks."""
    import requests
    from transformers import AutoTokenizer
    from data.rewrite_tasks import (
        PROTOCOL,
        assemble_rewrite_task,
        deterministic_assignment,
        generator_prompt,
        generator_response_schema,
        rewrite_source_records,
        validate_rewrite_task,
        verifier_prompt,
        verifier_response_schema,
    )
    from data.m3_scientific_corpus import (
        API_REWRITE_ORIGINS,
        assemble_scientific_rewrite,
        scientific_assignment,
        scientific_generator_prompt,
        scientific_manifest,
        validate_scientific_rewrite,
    )
    from data.m3_eval_panel import (
        API_CATEGORIES,
        emit_eval_rewrite_input,
        eval_panel_manifest,
        validate_eval_rewrite_input,
    )

    config = payload["config"]
    data = config["data"]
    api = config["api"]
    tokenizer_config = config["tokenizer"]
    input_path = _volume_path(str(data["input_uri"]))
    output_path = _volume_path(str(data["output_uri"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cost_cap = float(payload["api_reserved_cost_usd"])
    spent = 0.0
    reported = 0.0
    processed = 0
    failed = 0
    status = "failed"
    log_path = _log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def provider_json(
        *, model: str, prompt: str, schema_name: str, schema: dict, max_tokens: int
    ) -> tuple[dict, float]:
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
            "X-OpenRouter-Title": "Humanwrite M3 rewrite task construction",
        }

        def parse_json_object(content: str, *, stage: str) -> dict[str, object]:
            candidate = content.strip()
            if candidate.startswith("```"):
                candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError as exc:
                start = candidate.find("{")
                end = candidate.rfind("}")
                if start < 0 or end <= start:
                    raise RuntimeError(f"{stage} decode failed: {exc}") from exc
                try:
                    value = json.loads(candidate[start : end + 1])
                except json.JSONDecodeError as inner:
                    raise RuntimeError(f"{stage} decode failed: {inner}") from inner
            if not isinstance(value, dict):
                raise TypeError(f"{stage} provider response was not a JSON object")
            return value

        def raw_json_prompt() -> str:
            return (
                f"{prompt}\n\n"
                "Return only one JSON object with no markdown or commentary. "
                "It must satisfy this exact JSON schema.\n"
                f"schema_name: {schema_name}\n"
                f"schema_json: {json.dumps(schema, sort_keys=True, separators=(',', ':'))}"
            )

        def send(payload: dict, *, stage: str) -> tuple[dict, float]:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=240,
            )
            if not response.ok:
                detail = re.sub(r"\s+", " ", response.text)[:600]
                raise RuntimeError(f"{stage} provider HTTP {response.status_code}: {detail}")
            body = response.json()
            usage_cost = (body.get("usage") or {}).get("cost")
            if usage_cost is None:
                raise RuntimeError(f"{stage} OpenRouter response omitted usage.cost")
            return body, float(usage_cost)

        def decode(body: dict, *, stage: str) -> tuple[dict, float]:
            usage_cost = (body.get("usage") or {}).get("cost")
            if usage_cost is None:
                raise RuntimeError(f"{stage} OpenRouter response omitted usage.cost")
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise RuntimeError(f"{stage} finish_reason={choice.get('finish_reason')}")
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise TypeError(f"{stage} provider message content was not a string")
            return parse_json_object(content, stage=stage), float(usage_cost)

        request_payload = structured_chat_request(
            model=model,
            prompt=prompt,
            response_format=_json_schema_response_format(schema_name, schema),
            max_completion_tokens=max_tokens,
        )
        try:
            return decode(send(request_payload, stage="json_schema")[0], stage="json_schema")
        except RuntimeError as exc:
            message = str(exc)
            if not schema_transport_fallback_allowed(model, message):
                raise
        try:
            body, _usage_cost = send(
                structured_chat_request(
                    model=model,
                    prompt=prompt,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                ),
                stage="json_object",
            )
            return decode(body, stage="json_object")
        except Exception as json_object_exc:
            try:
                body, _usage_cost = send(
                    chat_request(
                        model=model,
                        prompt=raw_json_prompt(),
                        max_completion_tokens=max_tokens,
                    ),
                    stage="raw_json_prompt",
                )
                return decode(body, stage="raw_json_prompt")
            except Exception as raw_exc:
                raise RuntimeError(
                    "qwen transport fallback exhausted: "
                    f"json_object={json_object_exc}; raw_json_prompt={raw_exc}"
                ) from raw_exc

    try:
        protocol = str(api["protocol"])
        if protocol not in {
            PROTOCOL,
            M3_SCIENTIFIC_REWRITE_PROTOCOL,
            M3_BASELINE_VERIFY_PROTOCOL,
            M3_EVAL_REWRITE_PROTOCOL,
        }:
            raise ValueError("M3 rewrite-task protocol mismatch")
        checkpoint_volume.reload()
        if not input_path.is_file() or _file_sha256(input_path) != data["input_sha256"]:
            raise ValueError("M3 rewrite-task input hash mismatch")
        all_records = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][: int(data["max_records"])]
        if len(all_records) != int(data["max_records"]):
            raise ValueError("M3 rewrite-task input cardinality mismatch")
        manifest_by_id: dict[str, dict] = {}
        if protocol == M3_SCIENTIFIC_REWRITE_PROTOCOL:
            manifest = scientific_manifest(all_records)
            manifest_by_id = {
                str(row["fingerprint"]): row
                for row in manifest
                if row["origin"] in API_REWRITE_ORIGINS
            }
            records = [
                row for row in all_records if str(row["fingerprint"]) in manifest_by_id
            ]
        elif protocol == M3_EVAL_REWRITE_PROTOCOL:
            manifest = eval_panel_manifest(all_records)
            manifest_by_id = {
                str(row["fingerprint"]): row
                for row in manifest
                if row["category"] in API_CATEGORIES
            }
            records = [
                row for row in all_records if str(row["fingerprint"]) in manifest_by_id
            ]
        elif protocol == M3_BASELINE_VERIFY_PROTOCOL:
            records = all_records
            manifest_by_id = {
                str(row["fingerprint"]): {
                    "fingerprint": str(row["fingerprint"]),
                    "origin": "baseline_model_draft",
                    **scientific_assignment(row, "baseline_model_draft"),
                }
                for row in records
            }
        else:
            records = rewrite_source_records(all_records)
        if len(records) != int(data["target_records"]):
            raise ValueError("M3 rewrite-task target cardinality mismatch")
        source_index = {str(row["fingerprint"]): row for row in records}
        if len(source_index) != len(records):
            raise ValueError("M3 rewrite-task sources require unique fingerprints")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config["model"],
            revision=tokenizer_config["revision"],
            token=os.environ.get("HF_TOKEN"),
            cache_dir="/checkpoints/hf-cache",
            trust_remote_code=True,
        )

        def token_counter(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        completed: set[str] = set()
        if output_path.exists():
            for raw in output_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                fingerprint = str(row.get("fingerprint") or "")
                if fingerprint in completed or fingerprint not in source_index:
                    raise ValueError("existing M3 rewrite output has invalid identity")
                if protocol in {
                    M3_SCIENTIFIC_REWRITE_PROTOCOL,
                    M3_BASELINE_VERIFY_PROTOCOL,
                }:
                    manifest_row = manifest_by_id[fingerprint]
                    validate_scientific_rewrite(
                        row,
                        source=source_index[fingerprint],
                        origin=str(manifest_row["origin"]),
                        token_counter=token_counter,
                        semantic_similarity_min=float(api["semantic_similarity_min"]),
                        expected_assignment={
                            field: str(manifest_row[field])
                            for field in (
                                "generator_model",
                                "verifier_model",
                                "template_id",
                            )
                        },
                    )
                elif protocol == M3_EVAL_REWRITE_PROTOCOL:
                    validate_eval_rewrite_input(
                        row,
                        source=source_index[fingerprint],
                        manifest_row=manifest_by_id[fingerprint],
                        token_counter=token_counter,
                        semantic_similarity_min=float(api["semantic_similarity_min"]),
                    )
                else:
                    validate_rewrite_task(
                        row,
                        source=source_index[fingerprint],
                        token_counter=token_counter,
                        semantic_similarity_min=float(api["semantic_similarity_min"]),
                    )
                completed.add(fingerprint)

        def synthesize_one(source: dict) -> tuple[dict | None, Exception | None, float]:
            if protocol in {
                M3_SCIENTIFIC_REWRITE_PROTOCOL,
                M3_BASELINE_VERIFY_PROTOCOL,
                M3_EVAL_REWRITE_PROTOCOL,
            }:
                manifest_row = manifest_by_id[str(source["fingerprint"])]
                assignment = {
                    field: str(manifest_row[field])
                    for field in ("generator_model", "verifier_model", "template_id")
                }
                origin = str(
                    manifest_row["scientific_origin"]
                    if protocol == M3_EVAL_REWRITE_PROTOCOL
                    else manifest_row["origin"]
                )
            else:
                assignment = deterministic_assignment(str(source["fingerprint"]))
                origin = ""
            record_spent = 0.0
            last_error: Exception | None = None
            for attempt in range(1, int(api["max_attempts"]) + 1):
                try:
                    if protocol == M3_BASELINE_VERIFY_PROTOCOL:
                        candidates = source.get("candidates")
                        if not isinstance(candidates, list) or attempt > len(candidates):
                            raise ValueError("baseline candidate attempt is unavailable")
                        candidate = candidates[attempt - 1]
                        generated = {
                            "document_fingerprint": str(source["fingerprint"]),
                            "source_text": str(candidate["input_text"]),
                            "rewrite_instruction": str(source["rewrite_instruction"]),
                        }
                        generation_cost = 0.0
                    else:
                        generated, generation_cost = provider_json(
                            model=assignment["generator_model"],
                            prompt=(
                                scientific_generator_prompt(
                                    source,
                                    assignment,
                                    origin,
                                    attempt=attempt,
                                    previous_error=str(last_error or ""),
                                )
                                if protocol in {
                                    M3_SCIENTIFIC_REWRITE_PROTOCOL,
                                    M3_EVAL_REWRITE_PROTOCOL,
                                }
                                else generator_prompt(
                                    source,
                                    assignment,
                                    attempt=attempt,
                                    previous_error=str(last_error or ""),
                                )
                            ),
                            schema_name="m3_rewrite_source",
                            schema=generator_response_schema(),
                            max_tokens=3500,
                        )
                    record_spent += generation_cost
                    generated["generation_attempt"] = attempt
                    verified, verification_cost = provider_json(
                        model=assignment["verifier_model"],
                        prompt=verifier_prompt(source, generated),
                        schema_name="m3_rewrite_verification",
                        schema=verifier_response_schema(),
                        max_tokens=4000,
                    )
                    record_spent += verification_cost
                    if protocol in {
                        M3_SCIENTIFIC_REWRITE_PROTOCOL,
                        M3_BASELINE_VERIFY_PROTOCOL,
                        M3_EVAL_REWRITE_PROTOCOL,
                    }:
                        emitted = assemble_scientific_rewrite(
                            source=source,
                            origin=origin,
                            generated=generated,
                            verified=verified,
                            assignment=assignment,
                            token_counter=token_counter,
                            semantic_similarity_min=float(api["semantic_similarity_min"]),
                        )
                        if protocol == M3_EVAL_REWRITE_PROTOCOL:
                            emitted = emit_eval_rewrite_input(
                                emitted, str(manifest_row["category"])
                            )
                    else:
                        emitted = assemble_rewrite_task(
                            source=source,
                            generated=generated,
                            verified=verified,
                            assignment=assignment,
                            token_counter=token_counter,
                            semantic_similarity_min=float(api["semantic_similarity_min"]),
                        )
                    return emitted, None, record_spent
                except Exception as exc:
                    last_error = exc
            return None, last_error, record_spent

        pending = [record for record in records if str(record["fingerprint"]) not in completed]
        with (
            output_path.open("a", encoding="utf-8") as sink,
            ThreadPoolExecutor(max_workers=int(api["concurrency"])) as pool,
        ):
            source_iterator = iter(pending)
            futures = {}
            for _ in range(int(api["concurrency"])):
                source = next(source_iterator, None)
                if source is None:
                    break
                futures[pool.submit(synthesize_one, source)] = source
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    source = futures.pop(future)
                    emitted, last_error, record_spent = future.result()
                    fingerprint = str(source["fingerprint"])
                    spent += record_spent
                    if emitted is None:
                        failed += 1
                        detail = re.sub(r"\s+", " ", str(last_error or "unknown"))[:240]
                        with log_path.open("a", encoding="utf-8") as logs:
                            logs.write(
                                f"record failure id={fingerprint} "
                                f"error={type(last_error).__name__} detail={detail}\n"
                            )
                    else:
                        sink.write(json.dumps(emitted, ensure_ascii=False, sort_keys=True) + "\n")
                        sink.flush()
                        completed.add(fingerprint)
                        processed += 1
                    if spent < cost_cap:
                        next_source = next(source_iterator, None)
                        if next_source is not None:
                            futures[pool.submit(synthesize_one, next_source)] = next_source
                checkpoint_volume.commit()
                if processed and (processed % 16 == 0 or spent - reported >= 0.02):
                    _record({
                        "kind": "api_cost",
                        "run_id": run_id,
                        "cost_usd": round(spent - reported, 6),
                    })
                    reported = spent
                    with log_path.open("a", encoding="utf-8") as logs:
                        logs.write(
                            f"processed={processed} total_completed={len(completed)} "
                            f"api_cost_usd={spent:.6f} concurrency={api['concurrency']}\n"
                        )
        checkpoint_volume.commit()
        status = "completed" if len(completed) == len(records) and not failed else "failed"
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(f"M3 rewrite synthesis failure: {type(exc).__name__}: {exc}\n")
    finally:
        if spent > reported:
            _record({
                "kind": "api_cost",
                "run_id": run_id,
                "cost_usd": round(spent - reported, 6),
            })
        _record({
            "kind": "run_update",
            "run_id": run_id,
            "status": status,
            "finished_at": time.time(),
            "actual_api_cost_usd": round(spent, 6),
            "records_processed": processed,
            "records_failed": failed,
            "output_uri": str(data["output_uri"]),
            **_volume_artifact(str(data["output_uri"]), output_path, sha_key="output_sha256"),
        })
    return {
        "run_id": run_id,
        "status": status,
        "records_processed": processed,
        "records_failed": failed,
        "actual_api_cost_usd": round(spent, 6),
    }


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=2 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def rewrite_judging_worker(run_id: str, payload: dict) -> dict:
    """Run the frozen, resumable, two-family M3 blinded rewrite judge."""
    import requests
    from data.m3_rewrite_judge import build_tasks, summarize

    config = payload["config"]
    data = config["data"]
    judge = config["judge"]
    panel_path = _volume_path(str(data["panel_uri"]))
    sft_path = _volume_path(str(data["sft_uri"]))
    treatment_path = _volume_path(str(data["treatment_uri"]))
    output_path = _volume_path(str(data["output_uri"]))
    manifest_path = _volume_path(str(data["manifest_uri"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cost_cap = float(payload["api_reserved_cost_usd"])
    spent = 0.0
    reported = 0.0
    processed = 0
    failed = 0
    status = "failed"

    def rows(path: Path, expected_sha: str) -> list[dict]:
        if not path.is_file() or path.is_symlink() or _file_sha256(path) != expected_sha:
            raise RuntimeError(f"judge artifact binding failed: {path.name}")
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    class JudgeRequestError(RuntimeError):
        def __init__(self, message: str, cost: float):
            super().__init__(message)
            self.cost = cost

    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
        "X-OpenRouter-Title": "Humanwrite M3 blinded rewrite judge",
    }

    def request_one(task: dict) -> tuple[dict, float]:
        attempt_cost = 0.0
        last_error: Exception | None = None
        for attempt in range(int(judge["retry_attempts"])):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": task["model"],
                        "messages": [{"role": "user", "content": task["prompt"]}],
                        "temperature": judge["temperature"],
                        "max_completion_tokens": judge["max_completion_tokens"],
                        "reasoning": {"effort": "minimal", "exclude": True},
                        "provider": {"require_parameters": True},
                    },
                    timeout=180,
                )
                response.raise_for_status()
                body = response.json()
                usage_cost = (body.get("usage") or {}).get("cost")
                if usage_cost is None:
                    raise RuntimeError("judge response omitted usage.cost")
                attempt_cost += float(usage_cost)
                content = str(body["choices"][0]["message"].get("content") or "").strip().upper()
                if task["task_type"] == "pairwise":
                    if re.fullmatch(r"A|B|TIE", content) is None:
                        raise RuntimeError(f"pairwise response contract failed: {content!r}")
                    task_result = {
                        "treatment_side": task["treatment_side"],
                        "choice": content,
                        "outcome": (
                            "tie"
                            if content == "TIE"
                            else "win"
                            if content == task["treatment_side"]
                            else "loss"
                        ),
                    }
                else:
                    if re.fullmatch(r"PASS|FAIL", content) is None:
                        raise RuntimeError(f"preservation response contract failed: {content!r}")
                    task_result = {
                        "arm": task["arm"],
                        "choice": content,
                        "passed": content == "PASS",
                    }
                return (
                    {
                        "artifact_schema": M3_REWRITE_JUDGE_PROTOCOL,
                        "task_id": task["task_id"],
                        "task_type": task["task_type"],
                        "model": task["model"],
                        "dimension": task["dimension"],
                        "fingerprint": task["fingerprint"],
                        "attempt": attempt + 1,
                        "cost_usd": round(attempt_cost, 8),
                        **task_result,
                    },
                    attempt_cost,
                )
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt)
        raise JudgeRequestError(str(last_error), attempt_cost)

    try:
        checkpoint_volume.reload()
        panel = rows(panel_path, str(data["panel_sha256"]))
        sft = rows(sft_path, str(data["sft_sha256"]))
        treatment = rows(treatment_path, str(data["treatment_sha256"]))
        if len(panel) != int(data["panel_records"]):
            raise RuntimeError("judge panel cardinality mismatch")
        tasks = build_tasks(panel, sft, treatment)
        task_by_id = {str(task["task_id"]): task for task in tasks}
        completed: dict[str, dict] = {}
        if output_path.exists():
            for row in rows(output_path, _file_sha256(output_path)):
                key = str(row.get("task_id") or "")
                task = task_by_id.get(key)
                common_valid = bool(
                    task is not None
                    and row.get("task_type") == task["task_type"]
                    and row.get("model") == task["model"]
                    and row.get("dimension") == task["dimension"]
                    and row.get("fingerprint") == task["fingerprint"]
                )
                pairwise_valid = bool(
                    task is not None
                    and task["task_type"] == "pairwise"
                    and row.get("choice") in {"A", "B", "TIE"}
                    and row.get("treatment_side") == task["treatment_side"]
                    and row.get("outcome")
                    == (
                        "tie"
                        if row.get("choice") == "TIE"
                        else "win"
                        if row.get("choice") == task["treatment_side"]
                        else "loss"
                    )
                )
                preservation_valid = bool(
                    task is not None
                    and task["task_type"] == "preservation"
                    and row.get("choice") in {"PASS", "FAIL"}
                    and row.get("arm") == task["arm"]
                    and row.get("passed") is (row.get("choice") == "PASS")
                )
                if (
                    row.get("artifact_schema") != M3_REWRITE_JUDGE_PROTOCOL
                    or not common_valid
                    or not (pairwise_valid or preservation_valid)
                    or key in completed
                ):
                    raise RuntimeError("existing judge output is invalid")
                completed[key] = row
        pending = [
            task for task in tasks if str(task["task_id"]) not in completed
        ]
        with output_path.open("a", encoding="utf-8") as sink:
            for start in range(0, len(pending), int(judge["concurrency"])):
                batch = pending[start : start + int(judge["concurrency"])]
                if spent >= cost_cap:
                    break
                with ThreadPoolExecutor(max_workers=int(judge["concurrency"])) as pool:
                    futures = {pool.submit(request_one, task): task for task in batch}
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            row, cost = future.result()
                            spent += cost
                            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                            sink.flush()
                            key = str(task["task_id"])
                            completed[key] = row
                            processed += 1
                        except JudgeRequestError as exc:
                            spent += exc.cost
                            failed += 1
                            with log_path.open("a", encoding="utf-8") as logs:
                                logs.write(
                                    f"record failure id={task['fingerprint']} "
                                    f"error=JudgeRequestError detail={exc}\n"
                                )
                checkpoint_volume.commit()
                if spent > reported:
                    _record({
                        "kind": "api_cost",
                        "run_id": run_id,
                        "cost_usd": round(spent - reported, 6),
                    })
                    reported = spent
                with log_path.open("a", encoding="utf-8") as logs:
                    logs.write(
                        f"processed={processed} total_completed={len(completed)} "
                        f"api_cost_usd={spent:.6f} concurrency={judge['concurrency']}\n"
                    )
        if len(completed) == len(tasks) and failed == 0:
            result_rows = [completed[key] for key in sorted(completed)]
            output_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in result_rows
                ),
                encoding="utf-8",
            )
            summary = summarize(result_rows)
            manifest = {
                "artifact_schema": "humanwrite.m3.rewrite_judge_manifest.v2",
                "status": "completed",
                "comparison_id": config["run"]["comparison_id"],
                "input_hashes": {
                    "panel": data["panel_sha256"],
                    "sft": data["sft_sha256"],
                    "treatment": data["treatment_sha256"],
                },
                "results_uri": data["output_uri"],
                "results_sha256": _file_sha256(output_path),
                "summary": summary,
                "cost_usd": round(sum(float(row["cost_usd"]) for row in result_rows), 6),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            checkpoint_volume.commit()
            status = "completed"
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(f"M3 rewrite judge failure: {type(exc).__name__}: {exc}\n")
    finally:
        if spent > reported:
            _record({
                "kind": "api_cost",
                "run_id": run_id,
                "cost_usd": round(spent - reported, 6),
            })
        update = {
            "kind": "run_update",
            "run_id": run_id,
            "status": status,
            "finished_at": time.time(),
            "actual_api_cost_usd": round(spent, 6),
            "records_processed": processed,
            "records_failed": failed,
            "output_uri": str(data["output_uri"]),
            **_volume_artifact(str(data["output_uri"]), output_path, sha_key="output_sha256"),
        }
        if manifest_path.is_file():
            update["manifest_uri"] = str(data["manifest_uri"])
            update["manifest_sha256"] = _file_sha256(manifest_path)
        _record(update)
    return {
        "run_id": run_id,
        "status": status,
        "records_processed": processed,
        "records_failed": failed,
        "actual_api_cost_usd": round(spent, 6),
    }


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=8 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def model_cache_worker(run_id: str, payload: dict) -> dict:
    """Populate the pinned 14B snapshot on CPU before renting an accelerator."""
    from huggingface_hub import snapshot_download

    started = time.time()
    config = payload["config"]
    model = config["model"]
    log_path = _log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = "failed"
    snapshot_path = ""
    try:
        snapshot_path = snapshot_download(
            repo_id=str(model["base"]),
            revision=str(model["revision"]),
            token=os.environ.get("HF_TOKEN"),
            cache_dir="/checkpoints/hf-cache",
            allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
            max_workers=16,
        )
        snapshot = Path(snapshot_path)
        shards = sorted(snapshot.glob("*.safetensors"))
        if not shards or not (snapshot / "model.safetensors.index.json").is_file():
            raise RuntimeError("pinned Qwen3-14B cache is incomplete")
        manifest = {
            "artifact_schema": "humanwrite.model_cache_result.v1",
            "run_id": run_id,
            "base_model": str(model["base"]),
            "revision": str(model["revision"]),
            "snapshot_path": snapshot_path,
            "safetensor_files": len(shards),
            "safetensor_bytes": sum(path.stat().st_size for path in shards),
        }
        write_path = _artifact_dir(run_id) / "model_cache_manifest.json"
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        log_path.write_text(
            f"model cache complete shards={len(shards)} bytes={manifest['safetensor_bytes']}\n",
            encoding="utf-8",
        )
        checkpoint_volume.commit()
        status = "completed"
    except Exception as exc:
        log_path.write_text(
            f"model cache failure: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        checkpoint_volume.commit()
    finally:
        _record(
            {
                "kind": "run_update",
                "run_id": run_id,
                "status": status,
                "finished_at": time.time(),
                "accel_seconds": 0.0,
                "actual_cost_usd": 0.0,
                "artifact_dir": str(_artifact_dir(run_id)),
                "wall_seconds": round(time.time() - started, 3),
                "snapshot_path": snapshot_path,
            }
        )
    return {"run_id": run_id, "status": status, "snapshot_path": snapshot_path}


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
    force_empty_quotations = bool((config.get("api") or {}).get("force_empty_quotations"))
    prompt_repair_only = bool((config.get("api") or {}).get("prompt_repair_only"))
    cost_cap = float(payload["api_reserved_cost_usd"])
    spent = 0.0
    reported_spent = 0.0
    processed = 0
    failed_records = 0
    status = "failed"
    log_path = _log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = os.environ["DFTR_OPENROUTER_MODEL"]
        if str((config.get("api") or {}).get("model")) != model:
            raise ValueError("brief synthesis model does not match the frozen deployment model")
        fallback_model = str((config.get("api") or {}).get("fallback_model") or "")
        if fallback_model and fallback_model != BRIEF_FALLBACK_MODEL:
            raise ValueError("brief synthesis fallback model is not allowlisted")
        api_key = os.environ["OPENROUTER_API_KEY"]
        checkpoint_volume.reload()
        if not input_path.is_file():
            raise FileNotFoundError(f"input artifact not found: {data_config['input_uri']}")
        if _file_sha256(input_path) != str(data_config["input_sha256"]):
            raise ValueError("brief synthesis input SHA-256 mismatch")
        records = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:max_records]
        target_ids = {record_id(record) for record in records}
        empty_outline_ids = exact_empty_outline_ids(records)
        source_index = {record_id(record): record for record in records}
        completed_ids = set()
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                source_id = record_id(row)
                if source_id in completed_ids or source_id not in source_index:
                    raise ValueError("existing brief output has duplicate or unknown record ID")
                source = source_index[source_id]
                if prompt_repair_only:
                    for field, source_value in source.items():
                        if field != "user_prompt" and row.get(field) != source_value:
                            raise ValueError("prompt repair changed a frozen source field")
                    validate_repaired_user_prompt(
                        row.get("user_prompt"), source_text=str(source["completion"])
                    )
                else:
                    for field in (
                        "completion", "domain", "fineweb_id", "fingerprint", "source_config",
                        "source_revision", "split", "url", "word_count",
                    ):
                        if row.get(field) != source.get(field):
                            raise ValueError("existing brief output changed a source field")
                    validate_brief(
                        row,
                        source_text=str(source["completion"]),
                        force_empty_outline=source_id in empty_outline_ids,
                    )
                    if bool(row.get("outline")) == (source_id in empty_outline_ids):
                        raise ValueError("existing brief output has wrong empty-outline assignment")
                completed_ids.add(source_id)
        if force_empty_quotations:
            max_missing = int((config.get("recovery") or {}).get("max_missing_records", 0))
            missing_count = len(target_ids - completed_ids)
            if max_missing <= 0 or missing_count > max_missing:
                raise ValueError(
                    "quote-free recovery exceeds the preregistered missing-record bound"
                )
        with output_path.open("a", encoding="utf-8") as sink:
            for record in records:
                source_id = record_id(record)
                if spent >= cost_cap:
                    break
                if source_id in completed_ids:
                    continue
                text = str(record.get("completion") or record.get("text") or "")
                if not text.strip():
                    failed_records += 1
                    continue
                brief = None
                last_error = None
                safety_neutral = False
                for _attempt in range(2):
                    try:
                        active_model = fallback_model if safety_neutral and fallback_model else model
                        request_payload = structured_chat_request(
                            model=active_model,
                            prompt=(
                                _prompt_repair_prompt(text[:120_000])
                                if prompt_repair_only
                                else _brief_prompt(
                                    text[:120_000],
                                    force_empty_outline=source_id in empty_outline_ids,
                                    safety_neutral=safety_neutral,
                                )
                            ),
                            response_format=(
                                prompt_repair_response_format()
                                if prompt_repair_only
                                else brief_response_format(
                                    force_empty_outline=source_id in empty_outline_ids
                                )
                            ),
                            max_completion_tokens=4000,
                            reasoning=(
                                {"effort": "minimal", "exclude": True}
                                if active_model == model
                                else None
                            ),
                        )
                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                                "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                                "X-OpenRouter-Title": "Humanwrite DFT-R brief synthesis",
                            },
                            json=request_payload,
                            timeout=180,
                        )
                        response.raise_for_status()
                        body = response.json()
                        usage_cost = (body.get("usage") or {}).get("cost")
                        if usage_cost is None:
                            raise RuntimeError(
                                "OpenRouter response omitted usage.cost; refusing unmetered call"
                            )
                        spent += float(usage_cost)
                        choice = body["choices"][0]
                        finish_reason = str(choice.get("finish_reason") or "unknown")
                        if finish_reason != "stop":
                            if finish_reason == "content_filter":
                                safety_neutral = True
                            raise RuntimeError(f"provider finish_reason={finish_reason}")
                        content_value = choice["message"].get("content")
                        if not isinstance(content_value, str):
                            raise TypeError("provider message content was not a string")
                        content = content_value.strip()
                        if content.startswith("```"):
                            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                        brief_value = json.loads(content)
                        if force_empty_quotations:
                            brief_value = empty_brief_quotations(brief_value)
                        brief = (
                            {"user_prompt": validate_repaired_user_prompt(
                                brief_value, source_text=text
                            )}
                            if prompt_repair_only
                            else validate_brief(
                                brief_value,
                                source_text=text,
                                force_empty_outline=source_id in empty_outline_ids,
                            )
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                if brief is None:
                    failed_records += 1
                    error_detail = re.sub(r"\s+", " ", str(last_error or "unknown"))[:160]
                    with log_path.open("a", encoding="utf-8") as logs:
                        logs.write(
                            f"record failure id={source_id} error={type(last_error).__name__} "
                            f"detail={error_detail}\n"
                        )
                    continue
                emitted = dict(record)
                emitted.update(brief)
                if not prompt_repair_only:
                    emitted["generation_mode"] = "generate"
                    emitted["completion"] = text
                sink.write(json.dumps(emitted, ensure_ascii=False, sort_keys=True) + "\n")
                sink.flush()
                processed += 1
                completed_ids.add(source_id)
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
        status = "completed" if completed_ids == target_ids and not failed_records else "failed"
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
            "records_failed": failed_records,
            "output_uri": str(data_config["output_uri"]),
            **_volume_artifact(str(data_config["output_uri"]), output_path, sha_key="output_sha256"),
        })
    return {"run_id": run_id, "status": status, "records_processed": processed,
            "records_failed": failed_records,
            "actual_api_cost_usd": round(spent, 6)}


@app.function(
    image=worker_image,
    secrets=[provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
    timeout=8 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def document_cleaning_worker(run_id: str, payload: dict) -> dict:
    """Qwen3-32B selects original lines; fixed code performs the only edit."""
    import requests

    config = payload["config"]
    data = config["data"]
    input_path = _volume_path(str(data["input_uri"]))
    output_path = _volume_path(str(data["output_uri"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cost_cap = float(payload["api_reserved_cost_usd"])
    spent = reported = 0.0
    processed = failed = 0
    status = "failed"
    log_path = _log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def clean_one(record: dict, api_key: str) -> tuple[dict | None, Exception | None, float]:
        source_text = str(record.get("completion") or "")
        source_fingerprint = str(record.get("fingerprint") or "")
        record_spent = 0.0
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                    "X-OpenRouter-Title": "Humanwrite FineWeb line cleaning",
                },
                json=structured_chat_request(
                    model=CLEANING_MODEL,
                    prompt=numbered_cleaning_prompt(
                        source_text,
                        min_word_count=int(config["quality"]["min_word_count"]),
                        max_word_count=int(config["quality"]["max_word_count"]),
                    ),
                    response_format=cleaning_response_format(),
                    max_completion_tokens=2000,
                    reasoning={"effort": "minimal", "exclude": True},
                ),
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            usage_cost = (body.get("usage") or {}).get("cost")
            if usage_cost is None:
                raise RuntimeError("OpenRouter response omitted usage.cost")
            record_spent = float(usage_cost)
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise RuntimeError(f"provider finish_reason={choice.get('finish_reason')}")
            value = json.loads(str(choice["message"]["content"]).strip())
            cleaned = apply_line_selection(value, source_text=source_text)
            word_count = len(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", cleaned))
            limits = config["quality"]
            if not int(limits["min_word_count"]) <= word_count <= int(limits["max_word_count"]):
                raise ValueError("cleaned document violates word-count bounds")
            emitted = dict(record)
            emitted.update({
                "source_fingerprint": source_fingerprint,
                "source_word_count": record.get("word_count"),
                "completion": cleaned,
                "fingerprint": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                "word_count": word_count,
                "cleaning_model": CLEANING_MODEL,
                "cleaning_mode": "ordered_original_line_subset.v1",
            })
            return emitted, None, record_spent
        except Exception as exc:
            return None, exc, record_spent

    try:
        if config["api"]["model"] != CLEANING_MODEL:
            raise ValueError("document cleaning requires the frozen Qwen3-32B route")
        checkpoint_volume.reload()
        if not input_path.is_file() or _file_sha256(input_path) != data["input_sha256"]:
            raise ValueError("document-cleaning input hash mismatch")
        records = [
            json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][: int(data["max_records"])]
        target_records = int(data["target_records"])
        completed = set()
        if output_path.exists():
            for raw in output_path.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    row = json.loads(raw)
                    source_id = str(row.get("source_fingerprint") or "")
                    if not source_id or source_id in completed:
                        raise ValueError("existing cleaning output has invalid provenance")
                    completed.add(source_id)
        api_key = os.environ["OPENROUTER_API_KEY"]
        pending = [record for record in records if str(record.get("fingerprint") or "") not in completed]
        with (
            output_path.open("a", encoding="utf-8") as sink,
            ThreadPoolExecutor(max_workers=DOCUMENT_CLEANING_CONCURRENCY) as pool,
        ):
            for offset in range(0, len(pending), DOCUMENT_CLEANING_CONCURRENCY):
                if len(completed) >= target_records or spent >= cost_cap:
                    break
                batch = pending[offset : offset + DOCUMENT_CLEANING_CONCURRENCY]
                results = [future.result() for future in [pool.submit(clean_one, record, api_key) for record in batch]]
                for record, (emitted, error, record_spent) in zip(batch, results):
                    spent += record_spent
                    source_fingerprint = str(record.get("fingerprint") or "")
                    if len(completed) >= target_records:
                        continue
                    if emitted is None:
                        failed += 1
                        with log_path.open("a", encoding="utf-8") as logs:
                            logs.write(
                                f"record failure id={source_fingerprint} "
                                f"error={type(error).__name__} detail={str(error)[:160]}\n"
                            )
                    else:
                        sink.write(json.dumps(emitted, ensure_ascii=False, sort_keys=True) + "\n")
                        sink.flush()
                        completed.add(source_fingerprint)
                        processed += 1
                checkpoint_volume.commit()
                if processed and (processed % 50 == 0 or spent - reported >= 0.02):
                    _record({"kind": "api_cost", "run_id": run_id, "cost_usd": round(spent - reported, 6)})
                    reported = spent
                    with log_path.open("a", encoding="utf-8") as logs:
                        logs.write(
                            f"processed={processed} total_completed={len(completed)} "
                            f"api_cost_usd={spent:.6f} concurrency={DOCUMENT_CLEANING_CONCURRENCY}\n"
                        )
        checkpoint_volume.commit()
        status = "completed" if len(completed) == target_records else "failed"
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as logs:
            logs.write(f"document cleaning failure: {type(exc).__name__}: {exc}\n")
    finally:
        if spent > reported:
            _record({"kind": "api_cost", "run_id": run_id, "cost_usd": round(spent - reported, 6)})
        _record({"kind": "run_update", "run_id": run_id, "status": status,
                 "finished_at": time.time(), "actual_api_cost_usd": round(spent, 6),
                 "records_processed": processed, "records_failed": failed,
                 "output_uri": str(data["output_uri"]),
                 **_volume_artifact(str(data["output_uri"]), output_path, sha_key="output_sha256")})
    return {"run_id": run_id, "status": status, "records_processed": processed,
            "records_failed": failed, "actual_api_cost_usd": round(spent, 6)}


@app.function(
    image=base_image,
    secrets=[gateway_secret, provider_secret],
    volumes={"/state": state_volume, CHECKPOINT_PATH: checkpoint_volume},
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
            policy = validate_launch(payload, backend="modal")
        except PolicyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        run_id = str(payload.get("run_id", ""))
        if not re.fullmatch(r"dftr-[0-9]+-[0-9a-f]{8}", run_id):
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
            "workflow_step": str((payload.get("config") or {}).get("workflow", {}).get("step") or ""),
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
                brief_api = (payload.get("config") or {}).get("api") or {}
                worker = (
                    lower_variance_brief_synthesis_worker
                    if brief_api.get("protocol") == LOWER_VARIANCE_BRIEF_PROTOCOL
                    else brief_synthesis_worker
                )
                call = worker.with_options(
                    timeout=policy.timeout_seconds + 120,
                ).spawn(run_id, worker_payload)
            elif policy.task_kind == "document_cleaning":
                call = document_cleaning_worker.with_options(
                    timeout=policy.timeout_seconds + 120,
                ).spawn(run_id, worker_payload)
            elif policy.task_kind == "rewrite_synthesis":
                call = rewrite_synthesis_worker.with_options(
                    timeout=policy.timeout_seconds + 120,
                ).spawn(run_id, worker_payload)
            elif policy.task_kind == "rewrite_judging":
                call = rewrite_judging_worker.with_options(
                    timeout=policy.timeout_seconds + 120,
                ).spawn(run_id, worker_payload)
            elif policy.task_kind == "model_cache":
                call = model_cache_worker.with_options(
                    timeout=policy.timeout_seconds + 120,
                ).spawn(run_id, worker_payload)
            elif policy.task_kind == "source_materialization":
                call = source_materialization_worker.with_options(
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
        if state.get("status") == "running" and state.get("function_call_id"):
            try:
                result = modal.FunctionCall.from_id(state["function_call_id"]).get(timeout=0)
            except TimeoutError:
                result = None
            except Exception as exc:
                _record({
                    "kind": "run_update", "run_id": run_id, "status": "failed",
                    "finished_at": time.time(), "worker_result_error": type(exc).__name__,
                })
                result = None
            if isinstance(result, dict):
                _record({"kind": "run_update", **result})
            state = run_snapshot(_events(), run_id) or state
        state = enrich_running_api_state(state, _log_path(run_id))
        if (
            state.get("status") == "completed"
            and state.get("workflow_step") in {"generate_dft", "generate_lower_variance"}
            and not state.get("wrapper_receipt_sha256")
        ):
            try:
                receipt = finalize_generation_receipt.remote(
                    run_id,
                    {
                        "config_hash": state["config_hash"],
                        "git_sha": state["git_sha"],
                        "comparison": state["comparison"],
                    },
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"generation receipt finalization failed: {type(exc).__name__}",
                ) from exc
            _record({
                "kind": "run_update", "run_id": run_id,
                "wrapper_receipt_path": receipt["path"],
                "wrapper_receipt_sha256": receipt["sha256"],
            })
            state = run_snapshot(_events(), run_id) or state
        try:
            checkpoint_volume.reload()
        except Exception:
            pass
        metadata = missing_run_artifact_metadata(state, mount_path=CHECKPOINT_PATH)
        if metadata:
            _record({"kind": "run_update", "run_id": run_id, **metadata})
            state = run_snapshot(_events(), run_id) or state
        return {key: value for key, value in state.items() if key != "function_call_id"}

    @api.get("/logs/{run_id}")
    def logs(run_id: str, tail: int = Query(default=200, ge=1, le=5000),
             authorization: str | None = Header(default=None)):
        require_auth(authorization)
        if not run_snapshot(_events(), run_id):
            raise HTTPException(status_code=404, detail="run not found")
        try:
            checkpoint_volume.reload()
        except Exception:
            pass
        path = _log_path(run_id)
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
            # Current Modal call cancellation requires non-forced cancellation.
            # These workers are single-use containers, so the cancelled call still
            # owns no reusable execution state after it exits.
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=False)
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
