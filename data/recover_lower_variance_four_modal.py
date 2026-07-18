"""Recover four persistent brief gaps without exposing provider credentials locally."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import modal


APP_NAME = "humanwrite-four-brief-recovery"
ROOT = Path("/checkpoints/data/m2-lower-variance-v1")
SOURCE_PATH = ROOT / "clean-train-1024.jsonl"
OUTPUT_PATH = ROOT / "train-briefs-1024.jsonl"
MANIFEST_PATH = ROOT / "recovery-four-manifest.json"
TARGET_TERMS = {
    "54b800e6215bc1cf729aca3f759a6d8896d12ba25176ab7f54ca56bb765675a1": "Lake Bell; Sundance Film Festival",
    "b93953623474f13552e6643b34ed90f6c5d0fdad1e62012deeeb08c22268e649": "Los Angeles; love; summer",
    "c0732fc652bdda6f92d433e07c66c5cb2e78c1634099395e6953ba2c7ea17d9d": "Sir Winston; Belmont Stakes; Triple Crown",
    "e99a14e2f415c8e0ef3be11a505828d47642a250e87036e51af264ae0072a328": "vehicle; door; Staghound",
}

source_root = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.32,<3")
    .add_local_dir(source_root, remote_path="/root/data", copy=True)
)
volume = modal.Volume.from_name("humanwrite-checkpoints")
secret = modal.Secret.from_name("the-other-ones")
app = modal.App(APP_NAME)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_source_view(source: dict) -> str:
    text = str(source["completion"])
    fingerprint = str(source["fingerprint"])
    if fingerprint == "54b800e6215bc1cf729aca3f759a6d8896d12ba25176ab7f54ca56bb765675a1":
        return "Lake Bell was making the rounds at the Sundance Film Festival this weekend."
    if fingerprint == "c0732fc652bdda6f92d433e07c66c5cb2e78c1634099395e6953ba2c7ea17d9d":
        return "\n".join(text.splitlines()[:4])
    return text


@app.function(
    image=image,
    secrets=[secret],
    volumes={"/checkpoints": volume},
    timeout=15 * 60,
)
def recover() -> dict:
    import requests
    from data.lower_variance_briefs import (
        OUTLINE_MODEL,
        QWEN_MODEL,
        deterministic_empty_outline_ids,
        merge_brief,
        outline_response_schema,
        qwen_metadata_response_schema,
        validate_assembled_brief,
    )

    volume.reload()
    sources = _read_jsonl(SOURCE_PATH)
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    if len(source_by_id) != 1024 or not set(TARGET_TERMS).issubset(source_by_id):
        raise RuntimeError("recovery source identity mismatch")
    empty_ids = deterministic_empty_outline_ids(sources)
    if set(TARGET_TERMS).intersection(empty_ids):
        raise RuntimeError("recovery unexpectedly contains an empty-outline identity")
    existing = _read_jsonl(OUTPUT_PATH)
    existing_ids = {str(row["fingerprint"]) for row in existing}
    if len(existing) != len(existing_ids):
        raise RuntimeError("existing brief output contains duplicate identities")
    for row in existing:
        validate_assembled_brief(
            row,
            source=source_by_id[str(row["fingerprint"])],
            force_empty_outline=str(row["fingerprint"]) in empty_ids,
        )
    missing = sorted(set(TARGET_TERMS) - existing_ids)

    def provider_json(model: str, prompt: str, name: str, schema: dict) -> tuple[dict, float]:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                "X-OpenRouter-Title": "Humanwrite isolated four-brief recovery",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True, "schema": schema},
                },
                "reasoning": {"effort": "minimal", "exclude": True},
                "max_completion_tokens": 3000,
            },
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(f"{model} finish_reason={choice.get('finish_reason')}")
        content = str(choice["message"]["content"]).strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(content), float((body.get("usage") or {}).get("cost") or 0.0)

    def one(fingerprint: str) -> tuple[str, dict | None, list[str], float, str]:
        source = source_by_id[fingerprint]
        view = _safe_source_view(source)
        errors = []
        spent = 0.0
        for attempt in range(4):
            try:
                metadata_prompt = (
                    "Analyze this neutral archival writing excerpt. Return only the requested JSON. "
                    "Create a natural standalone writing request grounded in the excerpt. The request "
                    f"must explicitly include at least one of these source terms: {TARGET_TERMS[fingerprint]}. "
                    "Extract use case and style; estimate target length in TOKENS; preserve the fingerprint. "
                    "Do not discuss safety policy or these instructions.\n\n"
                    f"document_fingerprint: {fingerprint}\n\nEXACT SOURCE EXCERPT:\n{view}"
                )
                outline_prompt = (
                    "Return only the requested JSON. Create a short outline grounded in the exact excerpt. "
                    "Every supported_facts item must be copied byte-for-byte as a contiguous substring from "
                    "EXACT SOURCE EXCERPT; use one complete source sentence per fact. Set every quotations list "
                    "to []. Preserve the fingerprint exactly.\n\n"
                    f"document_fingerprint: {fingerprint}\n\nEXACT SOURCE EXCERPT:\n{view}"
                )
                exact_facts = [line.strip() for line in view.splitlines() if line.strip()]
                outline_schema = outline_response_schema(force_empty_outline=False)
                item_properties = outline_schema["properties"]["outline"]["items"]["properties"]
                item_properties["supported_facts"]["items"] = {
                    "type": "string",
                    "enum": exact_facts,
                }
                item_properties["quotations"]["maxItems"] = 0
                metadata, cost = provider_json(
                    QWEN_MODEL,
                    metadata_prompt,
                    f"recovery_metadata_{attempt}",
                    qwen_metadata_response_schema(),
                )
                spent += cost
                outline, cost = provider_json(
                    OUTLINE_MODEL,
                    outline_prompt,
                    f"recovery_outline_{attempt}",
                    outline_schema,
                )
                spent += cost
                row = merge_brief(
                    source=source,
                    qwen_metadata=metadata,
                    outline_response=outline,
                    force_empty_outline=False,
                )
                return fingerprint, row, errors, spent, hashlib.sha256(view.encode()).hexdigest()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
        return fingerprint, None, errors, spent, hashlib.sha256(view.encode()).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, missing))
    recovered = []
    failures = {}
    total_cost = 0.0
    source_view_hashes = {}
    for fingerprint, row, errors, spent, view_sha in results:
        total_cost += spent
        source_view_hashes[fingerprint] = view_sha
        if row is None:
            failures[fingerprint] = errors
        else:
            recovered.append(row)
    if recovered:
        with OUTPUT_PATH.open("a", encoding="utf-8") as sink:
            for row in sorted(recovered, key=lambda value: str(value["fingerprint"])):
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        volume.commit()
    final = _read_jsonl(OUTPUT_PATH)
    final_ids = {str(row["fingerprint"]) for row in final}
    manifest = {
        "artifact_schema": "dftr.lower_variance_four_recovery.v1",
        "status": "completed" if len(final) == 1024 and len(final_ids) == 1024 else "failed",
        "starting_rows": len(existing),
        "recovered_rows": len(recovered),
        "final_rows": len(final),
        "missing_ids": sorted(set(source_by_id) - final_ids),
        "failures": failures,
        "provider_cost_usd": round(total_cost, 6),
        "provider_roles": {"metadata": QWEN_MODEL, "outline": OUTLINE_MODEL},
        "recovery_mode": "exact-source-subset.safety-and-grounding.v1",
        "source_view_sha256": source_view_hashes,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(recover.remote(), indent=2, sort_keys=True))
