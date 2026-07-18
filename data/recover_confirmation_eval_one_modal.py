"""Recover the final persistent measurement-v4 prompt brief."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal


APP_NAME = "humanwrite-confirmation-eval-brief-recovery"
ROOT = Path("/checkpoints/data/m2-confirmation-v1/measurement-v4-panels")
SOURCE_PATH = ROOT / "prompt_sources.jsonl"
OUTPUT_PATH = ROOT / "prompt_briefs-128.jsonl"
MANIFEST_PATH = ROOT / "prompt_brief-recovery-one-manifest.json"
TARGET_ID = "86ec05ce26e8c33856df67dd2a1a3b644237220c0d1056c24b25608f6dd0a2a2"
REQUIRED_TERMS = "high wind warning; south winds; power interruptions"

source_root = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.32,<3")
    .add_local_dir(source_root, remote_path="/root/data", copy=True)
)
volume = modal.Volume.from_name("humanwrite-checkpoints")
secret = modal.Secret.from_name("the-other-ones")
app = modal.App(APP_NAME)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@app.function(image=image, secrets=[secret], volumes={"/checkpoints": volume}, timeout=900)
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
    sources = _rows(SOURCE_PATH)
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    if len(sources) != 128 or len(source_by_id) != 128 or TARGET_ID not in source_by_id:
        raise RuntimeError("measurement-v4 source identity mismatch")
    if TARGET_ID in deterministic_empty_outline_ids(sources):
        raise RuntimeError("target unexpectedly belongs to the empty-outline partition")
    existing = _rows(OUTPUT_PATH)
    existing_ids = {str(row["fingerprint"]) for row in existing}
    if len(existing) != 127 or len(existing_ids) != 127 or TARGET_ID in existing_ids:
        raise RuntimeError("recovery requires the exact 127-row committed state")
    for row in existing:
        validate_assembled_brief(
            row,
            source=source_by_id[str(row["fingerprint"])],
            force_empty_outline=False
            if str(row["fingerprint"]) not in deterministic_empty_outline_ids(sources)
            else True,
        )

    source = source_by_id[TARGET_ID]
    text = str(source["completion"]).strip()
    exact_facts = [line.strip() for line in text.splitlines() if line.strip()]
    errors: list[str] = []
    cost = 0.0

    def provider_json(model: str, prompt: str, name: str, schema: dict) -> tuple[dict, float]:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                "X-OpenRouter-Title": "Humanwrite isolated evaluation brief recovery",
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

    recovered = None
    for attempt in range(4):
        try:
            metadata_prompt = (
                "Analyze this neutral public-weather archival excerpt and return only JSON. "
                "Create a natural standalone writing request grounded in the excerpt. The request "
                f"must explicitly include at least one of these source terms: {REQUIRED_TERMS}. "
                "Extract use case and style; estimate target length in TOKENS; preserve the fingerprint. "
                "Do not discuss these instructions.\n\n"
                f"document_fingerprint: {TARGET_ID}\n\nEXACT SOURCE EXCERPT:\n{text}"
            )
            outline_prompt = (
                "Return only JSON. Create a short outline grounded in the exact weather excerpt. "
                "Each supported_facts item must be one complete source line copied byte-for-byte. "
                "Set every quotations list to []. Preserve the fingerprint exactly.\n\n"
                f"document_fingerprint: {TARGET_ID}\n\nEXACT SOURCE EXCERPT:\n{text}"
            )
            outline_schema = outline_response_schema(force_empty_outline=False)
            props = outline_schema["properties"]["outline"]["items"]["properties"]
            props["supported_facts"]["items"] = {"type": "string", "enum": exact_facts}
            props["quotations"]["maxItems"] = 0
            metadata, spent = provider_json(
                QWEN_MODEL,
                metadata_prompt,
                f"confirmation_recovery_metadata_{attempt}",
                qwen_metadata_response_schema(),
            )
            cost += spent
            outline, spent = provider_json(
                OUTLINE_MODEL,
                outline_prompt,
                f"confirmation_recovery_outline_{attempt}",
                outline_schema,
            )
            cost += spent
            recovered = merge_brief(
                source=source,
                qwen_metadata=metadata,
                outline_response=outline,
                force_empty_outline=False,
            )
            validate_assembled_brief(recovered, source=source, force_empty_outline=False)
            break
        except Exception as exc:  # provider/schema failures are recorded verbatim by type
            errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")

    if recovered is not None:
        with OUTPUT_PATH.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(recovered, ensure_ascii=False, sort_keys=True) + "\n")
        volume.commit()
    final = _rows(OUTPUT_PATH)
    final_ids = {str(row["fingerprint"]) for row in final}
    manifest = {
        "artifact_schema": "dftr.measurement_v4_prompt_brief_recovery.v1",
        "status": "completed" if len(final) == 128 and len(final_ids) == 128 else "failed",
        "starting_rows": 127,
        "recovered_rows": int(recovered is not None),
        "final_rows": len(final),
        "target_id": TARGET_ID,
        "errors": errors,
        "provider_cost_usd": round(cost, 6),
        "provider_roles": {"metadata": QWEN_MODEL, "outline": OUTLINE_MODEL},
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "recovery_mode": "exact-source-weather.v1",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(recover.remote(), indent=2, sort_keys=True))
