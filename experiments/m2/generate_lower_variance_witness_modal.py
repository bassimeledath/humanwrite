"""Generate the fixed baseline panel for the lower-variance three-arm screen.

This is a one-time, training-only artifact.  It samples the frozen seed-11 SFT
adapter over every normalized training brief, using the exact EOS-aware
generation contract consumed by ``lower_variance_train.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import modal


APP_NAME = "humanwrite-lower-variance-witness"
CHECKPOINT_ROOT = Path("/checkpoints")
ADAPTER_PATH = CHECKPOINT_ROOT / "runs/dftr-1784216516-91130dd3/seed-11"
BRIEFS_PATH = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/train-briefs-1024.jsonl"
OUTPUT_DIR = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/witness-v1"
OUTPUT_PATH = OUTPUT_DIR / "baseline-generated-1024.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "baseline-generated-1024.manifest.json"

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
BRIEFS_SHA256 = "419f927ac52cfd2ee6a4420638a14730ebce045a7191dce0314508c0356bc632"
ADAPTER_MODEL_SHA256 = "b7b590ca0d40b8b51951d44beb2e7928fccabbd6a2a7290f47e927da0fb81178"
ADAPTER_CONFIG_SHA256 = "9a72a9527c48cc2acf703f40047f1a6dec59e92fdc7021d59523ba5ed6fb965c"
PROMPT_SERIALIZER_SHA256 = "1f92174518dfac375abbbbcf4ceba0659b726cabb215e0561a9fbffc4036b4a1"
GENERATION_CONTRACT = {
    "sampling_distribution": "raw_policy_categorical.v1",
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 64,
    "stop_on_eos": True,
    "post_eos_behavior": "pad_and_mask",
    "teacher_forced_eos": "append_if_absent_after_truncation",
}
SAMPLING_SEED = 41001
BATCH_SIZE = 8


checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("humanwrite-provider-secrets")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.13.0",
    "transformers==4.57.6",
    "accelerate>=1.8,<2",
    "peft==0.19.1",
    "huggingface-hub>=0.33,<1",
)
app = modal.App(APP_NAME)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prompt(record: dict) -> str:
    brief = "\n".join(
        (
            f"Writing request: {str(record['user_prompt']).strip()}",
            f"Use case: {str(record['use_case']).strip()}",
            f"Style category: {str(record['style_kind']).strip()}",
            f"Style: {str(record['style']).strip()}",
            f"Detail mode: {str(record['detail_mode']).strip()}",
            f"Target length: about {int(record['target_length'])} tokens",
            f"Em dashes allowed: {'yes' if bool(record['em_dashes_allowed']) else 'no'}",
            "Grounding outline (use only these supported facts when non-empty): "
            + json.dumps(
                record["outline"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    return f"USER:\n{brief}\nASSISTANT:"


@app.function(
    image=image,
    gpu="L40S",
    timeout=90 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def generate(git_sha: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_volume.reload()
    for path in (ADAPTER_PATH, BRIEFS_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    if _file_sha256(BRIEFS_PATH) != BRIEFS_SHA256:
        raise RuntimeError("normalized training brief hash mismatch")
    if _file_sha256(ADAPTER_PATH / "adapter_model.safetensors") != ADAPTER_MODEL_SHA256:
        raise RuntimeError("initial adapter model hash mismatch")
    if _file_sha256(ADAPTER_PATH / "adapter_config.json") != ADAPTER_CONFIG_SHA256:
        raise RuntimeError("initial adapter config hash mismatch")

    rows = _read_jsonl(BRIEFS_PATH)
    if len(rows) != 1024:
        raise RuntimeError(f"expected 1024 training briefs, found {len(rows)}")
    prompt_ids = [str(row.get("source_fingerprint") or row.get("fingerprint") or "") for row in rows]
    if any(not prompt_id for prompt_id in prompt_ids) or len(set(prompt_ids)) != len(rows):
        raise RuntimeError("training briefs need 1024 unique prompt IDs")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SAMPLING_SEED)
    torch.cuda.manual_seed_all(SAMPLING_SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base, ADAPTER_PATH, local_files_only=True, is_trainable=False
    )
    model.eval()

    generated_rows: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            encoded = tokenizer(
                [_prompt(row) for row in batch],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            sequences = model.generate(
                **encoded,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            continuation = sequences[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for offset, (row, text) in enumerate(zip(batch, texts)):
                text = text.strip()
                if not text:
                    raise RuntimeError(f"empty baseline generation at row {start + offset}")
                generated_rows.append(
                    {
                        "prompt_id": prompt_ids[start + offset],
                        "generated_completion": text,
                        "sampling_seed": SAMPLING_SEED,
                        "batch_index": start // BATCH_SIZE,
                        "source_fingerprint": str(
                            row.get("source_fingerprint") or row.get("fingerprint")
                        ),
                    }
                )

    output_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in generated_rows
    )
    output_sha256 = hashlib.sha256(output_payload.encode("utf-8")).hexdigest()
    manifest = {
        "artifact_schema": "dftr.m2.lower_variance_baseline_witness.v1",
        "scientific_role": "training_only_not_evaluation",
        "git_sha": git_sha,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_path": str(ADAPTER_PATH),
        "adapter_model_sha256": ADAPTER_MODEL_SHA256,
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "briefs_path": str(BRIEFS_PATH),
        "briefs_sha256": BRIEFS_SHA256,
        "prompt_serializer_sha256": PROMPT_SERIALIZER_SHA256,
        "generation_contract": GENERATION_CONTRACT,
        "generation_contract_sha256": _canonical_hash(GENERATION_CONTRACT),
        "sampling_seed": SAMPLING_SEED,
        "batch_size": BATCH_SIZE,
        "documents": len(generated_rows),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": output_sha256,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output_payload, encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    root = Path(__file__).resolve().parents[2]
    git_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    print(json.dumps(generate.remote(git_sha), indent=2, sort_keys=True))
