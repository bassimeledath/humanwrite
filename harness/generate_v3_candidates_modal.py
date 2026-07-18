"""Generate matched candidate outputs after measurement-v3 is frozen."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import modal


APP_NAME = "humanwrite-measurement-v3-candidates"
CHECKPOINT_ROOT = Path("/checkpoints")
PANEL_ROOT = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/measurement-v3-panels"
PROTOCOL_ROOT = PANEL_ROOT / "protocol-v1"
PROTOCOL_PATH = PROTOCOL_ROOT / "measurement_protocol_v3.json"
BRIEFS_PATH = PANEL_ROOT / "prompt_briefs-128-normalized.jsonl"
PROMPT_MANIFEST_PATH = PANEL_ROOT / "prompt_sources.manifest.json"
OUTPUT_ROOT = PANEL_ROOT / "candidate-outputs-v1"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
BRIEFS_SHA256 = "c5371cff6e35cc0695082ab060d65bb2e0b6549ba6a2c0f58c488afbb3c06732"
METHOD_CONTRACT_SHA256 = "0b2a741bc412803bb8ff0d28ad902ea6cbbda32fb87b963d3afc72f11d37cdb6"
MASTER_SEED = 96001
MAX_NEW_TOKENS = 64

ARMS = {
    "SFT": {
        "checkpoint": CHECKPOINT_ROOT / "runs/dftr-1784339215-2799dbae/SFT",
        "config_sha256": "703d185168ccc206a7d7da1052c8a3a74b1c58c99172ffa5ebe7dfe74a3c64e5",
        "resumed_from_step": 0,
    },
    "TOKEN_MOMENT": {
        "checkpoint": CHECKPOINT_ROOT
        / "runs/dftr-1784340692-ee851016/TOKEN_MOMENT",
        "config_sha256": "ec6d07fbc2856da64a5a231c96b55ce0945317b4c353aa7a79432a10f0b87bbd",
        "resumed_from_step": 320,
    },
    "MMD_WITNESS": {
        "checkpoint": CHECKPOINT_ROOT
        / "runs/dftr-1784339214-065c0f7a/MMD_WITNESS",
        "config_sha256": "a48e3df9ae205c3c0cc8cb9707c76ff51742ed579201d8c713f8fcce8e198c",
        "resumed_from_step": 0,
    },
}

checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.13.0",
    "transformers==4.57.6",
    "accelerate>=1.8,<2",
    "peft==0.19.1",
    "huggingface-hub>=0.33,<1",
)
app = modal.App(APP_NAME)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
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


def _record_seed(prompt_id: str) -> int:
    digest = hashlib.sha256(f"{MASTER_SEED}:{prompt_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _validate_checkpoint(arm: str, checkpoint: Path) -> dict:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{arm} has no completed checkpoint manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = ARMS[arm]
    if (
        manifest.get("artifact_schema")
        != "dftr.m2.lower_variance_adapter_checkpoint.v1"
        or manifest.get("status") != "completed"
        or manifest.get("arm") != arm
        or manifest.get("steps") != 512
        or manifest.get("optimizer_examples") != 1024
        or manifest.get("config_sha256") != expected["config_sha256"]
        or manifest.get("resumed_from_step") != expected["resumed_from_step"]
        or manifest.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or manifest.get("base_model") != MODEL_ID
        or manifest.get("base_revision") != MODEL_REVISION
    ):
        raise RuntimeError(f"{arm} checkpoint does not satisfy the frozen contract")
    model_sha = manifest.get("file_sha256", {}).get("adapter_model.safetensors")
    if not model_sha or _sha(checkpoint / "adapter_model.safetensors") != model_sha:
        raise RuntimeError(f"{arm} adapter hash mismatch")
    return manifest


@app.function(
    image=image,
    gpu="L40S",
    timeout=45 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def generate_arm(arm: str, git_sha: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_volume.reload()
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if _sha(BRIEFS_PATH) != BRIEFS_SHA256:
        raise RuntimeError("evaluation brief hash mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        protocol.get("artifact_schema") != "dftr.measurement.protocol.v3"
        or protocol.get("status") != "ready"
        or protocol.get("frozen") is not True
        or protocol.get("candidate_outputs_opened") is not False
    ):
        raise RuntimeError("measurement-v3 protocol was not frozen candidate-blind")
    protocol_sha = _sha(PROTOCOL_PATH)
    checkpoint = ARMS[arm]["checkpoint"]
    checkpoint_manifest = _validate_checkpoint(arm, checkpoint)
    rows = _rows(BRIEFS_PATH)
    if len(rows) != 128:
        raise RuntimeError("measurement-v3 requires exactly 128 prompts")
    fingerprints = [str(row.get("fingerprint") or "") for row in rows]
    prompt_ids = [f"prompt-{item}" for item in fingerprints]
    if any(not item for item in fingerprints) or len(set(prompt_ids)) != 128:
        raise RuntimeError("evaluation prompt identities are incomplete")
    prompt_manifest = json.loads(PROMPT_MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen_prompt_ids = {
        str(record.get("prompt_id") or "")
        for record in prompt_manifest.get("records", [])
    }
    if set(prompt_ids) != frozen_prompt_ids:
        raise RuntimeError("evaluation briefs do not match the frozen prompt panel")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir=str(CHECKPOINT_ROOT / "hf-cache"),
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base, checkpoint, local_files_only=True, is_trainable=False
    )
    model.eval()
    output_rows = []
    with torch.inference_mode():
        for row, prompt_id in zip(rows, prompt_ids):
            seed = _record_seed(prompt_id)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            encoded = tokenizer(
                _prompt(row),
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            sequence = model.generate(
                **encoded,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            continuation = sequence[0, encoded["input_ids"].shape[1] :]
            text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError(f"empty {arm} output for {prompt_id}")
            output_rows.append(
                {
                    "arm": arm,
                    "prompt_id": prompt_id,
                    "generated_completion": text,
                    "sampling_seed": seed,
                    "generated_token_count": int(len(continuation)),
                    "visible_token_count": int(
                        len(tokenizer.encode(text, add_special_tokens=False))
                    ),
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{arm}.jsonl"
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in output_rows
    )
    output_path.write_text(payload, encoding="utf-8")
    manifest = {
        "artifact_schema": "dftr.measurement.candidate_outputs.v3",
        "arm": arm,
        "status": "completed",
        "candidate_outputs_opened_after_protocol_freeze": True,
        "git_sha": git_sha,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "briefs_sha256": BRIEFS_SHA256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_manifest_sha256": _sha(checkpoint / "checkpoint_manifest.json"),
        "adapter_model_sha256": checkpoint_manifest["file_sha256"][
            "adapter_model.safetensors"
        ],
        "sampling": {
            "seed_derivation": "sha256(master_seed:prompt_id).uint63.v1",
            "master_seed": MASTER_SEED,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_on_eos": True,
        },
        "documents": len(output_rows),
        "output_path": str(output_path),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    manifest_path = OUTPUT_ROOT / f"{arm}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    git_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    results = list(generate_arm.starmap([(arm, git_sha) for arm in ARMS]))
    print(json.dumps(results, indent=2, sort_keys=True))
