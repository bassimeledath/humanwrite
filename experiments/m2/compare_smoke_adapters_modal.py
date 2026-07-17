"""Generate a small, non-decision-grade pre/post comparison for the SFT smoke run."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import modal


APP_NAME = "humanwrite-smoke-comparison"
CHECKPOINT_ROOT = Path("/checkpoints")
PRE_ADAPTER = CHECKPOINT_ROOT / "runs/dftr-1784216516-91130dd3/seed-11"
POST_ADAPTER = CHECKPOINT_ROOT / "runs/dftr-1784329721-16bf1bb2/SFT"
BRIEFS_PATH = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/train-briefs-1024.jsonl"
ANCHORS_PATH = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/smoke/anchors-128.jsonl"
OUTPUT_PATH = CHECKPOINT_ROOT / "runs/dftr-1784329721-16bf1bb2/diagnostics/pre-post-samples-10.json"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.7,<3",
    "transformers==4.57.6",
    "accelerate>=1.8,<2",
    "peft>=0.16,<1",
    "huggingface-hub>=0.33,<1",
)
app = modal.App(APP_NAME)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            + json.dumps(record["outline"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    )
    return f"USER:\n{brief}\nASSISTANT:"


def _text_metrics(text: str) -> dict:
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text, re.UNICODE)
    bigrams = list(zip(words, words[1:]))
    repeated_bigrams = len(bigrams) - len(set((a.casefold(), b.casefold()) for a, b in bigrams))
    non_latin_letters = sum(
        1 for char in text if char.isalpha() and "LATIN" not in unicodedata.name(char, "")
    )
    return {
        "characters": len(text),
        "words": len(words),
        "emdashes": text.count("—"),
        "repeated_bigrams": repeated_bigrams,
        "non_latin_letters": non_latin_letters,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=20 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def compare() -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_volume.reload()
    for path in (PRE_ADAPTER, POST_ADAPTER, BRIEFS_PATH, ANCHORS_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    anchor_ids = {
        str(row.get("source_fingerprint") or row.get("fingerprint"))
        for row in _read_jsonl(ANCHORS_PATH)
    }
    candidates = [
        row
        for row in _read_jsonl(BRIEFS_PATH)
        if str(row.get("source_fingerprint") or row.get("fingerprint")) not in anchor_ids
    ]
    candidates.sort(key=lambda row: str(row.get("source_fingerprint") or row.get("fingerprint")))
    rows = candidates[:10]
    if len(rows) != 10:
        raise RuntimeError(f"need 10 held-out briefs, found {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(
        PRE_ADAPTER, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base, PRE_ADAPTER, adapter_name="before", local_files_only=True, is_trainable=False
    )
    model.load_adapter(POST_ADAPTER, adapter_name="after", is_trainable=False)
    model.eval()

    samples = []
    for index, row in enumerate(rows, start=1):
        rendered = _prompt(row)
        encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=1024)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        max_new_tokens = min(max(int(row["target_length"]), 96), 192)
        outputs = {}
        for adapter in ("before", "after"):
            model.set_adapter(adapter)
            seed = 7000 + index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=0,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            continuation = generated[0, encoded["input_ids"].shape[1] :]
            outputs[adapter] = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        samples.append(
            {
                "number": index,
                "prompt_id": str(row.get("source_fingerprint") or row.get("fingerprint")),
                "user_prompt": row["user_prompt"],
                "use_case": row["use_case"],
                "style": row["style"],
                "target_length_tokens": row["target_length"],
                "sampling_seed": 7000 + index,
                "before": outputs["before"],
                "after": outputs["after"],
                "before_metrics": _text_metrics(outputs["before"]),
                "after_metrics": _text_metrics(outputs["after"]),
                "byte_identical": outputs["before"].encode() == outputs["after"].encode(),
            }
        )

    payload = {
        "artifact_schema": "dftr.diagnostic.pre_post_samples.v1",
        "scientific_status": "informal diagnostic only; not a quality evaluation",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "before_adapter": str(PRE_ADAPTER),
        "after_adapter": str(POST_ADAPTER),
        "sampler": {"temperature": 0.8, "top_p": 0.95, "top_k": 0},
        "samples": samples,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint_volume.commit()
    return {
        "output_path": str(OUTPUT_PATH),
        "samples": len(samples),
        "byte_identical": sum(int(sample["byte_identical"]) for sample in samples),
        "content_sha256": payload["content_sha256"],
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(compare.remote(), indent=2, sort_keys=True))
