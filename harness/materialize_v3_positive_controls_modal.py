"""Build candidate-blind measurement-v3 positive controls on the fresh panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


APP_NAME = "humanwrite-measurement-v3-positive-controls"
CHECKPOINT_ROOT = Path("/checkpoints")
PANEL_ROOT = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/measurement-v3-panels"
PROMPT_BRIEFS = PANEL_ROOT / "prompt_briefs-128-normalized.jsonl"
INITIAL_ADAPTER = CHECKPOINT_ROOT / "runs/dftr-1784216516-91130dd3/seed-11"
OUTPUT_ROOT = PANEL_ROOT / "positive-controls"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPT_BRIEFS_SHA256 = "c5371cff6e35cc0695082ab060d65bb2e0b6549ba6a2c0f58c488afbb3c06732"
PANEL_DESIGN_SHA256 = "45125f9616c75c208221faaa6b54a8dd61e98779ba46bd3a731efb4f152b4b87"
TOKENIZATION_CONTRACT_SHA256 = "daa26e60d0352b16fabd93b608c8aa7538a23f2afe02b54367c85ce66509f5ac"
MASTER_SEED = 93001
ALPHA = 0.05

FAMILIES = {
    "bge-small-v1": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "batch_size": 32,
        "prompt_name": None,
    },
    "nemotron-8b-v1": {
        "model_id": "nvidia/llama-embed-nemotron-8b",
        "revision": "aa3b43a495a9b280d1bdb716da37c54bb495d630",
        "batch_size": 4,
        "prompt_name": "document",
    },
}


checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
source_root = Path(__file__).resolve().parent / "src"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0",
        "transformers==4.57.6",
        "sentence-transformers==5.2.2",
        "peft==0.19.1",
        "accelerate>=1.8,<2",
        "huggingface-hub>=0.33,<1",
    )
    .add_local_dir(source_root, remote_path="/root/harness-src", copy=True)
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


def _load_control_outputs(prompts: list[dict]) -> dict[str, list[str]] | None:
    generated: dict[str, list[str]] = {}
    expected_ids = [str(row["fingerprint"]) for row in prompts]
    for arm in ("base_model", "initial_sft"):
        path = OUTPUT_ROOT / f"{arm}-128.jsonl"
        if not path.is_file():
            return None
        rows = _rows(path)
        if (
            len(rows) != len(prompts)
            or [str(row.get("prompt_id") or "") for row in rows] != expected_ids
            or any(
                int(row.get("sampling_seed", -1)) != _record_seed(prompt_id)
                or not str(row.get("generated_completion") or "").strip()
                for row, prompt_id in zip(rows, expected_ids)
            )
        ):
            raise RuntimeError(f"existing positive-control outputs are invalid: {arm}")
        generated[arm] = [str(row["generated_completion"]) for row in rows]
    return generated


def _write_control_outputs(
    prompts: list[dict], generated: dict[str, list[str]]
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for arm, texts in generated.items():
        path = OUTPUT_ROOT / f"{arm}-128.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "prompt_id": str(row["fingerprint"]),
                        "generated_completion": text,
                        "sampling_seed": _record_seed(str(row["fingerprint"])),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for row, text in zip(prompts, texts)
            ),
            encoding="utf-8",
        )


def _precomputed_mmd_pvalue(
    left: object,
    right: object,
    bandwidths: tuple[float, ...],
    *,
    seed: int,
    draws: int = 1999,
) -> float:
    import numpy as np
    from harness.measurement_v3 import fixed_rbf_kernel

    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    pooled = np.concatenate((x, y), axis=0)
    kernel = fixed_rbf_kernel(pooled, pooled, bandwidths)
    n, total = len(x), len(pooled)

    def statistic(left_indices: object) -> float:
        selected = np.asarray(left_indices, dtype=np.int64)
        mask = np.ones(total, dtype=bool)
        mask[selected] = False
        other = np.flatnonzero(mask)
        kxx = kernel[np.ix_(selected, selected)]
        kyy = kernel[np.ix_(other, other)]
        kxy = kernel[np.ix_(selected, other)]
        xx = (kxx.sum() - np.trace(kxx)) / (len(selected) * (len(selected) - 1))
        yy = (kyy.sum() - np.trace(kyy)) / (len(other) * (len(other) - 1))
        return float(xx + yy - 2.0 * kxy.mean())

    observed = statistic(np.arange(n))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(draws):
        exceedances += statistic(rng.permutation(total)[:n]) >= observed
    return float((exceedances + 1) / (draws + 1))


@app.function(
    image=image,
    gpu="L40S",
    timeout=90 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def materialize() -> dict:
    import gc
    import sys

    import numpy as np
    import torch
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/harness-src")
    from harness.measurement_v3 import (
        human_floor_bandwidths,
        token_unigram_l2,
    )

    checkpoint_volume.reload()
    if _sha(PROMPT_BRIEFS) != PROMPT_BRIEFS_SHA256:
        raise RuntimeError("normalized evaluation prompt hash mismatch")
    prompts = _rows(PROMPT_BRIEFS)
    if len(prompts) != 128:
        raise RuntimeError("positive controls require exactly 128 prompts")
    tokenizer = AutoTokenizer.from_pretrained(
        INITIAL_ADAPTER, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    generated = _load_control_outputs(prompts)
    if generated is None:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            cache_dir=str(CHECKPOINT_ROOT / "hf-cache"),
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base, INITIAL_ADAPTER, local_files_only=True, is_trainable=False
        )
        model.eval()
        generated = {"base_model": [], "initial_sft": []}
        with torch.inference_mode():
            for row in prompts:
                prompt_id = str(row["fingerprint"])
                rendered = _prompt(row)
                encoded = tokenizer(
                    rendered, return_tensors="pt", truncation=True, max_length=1024
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                for arm in ("base_model", "initial_sft"):
                    seed = _record_seed(prompt_id)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    context = model.disable_adapter() if arm == "base_model" else None
                    if context is None:
                        sequence = model.generate(
                            **encoded,
                            do_sample=True,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            max_new_tokens=64,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    else:
                        with context:
                            sequence = model.generate(
                                **encoded,
                                do_sample=True,
                                temperature=1.0,
                                top_p=1.0,
                                top_k=0,
                                max_new_tokens=64,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    continuation = sequence[0, encoded["input_ids"].shape[1] :]
                    generated[arm].append(
                        tokenizer.decode(
                            continuation, skip_special_tokens=True
                        ).strip()
                    )
        _write_control_outputs(prompts, generated)
        checkpoint_volume.commit()
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    role_rows = {
        role: _rows(PANEL_ROOT / f"{role}.jsonl")
        for role in ("distribution_references", "human_floor_a", "human_floor_b")
    }
    reference_texts = [str(row["completion"]) for row in role_rows["distribution_references"]]
    floor_a_texts = [str(row["completion"]) for row in role_rows["human_floor_a"]]
    floor_b_texts = [str(row["completion"]) for row in role_rows["human_floor_b"]]
    prefix_texts = []
    for text in reference_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:64]
        prefix_texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    token_sets = {
        "reference": [tokenizer.encode(text, add_special_tokens=False) for text in reference_texts],
        "floor_a": [tokenizer.encode(text, add_special_tokens=False) for text in floor_a_texts],
        "floor_b": [tokenizer.encode(text, add_special_tokens=False) for text in floor_b_texts],
        "prefix64": [tokenizer.encode(text, add_special_tokens=False) for text in prefix_texts],
        "base_model": [tokenizer.encode(text, add_special_tokens=False) for text in generated["base_model"]],
        "initial_sft": [tokenizer.encode(text, add_special_tokens=False) for text in generated["initial_sft"]],
    }
    pvalues = {name: {} for name in (
        "human_vs_human_null",
        "prefix64_vs_full_human",
        "sft_vs_unpaired_humans",
        "base_model_vs_unpaired_humans",
    )}
    for family_offset, (family_id, config) in enumerate(FAMILIES.items()):
        human_bundle = json.loads(
            (PANEL_ROOT / "human-embeddings" / f"{family_id}.json").read_text(encoding="utf-8")
        )
        human_vectors = {
            row["document_id"]: np.asarray(row["embedding"], dtype=np.float32)
            for row in human_bundle["rows"]
        }
        references = np.asarray(
            [human_vectors[row["fingerprint"]] for row in role_rows["distribution_references"]]
        )
        floor_a = np.asarray(
            [human_vectors[row["fingerprint"]] for row in role_rows["human_floor_a"]]
        )
        floor_b = np.asarray(
            [human_vectors[row["fingerprint"]] for row in role_rows["human_floor_b"]]
        )
        bandwidths = human_floor_bandwidths(floor_a, floor_b)
        embedder = SentenceTransformer(
            config["model_id"],
            revision=config["revision"],
            trust_remote_code=True,
            cache_folder=str(CHECKPOINT_ROOT / "hf-cache"),
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        embedder.max_seq_length = 512

        def encode(texts: list[str], active_embedder: object = embedder) -> np.ndarray:
            kwargs = {
                "batch_size": config["batch_size"],
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            }
            if config["prompt_name"]:
                kwargs["prompt_name"] = config["prompt_name"]
            values = np.asarray(
                active_embedder.encode(texts, **kwargs), dtype=np.float32
            )
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            return values / norms

        prefix = encode(prefix_texts)
        sft = encode(generated["initial_sft"])
        base_outputs = encode(generated["base_model"])
        comparisons = {
            "human_vs_human_null": (floor_a, floor_b),
            "prefix64_vs_full_human": (
                prefix,
                np.concatenate((floor_a, floor_b), axis=0),
            ),
            "sft_vs_unpaired_humans": (sft, references),
            "base_model_vs_unpaired_humans": (base_outputs, references),
        }
        for control_offset, (name, (left, right)) in enumerate(comparisons.items()):
            pvalues[name][family_id] = _precomputed_mmd_pvalue(
                left,
                right,
                bandwidths,
                seed=94001 + 100 * family_offset + control_offset,
            )
        del embedder
        gc.collect()
        torch.cuda.empty_cache()

    l2_values = {
        "human_vs_human_null": token_unigram_l2(token_sets["floor_a"], token_sets["floor_b"])["l2"],
        "prefix64_vs_full_human": token_unigram_l2(token_sets["prefix64"], token_sets["reference"])["l2"],
        "sft_vs_unpaired_humans": token_unigram_l2(token_sets["initial_sft"], token_sets["reference"])["l2"],
        "base_model_vs_unpaired_humans": token_unigram_l2(token_sets["base_model"], token_sets["reference"])["l2"],
    }
    expected = {
        "human_vs_human_null": "not_detected",
        "prefix64_vs_full_human": "detected",
        "sft_vs_unpaired_humans": "detected",
        "base_model_vs_unpaired_humans": "detected",
    }
    controls = {}
    for name, expectation in expected.items():
        detected = all(value <= ALPHA for value in pvalues[name].values())
        if detected is not (expectation == "detected"):
            raise RuntimeError(f"positive control did not qualify: {name} {pvalues[name]}")
        controls[name] = {
            "expected": expectation,
            "family_pvalues": pvalues[name],
            "token_unigram_l2": float(l2_values[name]),
            "detected": detected,
        }
    artifact = {
        "artifact_schema": "dftr.measurement.positive_controls.v3",
        "status": "qualified",
        "candidate_outputs_opened": False,
        "panel_design_sha256": PANEL_DESIGN_SHA256,
        "tokenization_contract_sha256": TOKENIZATION_CONTRACT_SHA256,
        "embedding_family_ids": sorted(FAMILIES),
        "alpha": ALPHA,
        "controls": controls,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_path = OUTPUT_ROOT / "positive_controls.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_volume.commit()
    return {
        "status": "qualified",
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha(artifact_path),
        "controls": controls,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(materialize.remote(), indent=2, sort_keys=True))
