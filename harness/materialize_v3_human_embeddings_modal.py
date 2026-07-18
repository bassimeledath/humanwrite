"""Materialize the two candidate-blind human embedding families for measurement v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


APP_NAME = "humanwrite-measurement-v3-human-embeddings"
CHECKPOINT_ROOT = Path("/checkpoints")
PANEL_ROOT = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/measurement-v3-panels"
OUTPUT_ROOT = PANEL_ROOT / "human-embeddings"

FAMILIES = {
    "bge-small-v1": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "batch_size": 32,
        "max_tokens": 512,
        "prompt_name": None,
    },
    "nemotron-8b-v1": {
        "model_id": "nvidia/llama-embed-nemotron-8b",
        "revision": "aa3b43a495a9b280d1bdb716da37c54bb495d630",
        "batch_size": 4,
        "max_tokens": 512,
        "prompt_name": "passage",
    },
}


checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.13.0",
    "transformers==4.57.6",
    "sentence-transformers==5.2.2",
    "accelerate>=1.8,<2",
    "huggingface-hub>=0.33,<1",
)
app = modal.App(APP_NAME)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@app.function(
    image=image,
    gpu="L40S",
    timeout=90 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def embed_family(family_id: str) -> dict:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    checkpoint_volume.reload()
    if family_id not in FAMILIES:
        raise ValueError(f"unknown embedding family: {family_id}")
    config = FAMILIES[family_id]
    role_paths = [
        PANEL_ROOT / "distribution_references.jsonl",
        PANEL_ROOT / "human_floor_a.jsonl",
        PANEL_ROOT / "human_floor_b.jsonl",
    ]
    rows = [row for path in role_paths for row in _read_jsonl(path)]
    if len(rows) != 512:
        raise RuntimeError(f"expected 512 human documents, found {len(rows)}")
    document_ids = [str(row["fingerprint"]) for row in rows]
    if len(set(document_ids)) != 512:
        raise RuntimeError("human embedding identities are not unique")
    preprocessing = {
        "implementation": "sentence-transformers.encode.v1",
        "input": "utf8_verbatim_completion",
        "max_tokens": config["max_tokens"],
        "prompt_name": config["prompt_name"],
        "normalize_embeddings": True,
        "post_encode_normalization": "explicit_float32_l2.v1",
        "output_dtype": "float32",
    }
    model = SentenceTransformer(
        config["model_id"],
        revision=config["revision"],
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = int(config["max_tokens"])
    encode_kwargs = {
        "batch_size": int(config["batch_size"]),
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    if config["prompt_name"] is not None:
        encode_kwargs["prompt_name"] = config["prompt_name"]
    vectors = model.encode([str(row["completion"]) for row in rows], **encode_kwargs)
    vectors = np.asarray(vectors, dtype=np.float32)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != 512
        or vectors.shape[1] < 1
        or not np.isfinite(vectors).all()
    ):
        raise RuntimeError("embedding family returned an invalid matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or bool((norms <= 0).any()):
        raise RuntimeError("embedding family returned a zero or invalid vector")
    vectors = vectors / norms
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("explicit embedding normalization failed")
    artifact = {
        "artifact_schema": "dftr.measurement.embedding_family.v3",
        "status": "materialized",
        "family_id": family_id,
        "model_id": config["model_id"],
        "model_revision": config["revision"],
        "model_artifact_sha256": _canonical_hash(
            {"repo_id": config["model_id"], "revision": config["revision"]}
        ),
        "preprocessing_sha256": _canonical_hash(preprocessing),
        "rows": [
            {"document_id": document_id, "embedding": vector.tolist()}
            for document_id, vector in zip(document_ids, vectors)
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{family_id}.json"
    payload = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    checkpoint_volume.commit()
    return {
        "family_id": family_id,
        "model_id": config["model_id"],
        "rows": len(rows),
        "dimension": int(vectors.shape[1]),
        "output_path": str(output_path),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "preprocessing": preprocessing,
    }


@app.local_entrypoint()
def main() -> None:
    calls = {family: embed_family.spawn(family) for family in FAMILIES}
    results = {family: call.get() for family, call in calls.items()}
    print(json.dumps(results, indent=2, sort_keys=True))
