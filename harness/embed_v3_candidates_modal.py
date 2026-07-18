"""Embed all measurement-v3 candidate arms with both frozen evaluator families."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


APP_NAME = "humanwrite-measurement-v3-candidate-embeddings"
CHECKPOINT_ROOT = Path("/checkpoints")
PANEL_ROOT = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/measurement-v3-panels"
PROTOCOL_PATH = PANEL_ROOT / "protocol-v1/measurement_protocol_v3.json"
OUTPUT_ROOT = PANEL_ROOT / "candidate-outputs-v1"
HUMAN_ROOT = PANEL_ROOT / "human-embeddings"
ARMS = ("SFT", "TOKEN_MOMENT", "MMD_WITNESS")
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
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.13.0",
    "transformers==4.57.6",
    "sentence-transformers==5.2.2",
    "accelerate>=1.8,<2",
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


@app.function(
    image=image,
    gpu="L40S",
    timeout=45 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def embed_family(family_id: str) -> dict:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    checkpoint_volume.reload()
    if family_id not in FAMILIES:
        raise ValueError(f"unknown family: {family_id}")
    protocol_sha = _sha(PROTOCOL_PATH)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    family_binding = next(
        (
            item
            for item in protocol.get("embedding_families", [])
            if item.get("family_id") == family_id
        ),
        None,
    )
    config = FAMILIES[family_id]
    if (
        family_binding is None
        or family_binding.get("model_id") != config["model_id"]
        or family_binding.get("model_revision") != config["revision"]
    ):
        raise RuntimeError(f"{family_id} does not match the frozen protocol")
    human_bundle = json.loads((HUMAN_ROOT / f"{family_id}.json").read_text())
    if (
        human_bundle.get("model_id") != config["model_id"]
        or human_bundle.get("model_revision") != config["revision"]
        or human_bundle.get("preprocessing_sha256")
        != family_binding.get("preprocessing_sha256")
    ):
        raise RuntimeError(f"{family_id} preprocessing binding mismatch")

    all_rows: list[dict] = []
    candidate_hashes: dict[str, str] = {}
    prompt_order: list[str] | None = None
    for arm in ARMS:
        output_path = OUTPUT_ROOT / f"{arm}.jsonl"
        manifest = json.loads(
            (OUTPUT_ROOT / f"{arm}.manifest.json").read_text(encoding="utf-8")
        )
        rows = _rows(output_path)
        ids = [str(row.get("prompt_id") or "") for row in rows]
        if (
            manifest.get("status") != "completed"
            or manifest.get("protocol_sha256") != protocol_sha
            or manifest.get("output_sha256") != _sha(output_path)
            or len(rows) != 128
            or len(set(ids)) != 128
        ):
            raise RuntimeError(f"{arm} candidate output binding mismatch")
        if prompt_order is None:
            prompt_order = ids
        elif ids != prompt_order:
            raise RuntimeError("candidate arms do not share exact prompt order")
        candidate_hashes[arm] = manifest["output_sha256"]
        all_rows.extend(rows)

    model = SentenceTransformer(
        config["model_id"],
        revision=config["revision"],
        trust_remote_code=True,
        cache_folder=str(CHECKPOINT_ROOT / "hf-cache"),
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = 512
    kwargs = {
        "batch_size": config["batch_size"],
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if config["prompt_name"]:
        kwargs["prompt_name"] = config["prompt_name"]
    vectors = np.asarray(
        model.encode(
            [str(row["generated_completion"]) for row in all_rows], **kwargs
        ),
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5
    ):
        raise RuntimeError(f"{family_id} produced invalid normalized vectors")
    artifact = {
        "artifact_schema": "dftr.measurement.candidate_embeddings.v3",
        "status": "completed",
        "family_id": family_id,
        "model_id": config["model_id"],
        "model_revision": config["revision"],
        "preprocessing_sha256": family_binding["preprocessing_sha256"],
        "protocol_sha256": protocol_sha,
        "candidate_output_sha256": candidate_hashes,
        "dimensions": int(vectors.shape[1]),
        "rows": [
            {
                "arm": str(row["arm"]),
                "prompt_id": str(row["prompt_id"]),
                "embedding": vector.tolist(),
            }
            for row, vector in zip(all_rows, vectors)
        ],
    }
    output_path = OUTPUT_ROOT / f"{family_id}.candidate-embeddings.json"
    output_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()
    return {
        "family_id": family_id,
        "output_path": str(output_path),
        "output_sha256": _sha(output_path),
        "rows": len(artifact["rows"]),
        "dimensions": artifact["dimensions"],
    }


@app.local_entrypoint()
def main() -> None:
    results = list(embed_family.map(FAMILIES))
    print(json.dumps(results, indent=2, sort_keys=True))
