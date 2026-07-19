"""Embed the frozen 4K scale-ladder panel and candidates with two evaluators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_LOCAL = ROOT / "configs/m2/m2_scale_ladder_4k_scoring_contract_v1.json"
CONTRACT_REMOTE = Path("/root/m2_scale_ladder_4k_scoring_contract_v1.json")
CHECKPOINT_ROOT = Path("/checkpoints")
OUTPUT_ROOT = CHECKPOINT_ROOT / "data/m2-scale-ladder-v1/evaluation-4k-v1/embeddings"
APP_NAME = "humanwrite-scale-ladder-4k-embeddings"

checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0",
        "transformers==4.57.6",
        "sentence-transformers==5.2.2",
        "accelerate>=1.8,<2",
        "huggingface-hub>=0.33,<1",
    )
    .add_local_file(str(CONTRACT_LOCAL), str(CONTRACT_REMOTE))
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
    timeout=90 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def embed_family(family_id: str) -> dict:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    checkpoint_volume.reload()
    contract = json.loads(CONTRACT_REMOTE.read_text(encoding="utf-8"))
    contract_sha = _sha(CONTRACT_REMOTE)
    if family_id not in contract["embedding_families"]:
        raise ValueError(f"unknown embedding family: {family_id}")
    artifacts = contract["artifacts"]
    ordered_roles = (
        "SFT",
        "MMD_WITNESS",
        "distribution_references",
        "human_floor_a",
        "human_floor_b",
    )
    role_rows: dict[str, list[dict]] = {}
    for role in ordered_roles:
        binding = artifacts[role]
        path = Path(binding["path"])
        rows = _rows(path)
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha(path) != binding["sha256"]
            or len(rows) != binding["rows"]
        ):
            raise RuntimeError(f"artifact binding failed: {role}")
        role_rows[role] = rows
    candidate_ids = {
        role: [str(row["prompt_id"]) for row in role_rows[role]]
        for role in ("SFT", "MMD_WITNESS")
    }
    if candidate_ids["SFT"] != candidate_ids["MMD_WITNESS"] or len(
        set(candidate_ids["SFT"])
    ) != 128:
        raise RuntimeError("candidate prompt pairing failed")

    config = contract["embedding_families"][family_id]
    model = SentenceTransformer(
        config["model_id"],
        revision=config["revision"],
        trust_remote_code=True,
        cache_folder=str(CHECKPOINT_ROOT / "hf-cache"),
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = int(config["max_tokens"])
    flat: list[tuple[str, str, str]] = []
    for role in ordered_roles:
        id_field = "prompt_id" if role in {"SFT", "MMD_WITNESS"} else "fingerprint"
        text_field = "text" if role in {"SFT", "MMD_WITNESS"} else "completion"
        flat.extend(
            (role, str(row[id_field]), str(row[text_field])) for row in role_rows[role]
        )
    kwargs = {
        "batch_size": 32 if family_id == "bge-small-v1" else 4,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if config["prompt_name"] is not None:
        kwargs["prompt_name"] = config["prompt_name"]
    vectors = np.asarray(model.encode([item[2] for item in flat], **kwargs), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or bool((norms <= 0).any()):
        raise RuntimeError("embedding output is invalid")
    vectors = vectors / norms
    artifact = {
        "artifact_schema": "dftr.m2.scale_ladder_embeddings.v1",
        "status": "completed",
        "contract_sha256": contract_sha,
        "family_id": family_id,
        "model_id": config["model_id"],
        "model_revision": config["revision"],
        "preprocessing": {
            "implementation": "sentence-transformers.encode.v1",
            "max_tokens": config["max_tokens"],
            "prompt_name": config["prompt_name"],
            "normalize_embeddings": True,
            "output_dtype": "float32",
        },
        "artifact_sha256": {role: artifacts[role]["sha256"] for role in ordered_roles},
        "rows": [
            {"role": role, "document_id": document_id, "embedding": vector.tolist()}
            for (role, document_id, _text), vector in zip(flat, vectors)
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"{family_id}.json"
    output.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()
    return {
        "family_id": family_id,
        "rows": len(flat),
        "dimensions": int(vectors.shape[1]),
        "output_path": str(output),
        "output_sha256": _sha(output),
        "contract_sha256": contract_sha,
    }


@app.local_entrypoint()
def main() -> None:
    contract = json.loads(CONTRACT_LOCAL.read_text(encoding="utf-8"))
    calls = {
        family: embed_family.spawn(family) for family in contract["embedding_families"]
    }
    print(json.dumps({family: call.get() for family, call in calls.items()}, indent=2))
