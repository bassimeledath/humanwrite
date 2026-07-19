"""Two-family, candidate-independent embedding MMD score for M3 rewriting."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


SCHEMA = "humanwrite.m3.rewrite_embedding_score.v1"
STEP = "score_m3_rewrite_embeddings"
FAMILIES = {
    "bge-small-v1": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "prompt_name": None,
        "batch_size": 32,
        "max_tokens": 512,
    },
    "nemotron-8b-v1": {
        "model_id": "nvidia/llama-embed-nemotron-8b",
        "revision": "aa3b43a495a9b280d1bdb716da37c54bb495d630",
        "prompt_name": "document",
        "batch_size": 4,
        "max_tokens": 512,
    },
}
ROOT = "/checkpoints/data/m3-rewriting-14b-v1"


class M3EmbeddingScoreError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(
    *, clean_pool_sha256: str, panel_sha256: str, sft_sha256: str, treatment_sha256: str
) -> dict[str, Any]:
    values = (clean_pool_sha256, panel_sha256, sft_sha256, treatment_sha256)
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
        raise M3EmbeddingScoreError("input hashes must be lowercase SHA-256")
    config = {
        "artifact_schema": SCHEMA,
        "run": {
            "comparison_id": "M3-rewriting-14b-4k-embedding-score-v1",
            "arm": "HUMANWRITE14-vs-SFT14-two-family-MMD",
            "budget_class": "screen",
            "task_kind": "experiment",
            "command": ["python", "-m", "experiments.runner"],
            "seed": 6201,
        },
        "compute": {"gpu": "L40S", "gpus": 1, "timeout_min": 120},
        "data": {
            "clean_pool_path": f"{ROOT}/fresh-eval-clean-pool-640.jsonl",
            "clean_pool_sha256": clean_pool_sha256,
            "clean_pool_records": 640,
            "panel_path": f"{ROOT}/fresh-rewrite-eval-panel-256-v1.jsonl",
            "panel_sha256": panel_sha256,
            "panel_records": 256,
            "sft_path": f"{ROOT}/evaluation/sft14-outputs-256-v1.jsonl",
            "sft_sha256": sft_sha256,
            "treatment_path": f"{ROOT}/evaluation/humanwrite14-outputs-256-v1.jsonl",
            "treatment_sha256": treatment_sha256,
        },
        "representation": {
            "families": FAMILIES,
            "normalization": "explicit_float32_l2.v1",
            "bandwidth_source": "two_disjoint_128-document_unused-human-floors",
            "bandwidth_scales": [0.25, 0.5, 1.0, 2.0, 4.0],
        },
        "output": {
            "report_filename": "rewrite_embedding_score.json",
            "embedding_filename_template": "{family_id}.npz",
        },
        "workflow": {"protocol_version": SCHEMA, "step": STEP},
    }
    return config


def _load(path: Path, sha: str, records: int) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or file_sha256(path) != sha:
        raise M3EmbeddingScoreError(f"artifact binding failed: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != records:
        raise M3EmbeddingScoreError(f"artifact cardinality failed: {path}")
    return rows


def human_floor_rows(clean_pool: list[dict], panel: list[dict]) -> tuple[list[dict], list[dict]]:
    used = {str(row["fingerprint"]) for row in panel}
    unused = [row for row in clean_pool if str(row["fingerprint"]) not in used]
    unused.sort(key=lambda row: hashlib.sha256(f"m3-floor:{row['fingerprint']}".encode()).hexdigest())
    if len(unused) != 384 or len({row["fingerprint"] for row in unused}) != 384:
        raise M3EmbeddingScoreError("unused human floor pool mismatch")
    return unused[:128], unused[128:256]


def bandwidths(floor_a: np.ndarray, floor_b: np.ndarray) -> list[float]:
    pooled = np.concatenate((floor_a, floor_b), axis=0).astype(np.float64)
    squared = ((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(axis=2)
    positive = squared[np.triu_indices(len(pooled), 1)]
    positive = positive[positive > 0]
    if not len(positive):
        raise M3EmbeddingScoreError("human floor bandwidth is degenerate")
    median = float(np.median(positive))
    return [median * value for value in (0.25, 0.5, 1.0, 2.0, 4.0)]


def _kernel(left: np.ndarray, right: np.ndarray, scales: list[float]) -> np.ndarray:
    squared = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2)
    return np.mean([np.exp(-squared / (2.0 * scale)) for scale in scales], axis=0)


def mmd2_unbiased(left: np.ndarray, right: np.ndarray, scales: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        raise M3EmbeddingScoreError("MMD requires at least two rows")
    xx, yy, xy = _kernel(left, left, scales), _kernel(right, right, scales), _kernel(left, right, scales)
    return float(
        (xx.sum() - np.trace(xx)) / (len(left) * (len(left) - 1))
        + (yy.sum() - np.trace(yy)) / (len(right) * (len(right) - 1))
        - 2.0 * xy.mean()
    )


def run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    data = config["data"]
    clean = _load(Path(data["clean_pool_path"]), data["clean_pool_sha256"], 640)
    panel = _load(Path(data["panel_path"]), data["panel_sha256"], 256)
    sft = _load(Path(data["sft_path"]), data["sft_sha256"], 256)
    treatment = _load(Path(data["treatment_path"]), data["treatment_sha256"], 256)
    panel_ids = [str(row["fingerprint"]) for row in panel]
    arm_maps = {
        "SFT14": {str(row["fingerprint"]): str(row["output"]) for row in sft},
        "HUMANWRITE14": {str(row["fingerprint"]): str(row["output"]) for row in treatment},
    }
    if any(set(values) != set(panel_ids) for values in arm_maps.values()):
        raise M3EmbeddingScoreError("candidate alignment failed")
    floor_a, floor_b = human_floor_rows(clean, panel)
    roles = {
        "SFT14": [arm_maps["SFT14"][key] for key in panel_ids],
        "HUMANWRITE14": [arm_maps["HUMANWRITE14"][key] for key in panel_ids],
        "HUMAN_REFERENCE": [str(row["human_reference"]) for row in panel],
        "HUMAN_FLOOR_A": [str(row["completion"]) for row in floor_a],
        "HUMAN_FLOOR_B": [str(row["completion"]) for row in floor_b],
    }
    output_dir = Path(os.environ["DFTR_CHECKPOINT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"artifact_schema": SCHEMA, "run_id": run_id, "families": {}}
    for family_id, spec in FAMILIES.items():
        model = SentenceTransformer(
            spec["model_id"], revision=spec["revision"], trust_remote_code=True,
            cache_folder="/checkpoints/hf-cache", model_kwargs={"torch_dtype": torch.bfloat16},
        )
        model.max_seq_length = spec["max_tokens"]
        flat = [(role, index, text) for role, texts in roles.items() for index, text in enumerate(texts)]
        kwargs = dict(batch_size=spec["batch_size"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if spec["prompt_name"] is not None:
            kwargs["prompt_name"] = spec["prompt_name"]
        vectors = np.asarray(model.encode([row[2] for row in flat], **kwargs), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        by_role: dict[str, np.ndarray] = {}
        start = 0
        for role, texts in roles.items():
            by_role[role] = vectors[start : start + len(texts)]
            start += len(texts)
        scales = bandwidths(by_role["HUMAN_FLOOR_A"], by_role["HUMAN_FLOOR_B"])
        sft_mmd = mmd2_unbiased(by_role["SFT14"], by_role["HUMAN_REFERENCE"], scales)
        treatment_mmd = mmd2_unbiased(by_role["HUMANWRITE14"], by_role["HUMAN_REFERENCE"], scales)
        embedding_path = output_dir / f"{family_id}.npz"
        np.savez_compressed(embedding_path, **by_role)
        report["families"][family_id] = {
            "model_id": spec["model_id"], "revision": spec["revision"],
            "bandwidths": scales, "sft_mmd2": sft_mmd,
            "treatment_mmd2": treatment_mmd,
            "treatment_minus_sft": treatment_mmd - sft_mmd,
            "embedding_path": str(embedding_path), "embedding_sha256": file_sha256(embedding_path),
        }
        del model
        torch.cuda.empty_cache()
    report["input_hashes"] = {key: data[key] for key in data if key.endswith("sha256")}
    report_path = output_dir / config["output"]["report_filename"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["report_path"] = str(report_path)
    report["report_sha256"] = file_sha256(report_path)
    return report


__all__ = ["FAMILIES", "M3EmbeddingScoreError", "SCHEMA", "STEP", "bandwidths", "build_config", "human_floor_rows", "mmd2_unbiased", "run"]
