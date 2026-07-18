"""Decision-grade automatic scoring for frozen measurement-v3 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from harness.measurement_v3 import (
    EmbeddingFamily,
    UnpairedPanelDesign,
    human_calibrated_equivalence,
    human_calibrated_noninferiority,
    token_unigram_l2,
    two_family_distribution_report,
)


ARMS = ("SFT", "TOKEN_MOMENT", "MMD_WITNESS")
TREATMENTS = ("TOKEN_MOMENT", "MMD_WITNESS")
FAMILY_IDS = ("bge-small-v1", "nemotron-8b-v1")
TRAINING_REWARD_MODEL_IDS = ("Qwen/Qwen3-4B",)
PERMUTATION_DRAWS = 9_999
BOOTSTRAP_DRAWS = 10_000
MASTER_SEED = 97001


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _binding(protocol_root: Path, protocol: dict, name: str) -> Path:
    binding = protocol["artifact_bindings"][name]
    path = protocol_root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise RuntimeError(f"protocol binding changed: {name}")
    return path


def _text_by_id(panel_root: Path, role: str) -> dict[str, str]:
    rows = _jsonl(panel_root / f"{role}.jsonl")
    values = {str(row["fingerprint"]): str(row["completion"]) for row in rows}
    if len(values) != len(rows) or any(
        hashlib.sha256(text.encode("utf-8")).hexdigest() != document_id
        for document_id, text in values.items()
    ):
        raise RuntimeError(f"{role} text identities changed")
    return values


WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _unexpected_non_latin(text: str) -> float:
    return float(
        any(
            char.isalpha() and "LATIN" not in unicodedata.name(char, "")
            for char in text
        )
    )


def _repeated_bigram_rate(text: str) -> float:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return float((len(bigrams) - len(set(bigrams))) / len(bigrams))


def _hard_validity(texts: list[str]) -> dict[str, Any]:
    unexpected = [_unexpected_non_latin(text) for text in texts]
    replacements = [float("�" in text) for text in texts]
    empty = [float(not text.strip()) for text in texts]
    unique_fraction = len(set(texts)) / len(texts)
    return {
        "documents": len(texts),
        "empty_rate": float(np.mean(empty)),
        "unexpected_non_latin_rate": float(np.mean(unexpected)),
        "replacement_character_rate": float(np.mean(replacements)),
        "unique_fraction": float(unique_fraction),
        "pass": bool(
            not any(empty)
            and not any(replacements)
            and float(np.mean(unexpected)) <= 0.15
            and unique_fraction >= 0.80
        ),
    }


def _candidate_rows(
    candidate_root: Path, protocol_sha: str, arms: tuple[str, ...] = ARMS
) -> dict[str, list[dict]]:
    result = {}
    for arm in arms:
        output_path = candidate_root / f"{arm}.jsonl"
        manifest = _json(candidate_root / f"{arm}.manifest.json")
        rows = _jsonl(output_path)
        if (
            manifest.get("artifact_schema")
            != "dftr.measurement.candidate_outputs.v3"
            or manifest.get("status") != "completed"
            or manifest.get("arm") != arm
            or manifest.get("protocol_sha256") != protocol_sha
            or manifest.get("output_sha256") != _sha(output_path)
            or len(rows) != 128
        ):
            raise RuntimeError(f"candidate binding failed for {arm}")
        mapping = {str(row["prompt_id"]): row for row in rows}
        if len(mapping) != 128:
            raise RuntimeError(f"candidate prompt identities failed for {arm}")
        result[arm] = rows
    return result


def _candidate_embeddings(
    candidate_root: Path,
    protocol_sha: str,
    arms: tuple[str, ...] = ARMS,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for family_id in FAMILY_IDS:
        value = _json(candidate_root / f"{family_id}.candidate-embeddings.json")
        if (
            value.get("artifact_schema")
            != "dftr.measurement.candidate_embeddings.v3"
            or value.get("status") != "completed"
            or value.get("family_id") != family_id
            or value.get("protocol_sha256") != protocol_sha
            or len(value.get("rows", [])) != 128 * len(arms)
        ):
            raise RuntimeError(f"candidate embedding binding failed for {family_id}")
        family: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in arms}
        for row in value["rows"]:
            family[str(row["arm"])][str(row["prompt_id"])] = np.asarray(
                row["embedding"], dtype=np.float64
            )
        if any(len(family[arm]) != 128 for arm in arms):
            raise RuntimeError(f"candidate embedding cardinality failed for {family_id}")
        result[family_id] = family
    return result


def score(
    *,
    panel_root: Path,
    protocol_root: Path,
    candidate_root: Path,
    tokenizer_path: Path,
    arms: tuple[str, ...] = ARMS,
) -> dict[str, Any]:
    if "SFT" not in arms or len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError("arms require unique SFT control plus at least one treatment")
    treatments = tuple(arm for arm in arms if arm != "SFT")
    protocol_path = protocol_root / "measurement_protocol_v3.json"
    protocol = _json(protocol_path)
    protocol_sha = _sha(protocol_path)
    if (
        protocol.get("artifact_schema") != "dftr.measurement.protocol.v3"
        or protocol.get("status") != "ready"
        or protocol.get("frozen") is not True
        or protocol.get("candidate_outputs_opened") is not False
    ):
        raise RuntimeError("measurement-v3 protocol is not a valid candidate-blind freeze")
    decision_config = _json(_binding(protocol_root, protocol, "decision_power_config"))
    rule = decision_config["decision_rule"]
    controls = _json(_binding(protocol_root, protocol, "positive_controls"))
    positive_control_pass = controls.get("status") == "qualified"
    manifests = {
        "prompt_sources": _json(_binding(protocol_root, protocol, "prompt_manifest")),
        "distribution_references": _json(
            _binding(protocol_root, protocol, "semantic_reference_manifest")
        ),
        "human_floor_a": _json(_binding(protocol_root, protocol, "floor_a_manifest")),
        "human_floor_b": _json(_binding(protocol_root, protocol, "floor_b_manifest")),
    }
    design = UnpairedPanelDesign.build(
        prompt_ids=[row["prompt_id"] for row in manifests["prompt_sources"]["records"]],
        prompt_source_document_ids=[
            row["source_document_id"]
            for row in manifests["prompt_sources"]["records"]
        ],
        distribution_reference_ids=[
            row["document_id"]
            for row in manifests["distribution_references"]["records"]
        ],
        human_floor_a_ids=[
            row["document_id"] for row in manifests["human_floor_a"]["records"]
        ],
        human_floor_b_ids=[
            row["document_id"] for row in manifests["human_floor_b"]["records"]
        ],
    )
    candidates = _candidate_rows(candidate_root, protocol_sha, arms)
    candidate_maps = {
        arm: {str(row["prompt_id"]): row for row in rows}
        for arm, rows in candidates.items()
    }
    if any(set(mapping) != set(design.prompt_ids) for mapping in candidate_maps.values()):
        raise RuntimeError("candidate outputs do not match the frozen prompt identities")
    candidate_vectors = _candidate_embeddings(candidate_root, protocol_sha, arms)

    human_text = {
        role: _text_by_id(panel_root, role)
        for role in ("distribution_references", "human_floor_a", "human_floor_b")
    }
    human_embeddings: dict[str, dict[str, np.ndarray]] = {}
    family_metadata = {}
    for index, family_id in enumerate(FAMILY_IDS):
        binding_name = "embedding_family_a" if index == 0 else "embedding_family_b"
        bundle = _json(_binding(protocol_root, protocol, binding_name))
        if bundle.get("family_id") != family_id:
            raise RuntimeError("embedding family order changed")
        human_embeddings[family_id] = {
            str(row["document_id"]): np.asarray(row["embedding"], dtype=np.float64)
            for row in bundle["rows"]
        }
        family_metadata[family_id] = bundle

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True
    )

    def tokens(texts: list[str]) -> list[list[int]]:
        return [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    reference_texts = [
        human_text["distribution_references"][document_id]
        for document_id in design.distribution_reference_ids
    ]
    human_safety_texts = [
        human_text[role][document_id]
        for role, ids in (
            ("human_floor_a", design.human_floor_a_ids),
            ("human_floor_b", design.human_floor_b_ids),
        )
        for document_id in ids
    ]
    arm_texts = {
        arm: [
            str(candidate_maps[arm][prompt_id]["generated_completion"])
            for prompt_id in design.prompt_ids
        ]
        for arm in arms
    }
    token_l2 = {
        arm: token_unigram_l2(tokens(texts), tokens(reference_texts))
        for arm, texts in arm_texts.items()
    }
    validity = {arm: _hard_validity(texts) for arm, texts in arm_texts.items()}
    output_identity = {
        treatment: float(
            np.mean(
                [
                    left.encode("utf-8") == right.encode("utf-8")
                    for left, right in zip(arm_texts[treatment], arm_texts["SFT"])
                ]
            )
        )
        for treatment in treatments
    }

    comparisons = {}
    for treatment_index, treatment in enumerate(treatments):
        families = []
        for family_id in FAMILY_IDS:
            metadata = family_metadata[family_id]
            vectors = human_embeddings[family_id]
            families.append(
                EmbeddingFamily.build(
                    design,
                    family_id=family_id,
                    model_id=metadata["model_id"],
                    model_revision=metadata["model_revision"],
                    treatment=np.asarray(
                        [
                            candidate_vectors[family_id][treatment][prompt_id]
                            for prompt_id in design.prompt_ids
                        ]
                    ),
                    control=np.asarray(
                        [
                            candidate_vectors[family_id]["SFT"][prompt_id]
                            for prompt_id in design.prompt_ids
                        ]
                    ),
                    distribution_reference=np.asarray(
                        [vectors[item] for item in design.distribution_reference_ids]
                    ),
                    human_floor_a=np.asarray(
                        [vectors[item] for item in design.human_floor_a_ids]
                    ),
                    human_floor_b=np.asarray(
                        [vectors[item] for item in design.human_floor_b_ids]
                    ),
                )
            )
        distribution = two_family_distribution_report(
            design,
            families,
            training_reward_model_ids=TRAINING_REWARD_MODEL_IDS,
            permutation_draws=PERMUTATION_DRAWS,
            seed=MASTER_SEED + 100 * treatment_index,
        )
        non_latin = human_calibrated_equivalence(
            [_unexpected_non_latin(text) for text in arm_texts[treatment]],
            [_unexpected_non_latin(text) for text in human_safety_texts],
            margin=float(rule["equivalence_margin"]),
            draws=BOOTSTRAP_DRAWS,
            seed=MASTER_SEED + 100 * treatment_index + 31,
        )
        repetition = human_calibrated_noninferiority(
            [_repeated_bigram_rate(text) for text in arm_texts[treatment]],
            [_repeated_bigram_rate(text) for text in human_safety_texts],
            margin=float(rule["noninferiority_margin"]),
            lower_is_better=True,
            draws=BOOTSTRAP_DRAWS,
            seed=MASTER_SEED + 100 * treatment_index + 32,
        )
        family_passes = {
            family_id: bool(
                distribution["families"][family_id]["treatment_minus_control"]
                <= float(rule["family_effect_boundary"])
                and distribution["families"][family_id]["paired_treatment_test"][
                    "pvalue"
                ]
                <= float(rule["alpha"])
            )
            for family_id in FAMILY_IDS
        }
        token_difference = token_l2[treatment]["l2"] - token_l2["SFT"]["l2"]
        endpoint_passes = {
            "embedding_family_a_paired_mmd": family_passes[FAMILY_IDS[0]],
            "embedding_family_b_paired_mmd": family_passes[FAMILY_IDS[1]],
            "embedding_family_direction_agreement": distribution[
                "primary_direction_agreement"
            ],
            "token_unigram_l2": token_difference <= float(rule["token_l2_margin"]),
            "human_calibrated_equivalence": non_latin["decision"] == "pass",
            "human_calibrated_noninferiority": repetition["decision"] == "pass",
            "positive_control_qualification": positive_control_pass,
        }
        comparisons[treatment] = {
            "distribution": distribution,
            "token_unigram_l2": {
                "treatment": token_l2[treatment],
                "control": token_l2["SFT"],
                "treatment_minus_control": token_difference,
                "noninferiority_margin": rule["token_l2_margin"],
            },
            "unexpected_non_latin_equivalence": non_latin,
            "repeated_bigram_noninferiority": repetition,
            "byte_identity_with_control": output_identity[treatment],
            "endpoint_passes": endpoint_passes,
            "automatic_promotion_pass": bool(
                all(endpoint_passes.values())
                and validity[treatment]["pass"]
                and output_identity[treatment] < 0.60
            ),
        }

    return {
        "artifact_schema": "dftr.measurement.automatic_decision.v3",
        "status": "completed",
        "protocol_sha256": protocol_sha,
        "panel_design_sha256": design.identity_sha256,
        "decision_rule": rule,
        "permutation_draws": PERMUTATION_DRAWS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "token_unigram_l2_by_arm": token_l2,
        "hard_validity_by_arm": validity,
        "comparisons": comparisons,
        "automatic_winner": next(
            (
                treatment
                for treatment in treatments
                if comparisons[treatment]["automatic_promotion_pass"]
            ),
            None,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    result = score(
        panel_root=args.panel_root,
        protocol_root=args.protocol_root,
        candidate_root=args.candidate_root,
        tokenizer_path=args.tokenizer_path,
        arms=arms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": _sha(args.output),
                "automatic_winner": result["automatic_winner"],
                "promotion": {
                    name: value["automatic_promotion_pass"]
                    for name, value in result["comparisons"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
