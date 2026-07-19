"""Score the frozen 4K SFT/MMD-witness outputs without training-model reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from harness.measurement_v3 import (
    EmbeddingFamily,
    UnpairedPanelDesign,
    token_unigram_l2,
    two_family_distribution_report,
)


FAMILIES = ("bge-small-v1", "nemotron-8b-v1")
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _unexpected_non_latin(text: str) -> float:
    return float(
        any(char.isalpha() and "LATIN" not in unicodedata.name(char, "") for char in text)
    )


def _repeated_bigram_rate(text: str) -> float:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    bigrams = list(zip(words, words[1:]))
    return 0.0 if not bigrams else (len(bigrams) - len(set(bigrams))) / len(bigrams)


def hard_validity(texts: list[str], rule: dict[str, float]) -> dict[str, Any]:
    values = {
        "empty_rate": float(np.mean([not text.strip() for text in texts])),
        "replacement_character_rate": float(np.mean(["�" in text for text in texts])),
        "unexpected_non_latin_rate": float(np.mean([_unexpected_non_latin(text) for text in texts])),
        "unique_fraction": float(len(set(texts)) / len(texts)),
    }
    values["pass"] = bool(
        values["empty_rate"] <= rule["empty_rate_max"]
        and values["replacement_character_rate"] <= rule["replacement_character_rate_max"]
        and values["unexpected_non_latin_rate"] <= rule["unexpected_non_latin_rate_max"]
        and values["unique_fraction"] >= rule["unique_fraction_min"]
    )
    return values


def binomial_lower_tail(wins: int, trials: int) -> float:
    if not 0 <= wins <= trials or trials < 1:
        raise ValueError("invalid binomial counts")
    return float(sum(math.comb(trials, k) for k in range(wins + 1)) / (2**trials))


def validate_contract(contract: dict[str, Any]) -> None:
    if (
        contract.get("artifact_schema") != "dftr.m2.scale_ladder_scoring_contract.v1"
        or contract.get("status") != "frozen_before_metric_computation"
        or set(contract.get("embedding_families") or {}) != set(FAMILIES)
        or (contract.get("decision") or {}).get("four_k_role") != "diagnostic_gate"
        or (contract.get("judge") or {}).get("model") != "openai/gpt-5.4-mini"
    ):
        raise ValueError("invalid frozen 4K scoring contract")


def score(
    *,
    contract_path: Path,
    panel_root: Path,
    sft_output: Path,
    mmd_output: Path,
    embedding_paths: dict[str, Path],
    judge_results: Path,
    judge_manifest: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    validate_contract(contract)
    contract_sha = _sha(contract_path)
    paths = {
        "prompt_briefs": panel_root / "prompt_briefs-128.jsonl",
        "distribution_references": panel_root / "distribution_references.jsonl",
        "human_floor_a": panel_root / "human_floor_a.jsonl",
        "human_floor_b": panel_root / "human_floor_b.jsonl",
        "SFT": sft_output,
        "MMD_WITNESS": mmd_output,
    }
    loaded: dict[str, list[dict]] = {}
    for role, path in paths.items():
        binding = contract["artifacts"][role]
        rows = _rows(path)
        if _sha(path) != binding["sha256"] or len(rows) != binding["rows"]:
            raise RuntimeError(f"scoring artifact binding failed: {role}")
        loaded[role] = rows

    prompt_ids = sorted(str(row.get("prompt_id") or row["fingerprint"]) for row in loaded["prompt_briefs"])
    source_by_prompt = {
        str(row.get("prompt_id") or row["fingerprint"]): str(row["source_fingerprint"])
        for row in loaded["prompt_briefs"]
    }
    candidate_maps = {
        arm: {str(row["prompt_id"]): str(row["text"]) for row in loaded[arm]}
        for arm in ("SFT", "MMD_WITNESS")
    }
    if len(prompt_ids) != 128 or any(set(mapping) != set(prompt_ids) for mapping in candidate_maps.values()):
        raise RuntimeError("candidate prompt identities do not match the frozen briefs")
    human_ids = {
        role: [str(row["fingerprint"]) for row in loaded[role]]
        for role in ("distribution_references", "human_floor_a", "human_floor_b")
    }
    design = UnpairedPanelDesign.build(
        prompt_ids=prompt_ids,
        prompt_source_document_ids=[source_by_prompt[item] for item in prompt_ids],
        distribution_reference_ids=human_ids["distribution_references"],
        human_floor_a_ids=human_ids["human_floor_a"],
        human_floor_b_ids=human_ids["human_floor_b"],
    )

    families = []
    for family_id in FAMILIES:
        bundle = json.loads(embedding_paths[family_id].read_text())
        config = contract["embedding_families"][family_id]
        accepted_contract_hashes = {contract_sha}
        transport_repair = contract.get("transport_repair") or {}
        if transport_repair.get("embedding_outputs_reusable") is True:
            accepted_contract_hashes.add(str(transport_repair.get("prior_contract_sha256")))
        if (
            bundle.get("artifact_schema") != "dftr.m2.scale_ladder_embeddings.v1"
            or bundle.get("status") != "completed"
            or bundle.get("contract_sha256") not in accepted_contract_hashes
            or bundle.get("family_id") != family_id
            or bundle.get("model_id") != config["model_id"]
            or bundle.get("model_revision") != config["revision"]
        ):
            raise RuntimeError(f"embedding binding failed: {family_id}")
        vectors: dict[str, dict[str, np.ndarray]] = {}
        for row in bundle["rows"]:
            vectors.setdefault(str(row["role"]), {})[str(row["document_id"])] = np.asarray(
                row["embedding"], dtype=np.float64
            )
        families.append(
            EmbeddingFamily.build(
                design,
                family_id=family_id,
                model_id=config["model_id"],
                model_revision=config["revision"],
                treatment=[vectors["MMD_WITNESS"][item] for item in prompt_ids],
                control=[vectors["SFT"][item] for item in prompt_ids],
                distribution_reference=[vectors["distribution_references"][item] for item in human_ids["distribution_references"]],
                human_floor_a=[vectors["human_floor_a"][item] for item in human_ids["human_floor_a"]],
                human_floor_b=[vectors["human_floor_b"][item] for item in human_ids["human_floor_b"]],
            )
        )
    statistics = contract["statistics"]
    distribution = two_family_distribution_report(
        design,
        families,
        training_reward_model_ids=("Qwen/Qwen3-4B",),
        permutation_draws=int(statistics["permutation_draws"]),
        seed=int(statistics["master_seed"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    reference_texts = [str(row["completion"]) for row in loaded["distribution_references"]]
    arm_texts = {arm: [candidate_maps[arm][item] for item in prompt_ids] for arm in candidate_maps}
    encode = lambda texts: [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    token_l2 = {arm: token_unigram_l2(encode(texts), encode(reference_texts)) for arm, texts in arm_texts.items()}
    validity = {arm: hard_validity(texts, contract["validity"]) for arm, texts in arm_texts.items()}
    identity = float(np.mean([left == right for left, right in zip(arm_texts["MMD_WITNESS"], arm_texts["SFT"])]))
    repetition = {
        arm: float(np.mean([_repeated_bigram_rate(text) for text in texts]))
        for arm, texts in arm_texts.items()
    }

    judge_binding = json.loads(judge_manifest.read_text())
    judge_rows = _rows(judge_results)
    if (
        judge_binding.get("artifact_schema") != "dftr.m2.scale_ladder_judge_results.v1"
        or judge_binding.get("status") != "completed"
        or judge_binding.get("contract_sha256") != contract_sha
        or judge_binding.get("output_sha256") != _sha(judge_results)
        or len(judge_rows) != 256
        or len({(row["dimension"], row["prompt_id"]) for row in judge_rows}) != 256
    ):
        raise RuntimeError("judge result binding failed")
    judge = {}
    for dimension in contract["judge"]["dimensions"]:
        rows = [row for row in judge_rows if row["dimension"] == dimension]
        wins = sum(bool(row["treatment_win"]) for row in rows)
        judge[dimension] = {
            "wins": wins,
            "trials": len(rows),
            "win_rate": wins / len(rows),
            "pvalue_treatment_worse": binomial_lower_tail(wins, len(rows)),
        }

    family_effects = {
        family_id: distribution["families"][family_id]["treatment_minus_control"]
        for family_id in FAMILIES
    }
    stop_reasons = []
    if not validity["MMD_WITNESS"]["pass"]:
        stop_reasons.append("MMD_WITNESS_hard_validity_failure")
    if judge["overall_quality"]["win_rate"] < 0.5 and judge["overall_quality"]["pvalue_treatment_worse"] <= statistics["alpha"]:
        stop_reasons.append("overall_quality_significantly_worse")
    if all(effect > 0 for effect in family_effects.values()) and judge["human_style"]["win_rate"] < 0.5:
        stop_reasons.append("both_embedding_families_worse_and_human_style_below_half")
    return {
        "artifact_schema": "dftr.m2.scale_ladder_4k_score.v1",
        "status": "completed",
        "contract_sha256": contract_sha,
        "panel_design_sha256": design.identity_sha256,
        "distribution": distribution,
        "token_unigram_l2": {
            "by_arm": token_l2,
            "treatment_minus_control": token_l2["MMD_WITNESS"]["l2"] - token_l2["SFT"]["l2"],
        },
        "hard_validity": validity,
        "repeated_bigram_rate": repetition,
        "byte_identity_with_control": identity,
        "judge": judge,
        "four_k_gate": {
            "decision": "stop" if stop_reasons else "unlock_16k",
            "stop_reasons": stop_reasons,
            "rule_source": contract["decision_rule_source"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--mmd-output", type=Path, required=True)
    parser.add_argument("--bge-embeddings", type=Path, required=True)
    parser.add_argument("--nemotron-embeddings", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(
        contract_path=args.contract,
        panel_root=args.panel_root,
        sft_output=args.sft_output,
        mmd_output=args.mmd_output,
        embedding_paths={"bge-small-v1": args.bge_embeddings, "nemotron-8b-v1": args.nemotron_embeddings},
        judge_results=args.judge_results,
        judge_manifest=args.judge_manifest,
        tokenizer_path=args.tokenizer_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "output_sha256": _sha(args.output), "four_k_gate": result["four_k_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
