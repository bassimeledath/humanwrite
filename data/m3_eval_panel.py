"""Fresh, source-disjoint M3 rewrite evaluation-panel construction helpers."""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any, Callable

from data.m3_scientific_corpus import (
    M3ScientificCorpusError,
    SCIENTIFIC_REWRITE_PROTOCOL,
    render_noop_prompt,
    render_scientific_rewrite_prompt,
    scientific_assignment,
    validate_scientific_rewrite,
)
from data.rewrite_tasks import protected_literals


EVAL_REWRITE_INPUT_PROTOCOL = "humanwrite.m3.eval_rewrite_input.v1"
EVAL_PANEL_PROTOCOL = "humanwrite.m3.rewrite_eval_panel.v1"
CATEGORY_COUNTS = {
    "natural_ai": 128,
    "fact_dense": 48,
    "light_edit": 48,
    "already_human_noop": 32,
}
API_CATEGORIES = {"natural_ai", "fact_dense", "light_edit"}


class M3EvalPanelError(ValueError):
    pass


def _rank(row: dict[str, Any]) -> str:
    fingerprint = str(row.get("fingerprint") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise M3EvalPanelError("evaluation source fingerprint is invalid")
    return hashlib.sha256(f"m3-eval-panel:{fingerprint}".encode()).hexdigest()


def eval_panel_manifest(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(sources) != 640 or len({str(row.get("fingerprint") or "") for row in sources}) != 640:
        raise M3EvalPanelError("fresh evaluation source bank must contain 640 unique rows")
    ordered = sorted(sources, key=_rank)
    selected: dict[str, str] = {}
    for row in ordered[:32]:
        selected[str(row["fingerprint"])] = "already_human_noop"
    remaining = [row for row in ordered if str(row["fingerprint"]) not in selected]
    fact_candidates = [row for row in remaining if protected_literals(str(row.get("completion") or ""))]
    if len(fact_candidates) < 48:
        raise M3EvalPanelError("fresh evaluation pool lacks 48 fact-dense targets")
    for row in fact_candidates[:48]:
        selected[str(row["fingerprint"])] = "fact_dense"
    remaining = [row for row in remaining if str(row["fingerprint"]) not in selected]
    for row in remaining[:48]:
        selected[str(row["fingerprint"])] = "light_edit"
    remaining = [row for row in remaining if str(row["fingerprint"]) not in selected]
    for row in remaining[:128]:
        selected[str(row["fingerprint"])] = "natural_ai"
    counters: Counter[str] = Counter()
    result = []
    for row in ordered:
        fingerprint = str(row["fingerprint"])
        category = selected.get(fingerprint)
        if category is None:
            continue
        if category == "already_human_noop":
            assignment = {
                "generator_model": "identity",
                "verifier_model": "deterministic",
                "template_id": "minimal_restraint",
            }
            origin = "already_human_noop"
        else:
            origin = "controlled_light_edit" if category == "light_edit" else "multi_provider_ai"
            assignment = scientific_assignment(row, origin, slot=counters[category])
            counters[category] += 1
        result.append(
            {
                "artifact_schema": "humanwrite.m3.eval_panel_manifest.v1",
                "fingerprint": fingerprint,
                "category": category,
                "scientific_origin": origin,
                **assignment,
            }
        )
    counts = Counter(str(row["category"]) for row in result)
    if len(result) != 256 or counts != CATEGORY_COUNTS:
        raise M3EvalPanelError(f"fresh evaluation category counts drifted: {dict(counts)}")
    for category in API_CATEGORIES:
        providers = Counter(
            str(row["generator_model"]) for row in result if row["category"] == category
        )
        if len(providers) != 2 or max(providers.values()) != CATEGORY_COUNTS[category] // 2:
            raise M3EvalPanelError(f"evaluation provider balance drifted for {category}")
    return result


def emit_eval_rewrite_input(scientific_row: dict[str, Any], category: str) -> dict[str, Any]:
    if category not in API_CATEGORIES or scientific_row.get("artifact_schema") != SCIENTIFIC_REWRITE_PROTOCOL:
        raise M3EvalPanelError("evaluation rewrite emission inputs are invalid")
    return {
        **scientific_row,
        "artifact_schema": EVAL_REWRITE_INPUT_PROTOCOL,
        "evaluation_category": category,
    }


def validate_eval_rewrite_input(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    manifest_row: dict[str, Any],
    token_counter: Callable[[str], int],
    semantic_similarity_min: float = 0.90,
) -> None:
    category = str(manifest_row.get("category") or "")
    if category not in API_CATEGORIES or row.get("evaluation_category") != category:
        raise M3EvalPanelError("evaluation rewrite category drifted")
    scientific = dict(row)
    scientific.pop("evaluation_category", None)
    scientific["artifact_schema"] = SCIENTIFIC_REWRITE_PROTOCOL
    try:
        validate_scientific_rewrite(
            scientific,
            source=source,
            origin=str(manifest_row["scientific_origin"]),
            token_counter=token_counter,
            semantic_similarity_min=semantic_similarity_min,
            expected_assignment={
                key: str(manifest_row[key])
                for key in ("generator_model", "verifier_model", "template_id")
            },
        )
    except M3ScientificCorpusError as exc:
        raise M3EvalPanelError(str(exc)) from exc


def assemble_eval_panel(
    sources: list[dict[str, Any]],
    api_rows: list[dict[str, Any]],
    *,
    token_counter: Callable[[str], int],
) -> list[dict[str, Any]]:
    manifest = eval_panel_manifest(sources)
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    manifest_by_id = {str(row["fingerprint"]): row for row in manifest}
    api_by_id: dict[str, dict[str, Any]] = {}
    for row in api_rows:
        fingerprint = str(row.get("fingerprint") or "")
        manifest_row = manifest_by_id.get(fingerprint)
        if manifest_row is None or manifest_row["category"] not in API_CATEGORIES or fingerprint in api_by_id:
            raise M3EvalPanelError("evaluation API rewrite identity mismatch")
        validate_eval_rewrite_input(
            row,
            source=source_by_id[fingerprint],
            manifest_row=manifest_row,
            token_counter=token_counter,
        )
        api_by_id[fingerprint] = row
    expected_api = {
        str(row["fingerprint"]) for row in manifest if row["category"] in API_CATEGORIES
    }
    if set(api_by_id) != expected_api:
        raise M3EvalPanelError("evaluation API rewrite artifact is incomplete")
    result = []
    for assignment in manifest:
        fingerprint = str(assignment["fingerprint"])
        source = source_by_id[fingerprint]
        category = str(assignment["category"])
        target = str(source.get("completion") or "").strip()
        if category == "already_human_noop":
            input_text = target
            prompt = render_noop_prompt(source)
            literals = list(protected_literals(target))
        else:
            rewrite = api_by_id[fingerprint]
            input_text = str(rewrite["input_text"])
            prompt = render_scientific_rewrite_prompt(rewrite, source)
            literals = list(rewrite["protected_literals"])
        result.append(
            {
                "artifact_schema": EVAL_PANEL_PROTOCOL,
                "fingerprint": fingerprint,
                "source_fingerprint": str(source.get("source_fingerprint") or fingerprint),
                "domain": str(source.get("domain") or ""),
                "category": category,
                "prompt": prompt,
                "input_text": input_text,
                "human_reference": target,
                "target_length": token_counter(target),
                "target_length_unit": "tokens",
                "protected_literals": literals,
                "generator_model": str(assignment["generator_model"]),
                "verifier_model": str(assignment["verifier_model"]),
                "template_id": str(assignment["template_id"]),
            }
        )
    counts = Counter(str(row["category"]) for row in result)
    if len(result) != 256 or counts != CATEGORY_COUNTS or len({row["domain"] for row in result}) != 256:
        raise M3EvalPanelError("assembled evaluation panel invariants failed")
    return result


__all__ = [
    "API_CATEGORIES",
    "CATEGORY_COUNTS",
    "EVAL_PANEL_PROTOCOL",
    "EVAL_REWRITE_INPUT_PROTOCOL",
    "M3EvalPanelError",
    "assemble_eval_panel",
    "emit_eval_rewrite_input",
    "eval_panel_manifest",
    "validate_eval_rewrite_input",
]
