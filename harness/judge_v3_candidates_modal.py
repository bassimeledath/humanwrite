"""Run frozen, blinded pairwise writing judgments over both v3 treatments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time

import modal


APP_NAME = "humanwrite-measurement-v3-quality-judge"
CHECKPOINT_ROOT = Path("/checkpoints")
PANEL_ROOTS = {
    "v3": CHECKPOINT_ROOT / "data/m2-lower-variance-v1/measurement-v3-panels",
    "v4": CHECKPOINT_ROOT / "data/m2-confirmation-v1/measurement-v4-panels",
}
ARMS_BY_CYCLE = {
    "v3": ("SFT", "TOKEN_MOMENT", "MMD_WITNESS"),
    "v4": ("SFT", "MMD_WITNESS"),
}
TREATMENTS_BY_CYCLE = {
    "v3": ("TOKEN_MOMENT", "MMD_WITNESS"),
    "v4": ("MMD_WITNESS",),
}

checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
contract_path_v3 = (
    Path(__file__).resolve().parents[1]
    / "configs/m2/m2_measurement_v3_judge_contract_v3.json"
)
contract_path_v4 = (
    Path(__file__).resolve().parents[1]
    / "configs/m2/m2_measurement_v4_judge_contract_v1.json"
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.32,<3")
    .add_local_file(contract_path_v3, remote_path="/root/judge_contract_v3.json", copy=True)
    .add_local_file(contract_path_v4, remote_path="/root/judge_contract_v4.json", copy=True)
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


def _side(contract: dict, treatment: str, dimension: str, prompt_id: str) -> str:
    seed = contract["randomization"]["master_seed"]
    digest = hashlib.sha256(
        f"{seed}:{treatment}:{dimension}:{prompt_id}".encode()
    ).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def _assignment(brief: dict) -> str:
    return "\n".join(
        (
            f"Writing request: {brief['user_prompt']}",
            f"Use case: {brief['use_case']}",
            f"Style: {brief['style']}",
            f"Target length: about {brief['target_length']} tokens",
            f"Em dashes allowed: {'yes' if brief['em_dashes_allowed'] else 'no'}",
            "Grounding outline: "
            + json.dumps(brief["outline"], ensure_ascii=False, sort_keys=True),
        )
    )


def _judge_prompt(assignment: str, rubric: str, candidate_a: str, candidate_b: str) -> str:
    return f"""Given the assignment and two candidate responses, {rubric}

Return only A or B. Do not explain your answer.

=== ASSIGNMENT ===
{assignment}

=== CANDIDATE A ===
{candidate_a}

=== CANDIDATE B ===
{candidate_b}
"""


@app.function(
    image=image,
    timeout=2 * 60 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def judge(cycle: str = "v3") -> dict:
    import requests

    checkpoint_volume.reload()
    if cycle not in PANEL_ROOTS:
        raise ValueError(f"unknown cycle: {cycle}")
    panel_root = PANEL_ROOTS[cycle]
    protocol_path = panel_root / "protocol-v1/measurement_protocol_v3.json"
    candidate_root = panel_root / "candidate-outputs-v1"
    briefs_path = panel_root / "prompt_briefs-128-normalized.jsonl"
    output_root = panel_root / "quality-judge-v1"
    arms = ARMS_BY_CYCLE[cycle]
    treatments = TREATMENTS_BY_CYCLE[cycle]
    local_contract = Path(f"/root/judge_contract_{cycle}.json")
    contract = json.loads(local_contract.read_text(encoding="utf-8"))
    model = os.environ.get("DFTR_JUDGE_MODEL", "")
    legacy_contract = (
        contract.get("artifact_schema") == "dftr.measurement.quality_judge_contract.v3"
        and contract.get("status") == "frozen_operational_amendment"
        and contract.get("candidate_outputs_opened") is True
        and contract.get("amendment", {}).get("scope") == "operational_only"
        and contract.get("amendment", {}).get("analytic_changes") is False
    )
    confirmation_contract = (
        contract.get("artifact_schema") == "dftr.measurement.quality_judge_contract.v4"
        and contract.get("status") == "frozen_candidate_blind"
        and contract.get("candidate_outputs_opened") is False
    )
    if not (legacy_contract or confirmation_contract) or model != contract.get("model"):
        raise RuntimeError("quality judge does not match its frozen contract")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = _sha(protocol_path)
    if (
        protocol.get("artifact_schema") != "dftr.measurement.protocol.v3"
        or protocol.get("status") != "ready"
        or protocol.get("frozen") is not True
        or protocol.get("candidate_outputs_opened") is not False
    ):
        raise RuntimeError("quality judge requires a candidate-blind protocol freeze")
    briefs = _rows(briefs_path)
    brief_map = {f"prompt-{row['fingerprint']}": row for row in briefs}
    candidates = {}
    candidate_hashes = {}
    for arm in arms:
        output_path = candidate_root / f"{arm}.jsonl"
        manifest = json.loads(
            (candidate_root / f"{arm}.manifest.json").read_text(encoding="utf-8")
        )
        rows = _rows(output_path)
        if (
            manifest.get("protocol_sha256") != protocol_sha
            or manifest.get("output_sha256") != _sha(output_path)
            or len(rows) != contract["prompt_count"]
        ):
            raise RuntimeError(f"quality judge candidate binding failed for {arm}")
        candidates[arm] = {
            str(row["prompt_id"]): str(row["generated_completion"]) for row in rows
        }
        candidate_hashes[arm] = manifest["output_sha256"]
    prompt_ids = sorted(brief_map)
    if any(set(candidates[arm]) != set(prompt_ids) for arm in arms):
        raise RuntimeError("quality judge prompt identities do not align")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "results.jsonl"
    completed = {}
    spent = 0.0
    if output_path.is_file():
        for row in _rows(output_path):
            key = (row["treatment"], row["dimension"], row["prompt_id"])
            if key in completed or row.get("choice") not in {"A", "B"}:
                raise RuntimeError("existing quality-judge result is invalid")
            completed[key] = row
            spent += float(row["cost_usd"])

    tasks = []
    for treatment in treatments:
        for dimension, rubric in contract["dimensions"].items():
            for prompt_id in prompt_ids:
                key = (treatment, dimension, prompt_id)
                if key in completed:
                    continue
                treatment_side = _side(contract, treatment, dimension, prompt_id)
                control_side = "B" if treatment_side == "A" else "A"
                texts = {
                    treatment_side: candidates[treatment][prompt_id],
                    control_side: candidates["SFT"][prompt_id],
                }
                tasks.append(
                    {
                        "key": key,
                        "treatment": treatment,
                        "dimension": dimension,
                        "prompt_id": prompt_id,
                        "treatment_side": treatment_side,
                        "prompt": _judge_prompt(
                            _assignment(brief_map[prompt_id]),
                            rubric,
                            texts["A"],
                            texts["B"],
                        ),
                    }
                )

    api_key = os.environ["OPENROUTER_API_KEY"]

    def request_one(task: dict) -> dict:
        last_error = None
        for attempt in range(contract["execution"]["retry_attempts"]):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                        "X-OpenRouter-Title": "Humanwrite blinded writing judge",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": task["prompt"]}],
                        "temperature": contract["request"]["temperature"],
                        "max_completion_tokens": contract["request"][
                            "max_completion_tokens"
                        ],
                        "reasoning": {"effort": "minimal", "exclude": True},
                    },
                    timeout=180,
                )
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                cost = usage.get("cost")
                content = str(body["choices"][0]["message"]["content"]).strip()
                match = re.fullmatch(r"[AB]", content)
                if cost is None or match is None:
                    choice = body["choices"][0]
                    raise RuntimeError(
                        "judge response violated cost or choice contract: "
                        f"content={content!r}, cost={cost!r}, "
                        f"finish_reason={choice.get('finish_reason')!r}, "
                        f"usage_keys={sorted(usage)}"
                    )
                return {
                    "treatment": task["treatment"],
                    "dimension": task["dimension"],
                    "prompt_id": task["prompt_id"],
                    "treatment_side": task["treatment_side"],
                    "choice": match.group(0),
                    "treatment_win": match.group(0) == task["treatment_side"],
                    "cost_usd": float(cost),
                    "model": model,
                    "attempt": attempt + 1,
                }
            except Exception as error:
                last_error = error
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"quality judge request failed: {last_error}")

    concurrency = int(contract["execution"]["concurrency"])
    cost_cap = float(contract["execution"]["cost_cap_usd"])
    with output_path.open("a", encoding="utf-8") as sink:
        for start in range(0, len(tasks), concurrency):
            batch = tasks[start : start + concurrency]
            if spent >= cost_cap:
                break
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(request_one, task) for task in batch]
                for future in as_completed(futures):
                    row = future.result()
                    spent += float(row["cost_usd"])
                    sink.write(json.dumps(row, sort_keys=True) + "\n")
                    sink.flush()
            checkpoint_volume.commit()
    expected = len(treatments) * len(contract["dimensions"]) * len(prompt_ids)
    rows = _rows(output_path)
    if len(rows) != expected:
        raise RuntimeError(
            f"quality judge incomplete: {len(rows)}/{expected}, spent={spent:.6f}"
        )
    manifest = {
        "artifact_schema": "dftr.measurement.quality_judge_results.v3",
        "status": "completed",
        "contract_sha256": _sha(local_contract),
        "protocol_sha256": protocol_sha,
        "candidate_output_sha256": candidate_hashes,
        "model": model,
        "comparisons": len(rows),
        "cost_usd": round(spent, 6),
        "output_path": str(output_path),
        "output_sha256": _sha(output_path),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_volume.commit()
    return manifest


@app.local_entrypoint()
def main(cycle: str = "v3") -> None:
    print(json.dumps(judge.remote(cycle), indent=2, sort_keys=True))
