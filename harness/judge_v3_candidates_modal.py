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
PANEL_ROOT = CHECKPOINT_ROOT / "data/m2-lower-variance-v1/measurement-v3-panels"
PROTOCOL_PATH = PANEL_ROOT / "protocol-v1/measurement_protocol_v3.json"
CANDIDATE_ROOT = PANEL_ROOT / "candidate-outputs-v1"
BRIEFS_PATH = PANEL_ROOT / "prompt_briefs-128-normalized.jsonl"
OUTPUT_ROOT = PANEL_ROOT / "quality-judge-v1"
LOCAL_CONTRACT = Path("/root/judge_contract.json")
ARMS = ("SFT", "TOKEN_MOMENT", "MMD_WITNESS")
TREATMENTS = ("TOKEN_MOMENT", "MMD_WITNESS")

checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
contract_path = (
    Path(__file__).resolve().parents[1]
    / "configs/m2/m2_measurement_v3_judge_contract_v1.json"
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.32,<3")
    .add_local_file(contract_path, remote_path=str(LOCAL_CONTRACT), copy=True)
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
def judge() -> dict:
    import requests

    checkpoint_volume.reload()
    contract = json.loads(LOCAL_CONTRACT.read_text(encoding="utf-8"))
    model = os.environ.get("DFTR_JUDGE_MODEL", "")
    if (
        contract.get("artifact_schema")
        != "dftr.measurement.quality_judge_contract.v3"
        or contract.get("status") != "frozen"
        or contract.get("candidate_outputs_opened") is not False
        or model != contract.get("model")
    ):
        raise RuntimeError("quality judge does not match its frozen contract")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_sha = _sha(PROTOCOL_PATH)
    if (
        protocol.get("artifact_schema") != "dftr.measurement.protocol.v3"
        or protocol.get("status") != "ready"
        or protocol.get("frozen") is not True
        or protocol.get("candidate_outputs_opened") is not False
    ):
        raise RuntimeError("quality judge requires a candidate-blind protocol freeze")
    briefs = _rows(BRIEFS_PATH)
    brief_map = {f"prompt-{row['fingerprint']}": row for row in briefs}
    candidates = {}
    candidate_hashes = {}
    for arm in ARMS:
        output_path = CANDIDATE_ROOT / f"{arm}.jsonl"
        manifest = json.loads(
            (CANDIDATE_ROOT / f"{arm}.manifest.json").read_text(encoding="utf-8")
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
    if any(set(candidates[arm]) != set(prompt_ids) for arm in ARMS):
        raise RuntimeError("quality judge prompt identities do not align")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / "results.jsonl"
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
    for treatment in TREATMENTS:
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
                cost = (body.get("usage") or {}).get("cost")
                content = str(body["choices"][0]["message"]["content"]).strip()
                match = re.fullmatch(r"[AB]", content)
                if cost is None or match is None:
                    raise RuntimeError("judge response violated cost or choice contract")
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
    expected = len(TREATMENTS) * len(contract["dimensions"]) * len(prompt_ids)
    rows = _rows(output_path)
    if len(rows) != expected:
        raise RuntimeError(
            f"quality judge incomplete: {len(rows)}/{expected}, spent={spent:.6f}"
        )
    manifest = {
        "artifact_schema": "dftr.measurement.quality_judge_results.v3",
        "status": "completed",
        "contract_sha256": _sha(LOCAL_CONTRACT),
        "protocol_sha256": protocol_sha,
        "candidate_output_sha256": candidate_hashes,
        "model": model,
        "comparisons": len(rows),
        "cost_usd": round(spent, 6),
        "output_path": str(output_path),
        "output_sha256": _sha(output_path),
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(judge.remote(), indent=2, sort_keys=True))
