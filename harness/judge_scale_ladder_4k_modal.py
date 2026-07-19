"""Blinded human-style and overall-quality judge for the frozen 4K comparison."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_LOCAL = ROOT / "configs/m2/m2_scale_ladder_4k_scoring_contract_v1.json"
CONTRACT_REMOTE = Path("/root/m2_scale_ladder_4k_scoring_contract_v1.json")
CHECKPOINT_ROOT = Path("/checkpoints")
OUTPUT_ROOT = CHECKPOINT_ROOT / "data/m2-scale-ladder-v1/evaluation-4k-v1/judge"
APP_NAME = "humanwrite-scale-ladder-4k-judge"

checkpoint_volume = modal.Volume.from_name("humanwrite-checkpoints")
provider_secret = modal.Secret.from_name("the-other-ones")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.32,<3")
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


def _assignment(row: dict) -> str:
    outline = row.get("outline") or []
    outline_text = "\n".join(f"- {item}" for item in outline) if outline else "(none)"
    return "\n".join(
        (
            f"Writing request: {row['user_prompt']}",
            f"Use case: {row['use_case']}",
            f"Style: {row['style']}",
            f"Target length: about {row['target_length']} tokens",
            f"Outline:\n{outline_text}",
        )
    )


def _side(master_seed: int, dimension: str, prompt_id: str) -> str:
    digest = hashlib.sha256(
        f"{master_seed}:{dimension}:{prompt_id}".encode("utf-8")
    ).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def _prompt(assignment: str, rubric: str, candidate_a: str, candidate_b: str) -> str:
    return f"""Given a writing assignment and candidates A and B, choose the better response.

Criterion: {rubric}

Return only A or B.

=== ASSIGNMENT ===
{assignment}

=== CANDIDATE A ===
{candidate_a}

=== CANDIDATE B ===
{candidate_b}
"""


@app.function(
    image=image,
    timeout=45 * 60,
    secrets=[provider_secret],
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def judge() -> dict:
    import requests

    checkpoint_volume.reload()
    contract = json.loads(CONTRACT_REMOTE.read_text(encoding="utf-8"))
    contract_sha = _sha(CONTRACT_REMOTE)
    artifacts = contract["artifacts"]
    loaded: dict[str, list[dict]] = {}
    for role in ("prompt_briefs", "SFT", "MMD_WITNESS"):
        binding = artifacts[role]
        path = Path(binding["path"])
        rows = _rows(path)
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha(path) != binding["sha256"]
            or len(rows) != binding["rows"]
        ):
            raise RuntimeError(f"judge artifact binding failed: {role}")
        loaded[role] = rows
    briefs = {str(row.get("prompt_id") or row["fingerprint"]): row for row in loaded["prompt_briefs"]}
    candidates = {
        role: {str(row["prompt_id"]): str(row["text"]) for row in loaded[role]}
        for role in ("SFT", "MMD_WITNESS")
    }
    prompt_ids = sorted(briefs)
    if len(prompt_ids) != 128 or any(set(values) != set(prompt_ids) for values in candidates.values()):
        raise RuntimeError("judge prompt pairing failed")

    judge_contract = contract["judge"]
    tasks: list[dict] = []
    for dimension, rubric in judge_contract["dimensions"].items():
        for prompt_id in prompt_ids:
            treatment_side = _side(
                int(judge_contract["randomization"]["master_seed"]), dimension, prompt_id
            )
            texts = {
                treatment_side: candidates["MMD_WITNESS"][prompt_id],
                "B" if treatment_side == "A" else "A": candidates["SFT"][prompt_id],
            }
            tasks.append(
                {
                    "dimension": dimension,
                    "prompt_id": prompt_id,
                    "treatment_side": treatment_side,
                    "prompt": _prompt(
                        _assignment(briefs[prompt_id]), rubric, texts["A"], texts["B"]
                    ),
                }
            )

    api_key = os.environ["OPENROUTER_API_KEY"]

    def request_one(task: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(int(judge_contract["retry_attempts"])):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/bassimeledath/humanwrite",
                        "X-OpenRouter-Title": "Humanwrite 4K blinded writing judge",
                    },
                    json={
                        "model": judge_contract["model"],
                        "messages": [{"role": "user", "content": task["prompt"]}],
                        "temperature": judge_contract["temperature"],
                        "max_completion_tokens": judge_contract["max_completion_tokens"],
                        "reasoning": {"effort": "minimal", "exclude": True},
                    },
                    timeout=180,
                )
                response.raise_for_status()
                body = response.json()
                content = str(body["choices"][0]["message"]["content"]).strip()
                usage = body.get("usage") or {}
                if re.fullmatch(r"[AB]", content) is None or usage.get("cost") is None:
                    choice = body.get("choices", [{}])[0]
                    raise RuntimeError(
                        "judge response violated the A/B or cost contract: "
                        f"content={content!r}, cost={usage.get('cost')!r}, "
                        f"finish_reason={choice.get('finish_reason')!r}, "
                        f"usage_keys={sorted(usage)}"
                    )
                return {
                    "dimension": task["dimension"],
                    "prompt_id": task["prompt_id"],
                    "treatment_side": task["treatment_side"],
                    "choice": content,
                    "treatment_win": content == task["treatment_side"],
                    "cost_usd": float(usage["cost"]),
                    "model": judge_contract["model"],
                    "attempt": attempt + 1,
                }
            except Exception as error:
                last_error = error
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"judge request failed: {last_error}")

    results: list[dict] = []
    spent = 0.0
    concurrency = int(judge_contract["concurrency"])
    for start in range(0, len(tasks), concurrency):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(request_one, task) for task in tasks[start : start + concurrency]]
            for future in as_completed(futures):
                row = future.result()
                spent += float(row["cost_usd"])
                if spent > float(judge_contract["cost_cap_usd"]):
                    raise RuntimeError("judge cost cap exceeded")
                results.append(row)
    results.sort(key=lambda row: (row["dimension"], row["prompt_id"]))
    if len(results) != 256:
        raise RuntimeError("judge result cardinality failed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / "results.jsonl"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    manifest = {
        "artifact_schema": "dftr.m2.scale_ladder_judge_results.v1",
        "status": "completed",
        "contract_sha256": contract_sha,
        "model": judge_contract["model"],
        "comparisons": len(results),
        "cost_usd": round(spent, 6),
        "output_path": str(output),
        "output_sha256": _sha(output),
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checkpoint_volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(judge.remote(), indent=2, sort_keys=True))
