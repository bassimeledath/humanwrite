"""Build a deterministic, side-by-side sample report for measurement v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("SFT", "TOKEN_MOMENT", "MMD_WITNESS")


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _block(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def build(briefs_path: Path, candidate_root: Path, output_path: Path) -> None:
    briefs = {
        f"prompt-{row['fingerprint']}": row for row in _rows(briefs_path)
    }
    candidates = {
        arm: {row["prompt_id"]: row for row in _rows(candidate_root / f"{arm}.jsonl")}
        for arm in ARMS
    }
    prompt_ids = sorted(candidates["SFT"])
    changed = [
        prompt_id
        for prompt_id in prompt_ids
        if any(
            candidates[arm][prompt_id]["generated_completion"]
            != candidates["SFT"][prompt_id]["generated_completion"]
            for arm in ARMS[1:]
        )
    ]
    selected = changed[:10]
    if len(selected) != 10:
        raise RuntimeError("fewer than ten changed measurement-v3 prompts")

    sections = [
        "# Humanwrite measurement-v3 samples",
        "",
        "These ten examples are the first ten changed prompt IDs in sorted order. "
        "Selection is deterministic and does not use metric or judge outcomes. The run "
        "was a 64-token screening experiment, so many responses end mid-sentence; this "
        "is evidence about the tuning methods, not a product-ready writing demo.",
        "",
    ]
    for index, prompt_id in enumerate(selected, start=1):
        brief = briefs[prompt_id]
        sections.extend(
            [
                f"## {index}. {prompt_id}",
                "",
                f"**Request:** {_block(str(brief['user_prompt']))}",
                "",
            ]
        )
        for arm in ARMS:
            completion = _block(
                str(candidates[arm][prompt_id]["generated_completion"])
            )
            sections.extend(
                [
                    f"### {arm}",
                    "",
                    completion,
                    "",
                ]
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.briefs, args.candidate_root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
