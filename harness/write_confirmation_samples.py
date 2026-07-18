"""Write a compact, deterministic side-by-side confirmation sample sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_samples(
    briefs_path: Path,
    sft_path: Path,
    treatment_path: Path,
    output_path: Path,
    count: int = 10,
) -> None:
    briefs = {f"prompt-{row['fingerprint']}": row for row in _rows(briefs_path)}
    sft = {str(row["prompt_id"]): row for row in _rows(sft_path)}
    treatment = {str(row["prompt_id"]): row for row in _rows(treatment_path)}
    if set(briefs) != set(sft) or set(briefs) != set(treatment):
        raise ValueError("sample artifacts do not share exact prompt identities")
    selected = sorted(
        briefs,
        key=lambda prompt_id: hashlib.sha256(
            f"confirmation-samples-v1:{prompt_id}".encode()
        ).hexdigest(),
    )[:count]
    sections = [
        "# Humanwrite 4B confirmation samples",
        "",
        "These examples are a deterministic, non-cherry-picked subset of the fresh 128-prompt panel.",
        "",
    ]
    for index, prompt_id in enumerate(selected, start=1):
        brief = briefs[prompt_id]
        sections.extend(
            [
                f"## Example {index}",
                "",
                f"**Request:** {brief['user_prompt']}",
                "",
                "### Matched SFT control",
                "",
                str(sft[prompt_id]["generated_completion"]).strip(),
                "",
                "### Stronger MMD-witness treatment",
                "",
                str(treatment[prompt_id]["generated_completion"]).strip(),
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    write_samples(args.briefs, args.sft, args.treatment, args.output, args.count)
    print(args.output)


if __name__ == "__main__":
    main()
