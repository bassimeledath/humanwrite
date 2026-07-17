"""Render the informal SFT smoke comparison as a self-contained Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _table_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br><br>")


def render(comparison: dict, training_step: dict) -> str:
    samples = comparison["samples"]
    identical = sum(int(sample["byte_identical"]) for sample in samples)
    before_words = sum(sample["before_metrics"]["words"] for sample in samples)
    after_words = sum(sample["after_metrics"]["words"] for sample in samples)
    before_repeated = sum(sample["before_metrics"]["repeated_bigrams"] for sample in samples)
    after_repeated = sum(sample["after_metrics"]["repeated_bigrams"] for sample in samples)
    before_emdashes = sum(sample["before_metrics"]["emdashes"] for sample in samples)
    after_emdashes = sum(sample["after_metrics"]["emdashes"] for sample in samples)
    before_non_latin = sum(sample["before_metrics"]["non_latin_letters"] for sample in samples)
    after_non_latin = sum(sample["after_metrics"]["non_latin_letters"] for sample in samples)

    lines = [
        "# Humanwrite: first fine-tuning smoke run",
        "",
        "Generated July 17, 2026.",
        "",
        "> This is an informal diagnostic, not evidence that human-likeness improved. The run made only one optimizer update using two examples. The **before** model is the exact initial adapter used by this run, not untouched Qwen3-4B; the **after** model is its one-step SFT checkpoint.",
        "",
        "## What the training run measured",
        "",
        "| Metric | Result | Meaning |",
        "|---|---:|---|",
        f"| Optimizer updates | {training_step['step'] + 1} | The model weights were updated once. |",
        f"| Training examples consumed | {training_step['optimizer_examples']} | Two of the 128 available briefs were used in this smoke test. |",
        f"| Completion tokens trained | {training_step['teacher_forced_completion_tokens']} | Tokens that directly contributed to the supervised loss. |",
        "| Trainable LoRA parameters | 132,120,576 | Adapter parameters available to update; the 4B base weights remained frozen. |",
        f"| SFT loss | {training_step['uniform_sft_loss']:.4f} | Finite training loss confirms the forward/backward path ran. One value cannot establish improvement. |",
        f"| Pre-clipping gradient norm | {training_step['preclip_total_gradient_norm']:.4f} | Nonzero gradients reached the adapter. They were clipped to the configured maximum of 1.0. |",
        "",
        "## Ten held-out side-by-side samples",
        "",
        "These briefs were outside the 128-record smoke snapshot. Before and after used the same prompt, temperature (0.8), top-p (0.95), and per-example random seed.",
        "",
        "### Quick output diagnostics",
        "",
        "| Metric across 10 samples | Before | After |",
        "|---|---:|---:|",
        f"| Total words | {before_words} | {after_words} |",
        f"| Mean words per sample | {before_words / len(samples):.1f} | {after_words / len(samples):.1f} |",
        f"| Repeated bigrams (simple count) | {before_repeated} | {after_repeated} |",
        f"| Em dashes | {before_emdashes} | {after_emdashes} |",
        f"| Unexpected non-Latin letters | {before_non_latin} | {after_non_latin} |",
        f"| Byte-identical pairs | {identical}/{len(samples)} | {identical}/{len(samples)} |",
        "",
        f"Six outputs changed and {identical} remained exactly identical. The small reduction in repeated bigrams is descriptive only: ten samples and one update are far too little for a quality claim, and some individual changed outputs became worse.",
        "",
    ]
    for sample in samples:
        marker = "identical" if sample["byte_identical"] else "changed"
        lines.extend(
            [
                f"### {sample['number']}. {sample['use_case']} ({marker})",
                "",
                f"**Prompt:** {_table_text(sample['user_prompt'])}",
                "",
                "| Before this run | After one fine-tuning step |",
                "|---|---|",
                f"| {_table_text(sample['before'])} | {_table_text(sample['after'])} |",
                "",
                f"Words: **{sample['before_metrics']['words']} before / {sample['after_metrics']['words']} after**. Sampling seed: `{sample['sampling_seed']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Honest interpretation",
            "",
            "The useful result is operational: the 4B model loaded, received a real gradient, updated its LoRA, saved a checkpoint, and produced valid text afterward. This run was deliberately too small to answer whether the model became more human-like. The ten examples show exactly what we should expect from one update: mostly identical or modestly changed continuations, with no consistent quality direction yet.",
            "",
            "The real answer will come from the larger matched fine-tuning arms and an independent held-out evaluation—not from selecting whichever examples look best.",
            "",
            f"Comparison artifact SHA-256: `{comparison['content_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("training_step", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    steps = [
        json.loads(line)
        for line in args.training_step.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(steps) != 1:
        raise SystemExit(f"expected exactly one training step, found {len(steps)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(comparison, steps[0]), encoding="utf-8")


if __name__ == "__main__":
    main()
