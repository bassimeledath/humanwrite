from __future__ import annotations

import numpy as np

from experiments.tier0.metrics import TRAINING_ONLY_NOTICE, batch_diagnostics, distribution_gap


def embedder(texts):
    rows = []
    for text in texts:
        rows.append([len(text), text.count("e"), text.count("a")])
    return np.asarray(rows, dtype=float)


def test_distribution_gap_and_diagnostics_are_offline():
    generated = [
        "Acme reduced pick time by fourteen percent and simplified scanner prompts.",
        "The harbor expansion adds dredging and rail access in a phased plan.",
    ]
    human = [
        "Acme reduced average pick time by 14 percent during the third quarter.",
        "City officials approved dredging, rail access, and a traffic plan.",
    ]
    outlines = [
        [{"section": "Acme", "supported_facts": ["Acme reduced pick time by 14 percent"], "quotations": []}],
        [{"section": "Harbor", "supported_facts": ["The plan funds dredging and rail access"], "quotations": []}],
    ]
    gap = distribution_gap(generated, human, embedder)
    assert gap["weighted_gap"] >= 0.0

    report = batch_diagnostics(generated, human, outlines, embedder, targets=[12, 12])
    assert report["notice"] == TRAINING_ONLY_NOTICE
    assert report["validity"]["outline_fact_recall"] > 0
    assert "mean_abs_target_error" in report["length"]


def test_collapse_probe_detects_repetition_signal():
    repeated = [
        "Start here. Start again. Start once more.",
        "Repeat this. Repeat that. Repeat everything.",
    ]
    human = [
        "A short article with varied openings.",
        "Another document that avoids the same sentence stem.",
    ]
    outlines = [[], []]
    report = batch_diagnostics(repeated, human, outlines, embedder)
    assert report["diversity"]["repeated_sentence_start_rate"] > 0
