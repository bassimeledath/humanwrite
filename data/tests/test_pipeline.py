from __future__ import annotations

import json

from data.pipeline import DEFAULT_INPUT, build_dataset, write_dataset


def test_fixture_build_is_deterministic(tmp_path):
    first = build_dataset(DEFAULT_INPUT)
    second = build_dataset(DEFAULT_INPUT)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    left = tmp_path / "left"
    right = tmp_path / "right"
    write_dataset(first, left)
    write_dataset(second, right)
    for name in (
        "cleaned_records.jsonl",
        "brief_records_all.jsonl",
        "train_briefs.jsonl",
        "dev_briefs.jsonl",
        "train_manifest.json",
        "dev_manifest.json",
        "split_hashes.json",
        "hidden_test_boundary.json",
        "summary.json",
        "source.json",
    ):
        assert (left / name).read_text(encoding="utf-8") == (right / name).read_text(encoding="utf-8")


def test_fixture_matches_m0_acceptance_contract():
    dataset = build_dataset(DEFAULT_INPUT)
    briefs = dataset["briefs"]
    assert len(briefs) == 8
    assert len(dataset["train"]) == 6
    assert len(dataset["dev"]) == 2
    assert sum(1 for row in briefs if not row["outline"]) == 2
    for row in briefs:
        assert set(row) == {
            "fineweb_id",
            "domain",
            "fingerprint",
            "split",
            "generation_mode",
            "use_case",
            "style_kind",
            "style",
            "detail_mode",
            "target_length",
            "em_dashes_allowed",
            "user_prompt",
            "outline",
            "completion",
        }


def test_hidden_test_boundary_is_metadata_only():
    boundary = build_dataset(DEFAULT_INPUT)["hidden_test_boundary"]
    assert boundary["materialized_locally"] is False
    assert "completion" in boundary["excluded_fields"]
    assert "outline" in boundary["excluded_fields"]
