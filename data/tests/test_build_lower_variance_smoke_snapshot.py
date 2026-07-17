from __future__ import annotations

import json

from data import build_lower_variance_smoke_snapshot as snapshot


def test_freezes_sorted_snapshot_and_marks_witness_as_proxy(tmp_path, monkeypatch):
    source = tmp_path / "briefs.jsonl"
    rows = [
        {"fingerprint": f"{index:064x}", "completion": f"human text {index}"}
        for index in reversed(range(4))
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["snapshot", "--input", str(source), "--output-dir", str(output), "--count", "3"],
    )
    assert snapshot.main() == 0
    anchors = [json.loads(line) for line in (output / "anchors-128.jsonl").read_text().splitlines()]
    witness = [json.loads(line) for line in (output / "witness-proxy-128.jsonl").read_text().splitlines()]
    assert [row["completion"] for row in anchors] == ["human text 0", "human text 1", "human text 2"]
    assert [row["generated_completion"] for row in witness] == [row["completion"] for row in anchors]
    assert {row["smoke_only_proxy"] for row in witness} == {
        "human_completion_not_a_model_rollout"
    }
