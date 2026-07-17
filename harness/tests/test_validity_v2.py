import pytest

from harness.metrics.distribution_v2 import MeasurementV2Error
from harness.metrics.validity_v2 import repetition_noninferiority, same_n_self_bleu


PLAIN = "One sentence starts. Another thought follows. Finally the text ends."
REPEATED = "Again the first line. Again the second line. Again the third line."


def test_repetition_is_explicitly_underpowered_at_small_n():
    result = repetition_noninferiority(
        [PLAIN] * 16, [PLAIN] * 16, margin=0.1, power_plan_passed=True
    )
    assert result["status"] == "underpowered"
    assert result["decision"] == "not_promoting"
    assert result["zero_candidate_events_never_fail_as_too_low"] is True


def test_repetition_noninferiority_can_fail_when_powered():
    result = repetition_noninferiority(
        [REPEATED] * 64, [PLAIN] * 64, margin=0.1, power_plan_passed=True
    )
    assert result["status"] == "ready"
    assert result["decision"] == "fail"
    assert result["candidate_successes"] == 64


def test_self_bleu_uses_same_n_and_n_minus_one_references():
    result = same_n_self_bleu(["alpha beta", "gamma delta"], ["one two", "three four"])
    assert result["documents_per_panel"] == 2
    assert result["references_per_document"] == 1
    with pytest.raises(MeasurementV2Error, match="equal"):
        same_n_self_bleu(["one", "two"], ["one"])
