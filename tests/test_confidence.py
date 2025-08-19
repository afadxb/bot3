from barchart_swing_bot.confidence import compute_confidence, is_actionable


def test_confidence_actionable():
    features = {"wtd_alpha": 3, "momentum": 1, "inst_score": 1}
    assert compute_confidence(features) == 80
    assert is_actionable(features)


def test_confidence_not_actionable():
    features = {"wtd_alpha": 1, "momentum": 0, "inst_score": 1}
    assert compute_confidence(features) == 40
    assert not is_actionable(features)
