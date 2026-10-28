import pytest

from alloy_agent.predictor import OpenPredictor

P = OpenPredictor()  # fit once for the module


def test_shapes():
    p = P.predict([{"ni": 18, "co": 9, "mo": 5, "ti": 0.7}])[0]
    assert set(p.mu) == set(p.sigma) == {"ys", "uts", "el"}
    assert all(s > 0 for s in p.sigma.values())


def test_fe_and_case_ignored():
    a = P.predict([{"ni": 18, "co": 9}])[0]
    b = P.predict([{"Ni": 18, "CO": 9, "Fe": 70}])[0]
    assert a.mu == b.mu


def test_unknown_element():
    with pytest.raises(ValueError, match="unknown element"):
        P.normalize({"cu": 1})


def test_calibrated_coverage():
    assert all(c >= 0.9 for c in P.coverage.values())


def test_ood_flag():
    assert P.predict([{"ni": 50}])[0].ood
    assert not P.predict([P.medians])[0].ood
