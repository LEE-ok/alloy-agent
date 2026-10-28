from alloy_agent.verify import strip_unsupported, verify

OUT = [{"candidates": [{"pred": {"ys": {"mu": 1523.4, "sigma": 61.2}}}]}]


def test_rounding_ok():
    assert verify("Yield is about 1523 MPa (±2σ, σ≈61).", OUT, []) == []


def test_fabricated_caught():
    assert verify("Yield reaches 1850 MPa.", OUT, []) == ["unsupported number: 1850.0"]


def test_user_numbers_whitelisted():
    assert verify("You asked for 2,000 MPa; best is 1523.", OUT, ["at least 2000 MPa"]) == []


def test_strip():
    t = "Yield reaches 1850 MPa."
    assert strip_unsupported(t, verify(t, OUT, [])) == "Yield reaches [?] MPa."
