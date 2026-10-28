from alloy_agent.verify import strip_unsupported, verify

OUT = [{"candidates": [{"pred": {"ys": {"mu": 1523.4, "sigma": 61.2}}}]}]


def test_rounding_ok():
    assert verify("Yield is about 1523 MPa (±2σ, σ≈61).", OUT, []) == []


def test_fabricated_caught():
    assert verify("Yield reaches 1850 MPa.", OUT, []) == ["unsupported number: 1850.0"]


def test_user_numbers_whitelisted():
    assert verify("You asked for 2,000 MPa; best is 1523.", OUT, ["at least 2000 MPa"]) == []


def test_derived_interval_bounds_ok():
    # 1523.4 ± 2·61.2 → 1400.6 .. 1645.8, half-width 122.4
    assert verify("95% interval 1401-1646 MPa (±122).", OUT, []) == []


def test_rounded_to_written_precision_ok():
    out = [{"composition": {"ti": 1.473, "ni": 17.164}}]
    assert verify("About 1.5% Ti and 17% Ni.", out, []) == []
    assert verify("About 1.7% Ti.", out, []) == ["unsupported number: 1.7"]


def test_strip():
    t = "Yield reaches 1850 MPa."
    assert strip_unsupported(t, verify(t, OUT, [])) == "Yield reaches [?] MPa."
