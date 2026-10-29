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


def test_comma_list_not_merged_into_one_number():
    out = [{"pred": {"ys": {"mu": 1693.0, "sigma": 112.5}}}]  # 2σ = 225
    assert verify("YS, 2σ: 1693,225", out, []) == []


def test_fabricated_caught_among_many_values():
    # 30 candidates spaced 20 MPa apart: a 1% window would cover nearly every value in between
    out = [{"candidates": [{"pred": {"ys": {"mu": 1500.0 + 20 * i, "sigma": 50.0}}} for i in range(30)]}]
    assert verify("Candidate 3 reaches 2050 MPa.", out, []) == ["unsupported number: 2050.0"]


def test_numbers_inside_ood_reason_strings_are_supported():
    out = [{"ood": True, "ood_reasons": ["ni=25 outside [0.01, 18.9]"]}]
    assert verify("Ni 25 is outside the training range up to 18.9.", out, []) == []


def test_strip():
    t = "Yield reaches 1850 MPa."
    assert strip_unsupported(t, verify(t, OUT, [])) == "Yield reaches [?] MPa."
