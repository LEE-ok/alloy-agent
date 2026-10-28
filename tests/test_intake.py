import pytest

from alloy_agent.intake import constraints_to_prompt, extract_constraints


def test_parse_valid():
    raw = '{"targets": [{"target": "ys", "min": 1700}, {"target": "el", "min": 8}], "element_caps": {"co": 10}}'
    c = extract_constraints("demo/spec_table.png", chat_raw=lambda m: raw)
    assert c.targets[0].min == 1700 and c.element_caps == {"co": 10}
    assert "1700" in constraints_to_prompt(c)


def test_parse_invalid():
    with pytest.raises(ValueError):
        extract_constraints("demo/spec_table.png", chat_raw=lambda m: "sorry")
