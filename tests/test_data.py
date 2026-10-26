from alloy_agent.data import ELEMENTS, load


def test_load():
    rows = load()
    assert len(rows) == 312
    assert all(r[e] is not None for r in rows for e in ELEMENTS)
    assert sum(r["elongation"] is None for r in rows) == 9
