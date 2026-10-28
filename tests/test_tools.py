import pytest
from pydantic import ValidationError

from alloy_agent.predictor import OpenPredictor
from alloy_agent.tools import Constraint, Objective, dispatch, pareto, search

P = OpenPredictor()


def test_search_easy_feasible():
    r = search(P, [Constraint(target="ys", min=1200)])
    assert r["feasible_count"] > 0 and r["candidates"][0]["score"] >= 0
    assert r["binding"] is None
    assert all(type(v) is float for v in r["candidates"][0]["composition"].values())  # plain JSON floats


def test_search_impossible_binding():
    r = search(P, [Constraint(target="ys", min=3000)])
    assert r["feasible_count"] == 0
    assert r["binding"]["target"] == "ys" and r["binding"]["short_by"] > 0


def test_pareto_non_dominated():
    f = pareto(P, [Objective(target="ys"), Objective(target="el")])["front"]
    pts = [(c["pred"]["ys"]["mu"], c["pred"]["el"]["mu"]) for c in f]
    assert not any(a[0] >= b[0] and a[1] >= b[1] and a != b for a in pts for b in pts)
    assert all("rank" in c for c in f)


def test_pareto_empty_explains():
    r = pareto(P, [Objective(target="ys"), Objective(target="el")], [Constraint(target="ys", min=3000)])
    assert r["front"] == [] and "no candidate" in r["note"]


def test_constraint_validation():
    with pytest.raises(ValidationError):
        Constraint(target="ys", min=2000, max=1000)
    with pytest.raises(ValidationError):
        Constraint(target="ys")
    assert "error" in dispatch(P, "search", '{"constraints": []}')


def test_dispatch_element_keys():
    assert "error" not in dispatch(P, "predict", '{"composition": {"Ni": 18, "Fe": 70}}')
    assert "unknown element" in dispatch(P, "predict", '{"composition": {"nickel": 18}}')["error"]


def test_dispatch_bad_json():
    assert "error" in dispatch(P, "search", "{not json")
    assert "error" in dispatch(P, "nope", "{}")


def test_search_bounds_outside_training_flagged():
    lo, hi = P.element_ranges["ni"]
    r = search(P, [Constraint(target="ys", min=1000)], bounds={"ni": (hi + 5, hi + 10)})
    assert r["candidates"] and all(c["ood"] for c in r["candidates"])
