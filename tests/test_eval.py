import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import score_turn  # noqa: E402

from alloy_agent.agent import TurnResult  # noqa: E402


def test_score_feasible():
    r = TurnResult("ok", [{"score": 0.1, "ood": False}], [{"tool": "search"}], [], False)
    assert score_turn(r, "feasible")["success"]


def test_score_infeasible_needs_binding_mention():
    r = TurnResult("Yield is the binding constraint (ys).", [{"score": -0.2, "ood": False}],
                   [{"tool": "search", "summary": "binding ys"}], [], False)
    assert score_turn(r, "infeasible")["success"]


def test_score_tradeoff_needs_front():
    r = TurnResult("Here is the front.", [{"rank": 2.0, "ood": False}], [{"tool": "pareto", "summary": "front of 1"}], [], False)
    assert score_turn(r, "tradeoff")["success"]
    assert not score_turn(TurnResult("none", [], [], [], False), "tradeoff")["success"]


def test_score_ood_needs_flag_reported():
    r = TurnResult("This composition is out of domain.", [], [{"tool": "predict", "summary": "ys 1500±200 (OOD)"}], [], False)
    assert score_turn(r, "ood")["success"]
