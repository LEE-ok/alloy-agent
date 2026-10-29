import json

from alloy_agent.agent import MAX_TOOL_CALLS, Session, render_table, run_turn
from alloy_agent.predictor import OpenPredictor

P = OpenPredictor()


def scripted(*replies):
    it = iter(replies)
    return lambda messages, tools: next(it)


def call(name, args):
    return {"content": None, "tool_calls": [{"id": "c1", "name": name, "arguments": json.dumps(args)}],
            "usage": {"total_tokens": 100}}


def say(text):
    return {"content": text, "tool_calls": [], "usage": {"total_tokens": 50}}


def test_search_then_answer():
    s = Session([], [], [])
    r = run_turn(s, "YS at least 1200 MPa",
                 scripted(call("search", {"constraints": [{"target": "ys", "min": 1200}]}), say("Here are options.")), P, None)
    assert r.table and r.issues == [] and r.trace[0]["tool"] == "search"
    assert "⚠" in render_table([dict(r.table[0], ood=True)])


def test_tool_cap():
    r = run_turn(Session([], [], []), "x", lambda m, t: call("predict", {"composition": {"ni": 18}}), P, None)
    assert r.capped and sum(1 for t in r.trace if t["tool"]) == MAX_TOOL_CALLS


def test_verifier_retry_then_strip():
    r = run_turn(Session([], [], []), "x", scripted(say("Yield 9999 MPa."), say("Still 9999 MPa.")), P, None)
    assert "9999" not in r.text and r.issues


def test_system_prompt_numbers_are_supported():
    r = run_turn(Session(), "x", scripted(say("The data covers about 1000-2500 MPa yield.")), P, None)
    assert r.issues == [] and "1000" in r.text


def test_empty_retry_keeps_stripped_original():
    r = run_turn(Session(), "x", scripted(say("Yield 9999 MPa."), say("")), P, None)
    assert r.text == "Yield [?] MPa."


def test_empty_front_keeps_previous_table():
    impossible = {"objectives": [{"target": "ys"}, {"target": "el"}], "constraints": [{"target": "ys", "min": 3000}]}
    r = run_turn(Session(), "x", scripted(call("search", {"constraints": [{"target": "ys", "min": 1200}]}),
                                          call("pareto", impossible), say("Done.")), P, None)
    assert r.table and r.table[0]["score"] >= 0


def test_followup_may_quote_previous_front():
    s = Session()
    run_turn(s, "trade-off", scripted(call("pareto", {"objectives": [{"target": "ys"}, {"target": "el"}]}),
                                      say("Front ready.")), P, None)
    ys = s.last_front[0]["pred"]["ys"]["mu"]
    r = run_turn(s, "more elongation", scripted(say(f"Earlier option #1 had {ys} MPa.")), P, None)
    assert r.issues == []


def test_followup_may_quote_earlier_user_numbers():
    s = Session()
    run_turn(s, "Yield strength at least 1500 MPa", scripted(say("OK.")), P, None)
    r = run_turn(s, "keep nickel low", scripted(say("Still targeting 1500 MPa.")), P, None)
    assert r.issues == []


def test_llm_error_ends_turn_gracefully():
    def boom(m, t):
        raise TimeoutError("timeout")
    r = run_turn(Session([], [], []), "x", boom, P, None)
    assert "error" in r.text.lower()


def test_multiturn_keeps_front():
    s = Session([], [], [])
    run_turn(s, "trade-off", scripted(call("pareto", {"objectives": [{"target": "ys"}, {"target": "el"}]}),
                                      say("Front ready.")), P, None)
    assert s.last_front
