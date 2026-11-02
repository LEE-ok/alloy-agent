import app


def test_missing_api_key_shows_message_not_crash(monkeypatch):
    for k in ("BASETEN_API_KEY", "MODEL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    out = app.respond("Yield at least 1500 MPa", [], None, "Muse Spark Open (Baseten)")
    assert "API key" in out[1][-1]["content"]


def test_public_demo_stops_calling_llm_after_turn_limit(monkeypatch):
    monkeypatch.setattr(app, "LIVE_TURN_LIMIT", 3)
    monkeypatch.setattr(app, "_live_turns", 3)
    monkeypatch.setattr(app, "make_chat", lambda *a: (_ for _ in ()).throw(AssertionError("LLM used past the limit")))
    out = app.respond("Yield at least 1500 MPa", [], None, "Muse Spark Open (Baseten)")
    assert "usage limit" in out[1][-1]["content"]
