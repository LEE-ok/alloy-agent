import app


def test_missing_api_key_shows_message_not_crash(monkeypatch):
    for k in ("BASETEN_API_KEY", "MODEL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    out = app.respond("Yield at least 1500 MPa", [], None, "Muse Spark Open (Baseten)")
    assert "API key" in out[1][-1]["content"]
