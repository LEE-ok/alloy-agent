from alloy_agent.llm import _key

BASETEN = "https://inference.baseten.co/v1"
META = "https://api.meta.ai/v1"


def test_each_provider_gets_its_own_key(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "b-key")
    monkeypatch.setenv("MODEL_API_KEY", "m-key")
    monkeypatch.setenv("LLM_API_KEY", "generic")
    assert _key(BASETEN) == "b-key" and _key(META) == "m-key"
    assert _key("https://other.example/v1") == "generic"
