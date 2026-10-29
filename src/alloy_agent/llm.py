"""OpenAI-compatible chat client chosen by env (Baseten Muse Spark Open by default, or Meta Model API)."""
import os
import time
from typing import Callable

import openai

Chat = Callable[[list[dict], list[dict]], dict]

DEFAULT_BASE_URL = "https://inference.baseten.co/v1"
DEFAULT_MODEL = "meta-models/muse-spark-open-1026"
MAX_TOKENS = 4000  # reasoning tokens count against this
BACKENDS = {  # label -> (base_url, model); shared by the app and the eval
    "Muse Spark Open (Baseten)": (DEFAULT_BASE_URL, DEFAULT_MODEL),
    "Muse Spark 1.3 (Meta)": ("https://api.meta.ai/v1", "muse-spark-1.3"),
}


def _key(base_url: str) -> str:
    """The provider's own key first, so switching backends never sends one provider's key to another."""
    own = "BASETEN_API_KEY" if "baseten.co" in base_url else "MODEL_API_KEY" if "api.meta.ai" in base_url else None
    key = (own and os.environ.get(own)) or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError(f"no API key in env ({own or 'LLM_API_KEY'})")
    return key


def make_chat(base_url: str | None = None, model: str | None = None) -> Chat:
    base_url = base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    client = openai.OpenAI(base_url=base_url, api_key=_key(base_url), timeout=120, max_retries=0)

    def chat(messages: list[dict], tools: list[dict]) -> dict:
        kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
        for attempt in (0, 1):
            try:
                r = client.chat.completions.create(model=model, messages=messages, max_tokens=MAX_TOKENS, **kwargs)
                break
            except (openai.APIError, TimeoutError):
                if attempt:
                    raise
                time.sleep(2)
        m = r.choices[0].message
        return {"content": m.content,
                "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                               for tc in m.tool_calls or []],
                "usage": {"total_tokens": r.usage.total_tokens if r.usage else 0}}

    chat.model = model
    return chat
