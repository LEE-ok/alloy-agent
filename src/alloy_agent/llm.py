"""OpenAI-compatible chat client chosen by env (Baseten Muse Spark Open by default, or Meta Model API)."""
import os
import time
from typing import Callable

import openai

Chat = Callable[[list[dict], list[dict]], dict]

DEFAULT_BASE_URL = "https://inference.baseten.co/v1"
DEFAULT_MODEL = "meta-models/muse-spark-open-1026"
MAX_TOKENS = 4000  # reasoning tokens count against this


def _key(base_url: str) -> str:
    key = os.environ.get("LLM_API_KEY") or os.environ.get(
        "BASETEN_API_KEY" if "baseten" in base_url else "MODEL_API_KEY")
    if not key:
        raise RuntimeError("no API key in env (LLM_API_KEY, BASETEN_API_KEY or MODEL_API_KEY)")
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
