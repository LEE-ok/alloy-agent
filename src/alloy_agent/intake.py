"""Read a spec sheet (PDF or image) with Muse Spark and turn it into Constraints.

Meta Model API input format (Responses API): PDFs are uploaded with files.create(purpose="user_data") and sent
as {"type": "input_file", "file_id"}; images go inline as {"type": "input_image", "image_url": data URL}.
"""
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Callable

import openai
from pydantic import BaseModel, Field, field_validator

from .data import ELEMENTS
from .tools import Constraint

INTAKE_MODEL = "muse-spark-1.3"
NAMES = {"ys": "yield strength", "uts": "tensile strength", "el": "elongation"}
UNITS = {"ys": "MPa", "uts": "MPa", "el": "%"}

PROMPT = """Read this material specification and extract the mechanical property requirements for a steel.
Return ONLY a JSON object, no prose, with this schema:
{"targets": [{"target": "ys" | "uts" | "el", "min": number or null, "max": number or null}],
 "element_caps": {"<element symbol, lowercase>": max wt%}}
ys = 0.2% yield strength in MPa, uts = ultimate tensile strength in MPa, el = elongation in %.
Convert ksi to MPa (1 ksi = 6.895 MPa). Use only elements from: """ + ", ".join(ELEMENTS) + """.
Leave out anything the document does not state."""


class Constraints(BaseModel):
    targets: list[Constraint] = Field(min_length=1)
    element_caps: dict[str, float] = {}

    @field_validator("element_caps")
    @classmethod
    def _known(cls, caps):
        caps = {k.strip().lower(): v for k, v in caps.items()}
        unknown = set(caps) - set(ELEMENTS)
        if unknown:
            raise ValueError(f"unknown element(s): {sorted(unknown)}")
        return caps


def _meta_chat_raw() -> Callable[[list[dict]], str]:
    client = openai.OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=120)

    def chat_raw(content: list[dict]) -> str:
        parts = []
        for p in content:
            if p["type"] != "local_file":
                parts.append(p)
                continue
            path = Path(p["path"])
            if path.suffix.lower() == ".pdf":
                with path.open("rb") as f:
                    parts.append({"type": "input_file", "file_id": client.files.create(file=f, purpose="user_data").id})
            else:
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
                parts.append({"type": "input_image",
                              "image_url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()})
        r = client.responses.create(model=INTAKE_MODEL, input=[{"role": "user", "content": parts}], max_output_tokens=4000)
        return r.output_text

    return chat_raw


def extract_constraints(path: str, chat_raw: Callable[[list[dict]], str] | None = None) -> Constraints:
    raw = (chat_raw or _meta_chat_raw())([{"type": "input_text", "text": PROMPT}, {"type": "local_file", "path": path}])
    m = re.search(r"\{.*\}", raw or "", re.S)  # tolerate code fences or stray prose around the JSON
    try:
        return Constraints(**json.loads(m.group() if m else raw))
    except Exception as e:
        raise ValueError(f"could not read constraints from the document: {e}") from e


def constraints_to_prompt(c: Constraints) -> str:
    parts = []
    for t in c.targets:
        bounds = [f"at least {t.min:g} {UNITS[t.target]}" if t.min is not None else "",
                  f"at most {t.max:g} {UNITS[t.target]}" if t.max is not None else ""]
        parts.append(f"{NAMES[t.target]} " + " and ".join(b for b in bounds if b))
    text = "Find steel compositions with " + ", ".join(parts) + "."
    if c.element_caps:
        text += " Element limits: " + ", ".join(f"{e.capitalize()} at most {v:g} wt%" for e, v in c.element_caps.items()) + "."
    return text
