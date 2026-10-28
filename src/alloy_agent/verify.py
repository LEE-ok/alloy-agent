"""Every number in the agent's prose must come from a tool output, the user's own words, or a tiny allowlist."""
import re

NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")  # no leading '-': hyphens in "mu-2σ" or "18Ni-9Co" are not signs
ALLOWED = {0, 1, 2, 3, 4, 5, 95}  # ordinals, "2σ", "95%"
TOL = 0.01


def numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUM.findall(text)]


def _walk(x, out: list[float]):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            _walk(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _walk(v, out)


def _supported(x: float, known: list[float]) -> bool:
    return x in ALLOWED or any(abs(x - v) <= TOL * max(abs(v), 1) for v in known)


def verify(text: str, tool_outputs: list[dict], user_texts: list[str]) -> list[str]:
    known: list[float] = []
    _walk(tool_outputs, known)
    for t in user_texts:
        known += numbers(t)
    return [f"unsupported number: {x}" for x in numbers(text) if not _supported(x, known)]


def strip_unsupported(text: str, issues: list[str]) -> str:
    bad = {float(i.rsplit(":", 1)[1]) for i in issues if i.startswith("unsupported number:")}
    return NUM.sub(lambda m: "[?]" if float(m.group().replace(",", "")) in bad else m.group(), text)
