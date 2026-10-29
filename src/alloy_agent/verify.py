"""Every number in the agent's prose must come from a tool output, the user's own words, or a tiny allowlist."""
import re

NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")  # no leading '-': hyphens in "mu-2σ" or "18Ni-9Co" are not signs
ALLOWED = {0, 1, 2, 3, 4, 5, 95}  # ordinals, "2σ", "95%"


def numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUM.findall(text)]


def _walk(x, out: list[float]):
    if isinstance(x, bool):
        return
    if isinstance(x, str):  # e.g. ood_reasons "ni=25 outside [0.01, 18.9]"
        out += numbers(x)
    elif isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, dict):
        if isinstance(x.get("mu"), (int, float)) and isinstance(x.get("sigma"), (int, float)):
            mu, s = x["mu"], x["sigma"]
            out += [mu - 2 * s, mu + 2 * s, 2 * s]  # derived 95% bounds, recomputed here rather than trusted
        for v in x.values():
            _walk(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _walk(v, out)


def _supported(s: str, known: list[float]) -> bool:
    """A known value rounded to the precision the number was written with ("1523" for 1523.4, "1.5" for 1.473).

    No relative tolerance: with a full Pareto front in context, a 1% window per value covers most of the
    strength range and would let invented numbers through.
    """
    x = float(s.replace(",", ""))
    digits = len(s.split(".")[1]) if "." in s else 0
    half_step = 0.5 * 10 ** -digits + 1e-9
    if x in ALLOWED or any(abs(x - v) <= half_step for v in known):
        return True
    # "1693,225" may be a list, not one thousands-separated number: accept if every part is supported
    return "," in s and all(_supported(p, known) for p in s.split(","))


def verify(text: str, tool_outputs: list[dict], user_texts: list[str]) -> list[str]:
    known: list[float] = []
    _walk(tool_outputs, known)
    for t in user_texts:
        known += numbers(t)
    return [f"unsupported number: {float(s.replace(',', ''))}" for s in NUM.findall(text) if not _supported(s, known)]


def strip_unsupported(text: str, issues: list[str]) -> str:
    bad = {float(i.rsplit(":", 1)[1]) for i in issues if i.startswith("unsupported number:")}
    return NUM.sub(lambda m: "[?]" if float(m.group().replace(",", "")) in bad else m.group(), text)
