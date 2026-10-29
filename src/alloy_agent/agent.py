"""One user turn: LLM ⇄ tools under hard caps, a code-filled candidate table, and the numeric verifier."""
import json
import time
from dataclasses import dataclass, field

from .llm import Chat
from .predictor import OpenPredictor
from .tools import TOOL_SPECS, dispatch
from .verify import strip_unsupported, verify

MAX_TOOL_CALLS = 6
MAX_REFINE = 2
TURN_TOKEN_CAP = 60_000
SESSION_TOKEN_CAP = 400_000

SYSTEM_PROMPT = """You are alloy-agent, a steel alloy design assistant.
You help users find steel compositions (wt%) that meet target mechanical properties.

Data scope: the predictor was trained on 312 open-data steels, mostly maraging and ultra-high-strength
steels (yield strength roughly 1000-2500 MPa, elongation 2-35%). Elements: C, Mn, Si, Cr, Ni, Mo, V, N, Nb,
Co, W, Al, Ti. Fe is the balance; never pass it. Predictions have calibrated uncertainty (mu ± 2 sigma ≈ 95%).

How to work:
- Turn the user's goal into constraints: ys/uts in MPa, el in %. Convert ksi to MPa (1 ksi = 6.895 MPa) before calling tools.
- Use `search` to find candidates meeting constraints. If none is feasible, read `binding`, then either
  relax that constraint sensibly or narrow element bounds, and search again (at most twice), or explain why
  the goal is out of reach.
- Use `pareto` when goals conflict (e.g. strength vs elongation) or the user asks about trade-offs.
  High `rank` = promising but uncertain: good candidates for the next experiment.
- Use `predict` to check a specific composition the user gives.
- Flag out-of-domain candidates honestly; never hide them.

Final answer rules:
- The app shows the candidate table itself. Write a short explanation: which candidate you recommend and why,
  the trade-offs, the binding constraint if any, and which experiments to run next.
- Never state a number that no tool returned. Do not invent compositions or properties.
- Quote tool numbers as given. Do not round, approximate ("~20%"), or compute new numbers (differences,
  percentages, gains); describe such comparisons in words instead. An automatic checker removes unsupported numbers.
- Reply in the user's language."""


@dataclass
class Session:
    messages: list[dict] = field(default_factory=list)
    last_front: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    tokens_used: int = 0


@dataclass
class TurnResult:
    text: str
    table: list[dict]
    trace: list[dict]
    issues: list[str]
    capped: bool


def _summary(name: str, result: dict) -> str:
    if "error" in result:
        return result["error"]
    if name == "search":
        s = f"{len(result['candidates'])} candidates, {result['feasible_count']} feasible"
        if b := result["binding"]:
            s += f"; binding {b['target']} short by {b['short_by']}"
        return s
    if name == "pareto":
        return f"front of {len(result['front'])}"
    p = result["pred"]
    return ", ".join(f"{t} {v['mu']}±{2 * v['sigma']:.1f}" for t, v in p.items()) + (" (OOD)" if result["ood"] else "")


def _front_context(front: list[dict]) -> str:
    rows = [{"id": c["id"], "composition": c["composition"], "mu": {t: v["mu"] for t, v in c["pred"].items()}}
            for c in front]
    return "[Previous Pareto front, for follow-ups: " + json.dumps(rows) + "]\n"


def run_turn(session: Session, user_text: str, chat: Chat, pred: OpenPredictor,
             log_path: str | None = "runs.jsonl") -> TurnResult:
    t0, start = time.time(), len(session.messages)
    if session.tokens_used >= SESSION_TOKEN_CAP:  # stop spending: no LLM call at all once the session budget is gone
        return TurnResult("Session token budget reached. Start a new session to continue.", [],
                          [{"step": 1, "tool": None, "args": None, "summary": "", "status": "cap reached"}], [], True)
    if not session.messages:
        session.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    prefix = _front_context(session.last_front) if session.last_front else ""
    prior = [{"front": session.last_front, "candidates": session.candidates}]  # earlier turns' tool numbers
    session.messages.append({"role": "user", "content": prefix + user_text})

    trace, outputs, table = [], [], []
    calls = refines = turn_tokens = 0
    capped, issues = False, []

    def ask(messages, tools):
        nonlocal turn_tokens
        msg = chat(messages, tools)
        used = msg.get("usage", {}).get("total_tokens", 0)
        turn_tokens += used
        session.tokens_used += used
        return msg

    try:
        while True:
            if calls >= MAX_TOOL_CALLS or turn_tokens >= TURN_TOKEN_CAP or session.tokens_used >= SESSION_TOKEN_CAP:
                capped = True
                trace.append({"step": len(trace) + 1, "tool": None, "args": None, "summary": "", "status": "cap reached"})
                msg = ask(session.messages + [{"role": "user", "content": "Tool budget reached. Answer now with what you have."}], [])
                text = msg["content"] or "Tool budget reached before a final answer."
                break
            msg = ask(session.messages, TOOL_SPECS)
            if not msg["tool_calls"]:
                text = msg["content"] or ""
                break
            session.messages.append({"role": "assistant", "content": msg["content"], "tool_calls": [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in msg["tool_calls"]]})
            for tc in msg["tool_calls"]:
                if calls >= MAX_TOOL_CALLS:  # every tool_call id still needs a reply
                    result = {"error": "tool budget reached; answer with what you have"}
                elif tc["name"] == "search" and refines >= MAX_REFINE:
                    calls += 1  # blocked calls still count, or a model that keeps retrying would loop forever
                    result = {"error": "refine limit reached; answer with what you have"}
                else:
                    calls += 1
                    result = dispatch(pred, tc["name"], tc["arguments"])
                    outputs.append(result)
                    if tc["name"] == "search" and "error" not in result:
                        table = session.candidates = result["candidates"]
                        refines += bool(result["binding"])
                    elif tc["name"] == "pareto" and result.get("front"):  # an empty front must not wipe the table
                        table = session.last_front = result["front"]
                    try:
                        args = json.loads(tc["arguments"])
                    except ValueError:
                        args = tc["arguments"]
                    trace.append({"step": len(trace) + 1, "tool": tc["name"], "args": args,
                                  "summary": _summary(tc["name"], result),
                                  "status": "error" if "error" in result else "ok"})
                session.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})

        if not text.strip():  # reasoning models sometimes end with no text: ask once, then fall back to code
            text = ask(session.messages + [{"role": "user", "content": "Write your final answer for the user now."}],
                       [])["content"] or ""
        if not text.strip():
            text = "No written answer was produced. Tool results: " + "; ".join(t["summary"] for t in trace if t["tool"])
        # numbers the model was given (prompt, tool specs, every user message so far) are not fabrications
        given = [SYSTEM_PROMPT, json.dumps(TOOL_SPECS)] + [m["content"] for m in session.messages if m["role"] == "user"]
        known = outputs + prior
        issues = verify(text, known, given)
        if issues:
            trace.append({"step": len(trace) + 1, "tool": None, "args": None, "summary": "; ".join(issues),
                          "status": "verifier rejected"})
            retry = session.messages + [{"role": "assistant", "content": text}, {"role": "user", "content":
                    f"Verifier: {'; '.join(issues)}. Rewrite using only numbers returned by tools."}]
            first, first_issues = text, issues
            text = ask(retry, [])["content"] or ""
            issues = verify(text, known, given)
            if not text.strip():  # reasoning models sometimes return nothing; keep the first answer, stripped
                text, issues = strip_unsupported(first, first_issues), first_issues
            elif issues:
                text = strip_unsupported(text, issues)
                trace.append({"step": len(trace) + 1, "tool": None, "args": None, "summary": "; ".join(issues),
                              "status": "unsupported numbers stripped"})
    except Exception as e:  # network/API failures end the turn visibly instead of crashing the app
        text = f"LLM error: {type(e).__name__}: {e}"

    session.messages.append({"role": "assistant", "content": text})
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": user_text, "text": text,
                                "table": table, "trace": trace, "issues": issues, "capped": capped,
                                "tokens": turn_tokens, "latency_s": round(time.time() - t0, 2),
                                "messages": session.messages[start:]}, ensure_ascii=False) + "\n")
    return TurnResult(text, table, trace, issues, capped)


def _fmt(v: dict, digits: int) -> str:
    return f"{v['mu']:.{digits}f} ± {2 * v['sigma']:.{digits}f}"


def render_table(cands: list[dict]) -> str:
    if not cands:
        return "_No candidates yet._"
    lines = ["| # | Composition (wt%, Fe balance) | YS (MPa) | UTS (MPa) | El (%) | Domain |",
             "|---|---|---|---|---|---|"]
    for c in cands:
        comp = ", ".join(f"{e.capitalize()} {w:g}" for e, w in sorted(c["composition"].items(), key=lambda x: -x[1]))
        p = c["pred"]
        lines.append(f"| {c['id']} | {comp} | {_fmt(p['ys'], 0)} | {_fmt(p['uts'], 0)} | {_fmt(p['el'], 1)} | "
                     f"{'⚠ out of domain' if c['ood'] else ''} |")
    return "\n".join(lines)
