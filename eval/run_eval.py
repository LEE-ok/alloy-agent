"""Run the eval queries through the agent and write one CSV per backend; --plot compares backends.

    python eval/run_eval.py --backend baseten [--limit N]
    python eval/run_eval.py --plot
"""
import argparse
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from alloy_agent.agent import Session, TurnResult, run_turn
from alloy_agent.llm import BACKENDS, make_chat

HERE = Path(__file__).resolve().parent
SHORT = {"baseten": "Muse Spark Open (Baseten)", "meta": "Muse Spark 1.3 (Meta)"}
FIELDS = ["id", "expect", "success", "clean", "tool_calls", "latency_s", "tokens", "ood_rate", "verifier_failures", "capped"]
NAMES = {"ys": "yield", "uts": "tensile", "el": "elongation"}


def score_turn(r: TurnResult, expect: str) -> dict:
    """success = the task outcome is right; clean = the verifier had nothing left to strip from the answer."""
    tools = [t for t in r.trace if t.get("tool")]
    summaries = " ".join(t.get("summary", "") for t in tools)
    text = r.text.lower()
    if expect == "feasible":
        success = bool(r.table) and r.table[0].get("score", -1) >= 0
    elif expect == "infeasible":
        bound = [t for t in NAMES if f"binding {t}" in summaries]
        success = bool(bound) and any(t in text or NAMES[t] in text for t in bound)
    elif expect == "tradeoff":
        success = any(t["tool"] == "pareto" and t.get("summary", "").startswith("front of ")
                      and t["summary"] != "front of 0" for t in tools)
    else:  # ood: the flag came back and the answer says so
        success = "(OOD)" in summaries and ("domain" in text or "ood" in text)
    return {"success": success, "clean": not r.issues, "tool_calls": len(tools),
            "ood_rate": sum(c["ood"] for c in r.table) / len(r.table) if r.table else 0.0,
            "verifier_failures": sum(t.get("status") in ("verifier rejected", "unsupported numbers stripped") for t in r.trace),
            "capped": r.capped}


def run(backend: str, limit: int | None):
    from alloy_agent.predictor import OpenPredictor
    pred, chat = OpenPredictor(), make_chat(*BACKENDS[SHORT[backend]])
    queries = [json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = HERE / f"results_{backend}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for q in queries[:limit]:
            s, t0 = Session(), time.time()
            log = str(HERE / f"runs_{backend}.jsonl")
            r = run_turn(s, q["query"], chat, pred, log)
            earlier_calls = 0
            if q["followup"]:  # the follow-up turn is what gets scored; tool calls count both turns
                earlier_calls = score_turn(r, q["expect"])["tool_calls"]
                r = run_turn(s, q["followup"], chat, pred, log)
            score = score_turn(r, q["expect"])
            score["tool_calls"] += earlier_calls
            row = {"id": q["id"], "expect": q["expect"], **score,
                   "latency_s": round(time.time() - t0, 1), "tokens": s.tokens_used}
            w.writerow(row)
            f.flush()
            print(row)
    print("wrote", out)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {b: list(csv.DictReader((HERE / f"results_{b}.csv").open(encoding="utf-8")))
            for b in SHORT if (HERE / f"results_{b}.csv").exists()}
    metrics = [("task success rate", lambda rs: sum(r["success"] == "True" for r in rs) / len(rs)),
               ("clean answer rate", lambda rs: sum(r["clean"] == "True" for r in rs) / len(rs)),
               ("mean tool calls", lambda rs: sum(int(r["tool_calls"]) for r in rs) / len(rs)),
               ("mean latency (s)", lambda rs: sum(float(r["latency_s"]) for r in rs) / len(rs))]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.2))
    for ax, (name, fn) in zip(axes, metrics):
        vals = [fn(rs) for rs in rows.values()]
        ax.bar([SHORT[b] for b in rows], vals, color=["#4c9be8", "#e8794c"][:len(vals)])
        ax.set_title(name)
        ax.tick_params(axis="x", labelsize=7)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "compare.png", dpi=150)
    print("wrote", HERE / "compare.png")


if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(SHORT))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    plot() if a.plot else run(a.backend, a.limit)
