"""Gradio UI: chat + spec upload on the left; agent trace, Pareto scatter and candidate table on the right.

REPLAY=path/to/runs.jsonl replays saved turns in order instead of calling an LLM (demo fallback).
"""
import json
import os

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from alloy_agent.agent import Session, TurnResult, render_table, run_turn
from alloy_agent.intake import constraints_to_prompt, extract_constraints
from alloy_agent.llm import BACKENDS, make_chat
from alloy_agent.predictor import OpenPredictor

load_dotenv()
PRED = OpenPredictor()
REPLAY = os.environ.get("REPLAY")
_replay = iter(json.loads(l) for l in open(REPLAY, encoding="utf-8")) if REPLAY else None

FOOTER = ("Predictions from a model trained on 312 open steels (MIT, Figshare Steel Strength Data). "
          "±2σ calibrated to ~95% out-of-fold coverage. Out-of-domain candidates are flagged, not hidden.")


def scatter_df(table: list[dict]) -> pd.DataFrame:
    # Plain field names: Vega mis-reads "(...)" and "%" in field names. Explicit dtypes: empty object
    # columns make Gradio pick nominal axes.
    return pd.DataFrame([{"ys": c["pred"]["ys"]["mu"], "el": c["pred"]["el"]["mu"], "id": c["id"],
                          "domain": "out of domain" if c["ood"] else "in domain"} for c in table],
                        columns=["ys", "el", "id", "domain"]).astype({"ys": float, "el": float})


def respond(text, history, session, backend):
    if not text.strip():
        return gr.skip()
    session = session or Session()
    if _replay:
        d = next(_replay, None)
        r = TurnResult(d["text"], d["table"], d["trace"], d["issues"], d["capped"]) if d else \
            TurnResult("Replay finished.", [], [], [], False)
    else:
        r = run_turn(session, text, make_chat(*BACKENDS[backend]), PRED)
    history = history + [{"role": "user", "content": text}, {"role": "assistant", "content": r.text}]
    return "", history, session, r.trace, scatter_df(r.table), render_table(r.table)


def upload(path):
    if not path:
        return gr.skip()
    try:
        return constraints_to_prompt(extract_constraints(path))
    except Exception as e:
        gr.Warning(f"Could not read the spec sheet: {e}")
        return gr.skip()


with gr.Blocks(title="alloy-agent") as demo:
    gr.Markdown("# alloy-agent\nDescribe the steel you need, or upload a spec sheet. "
                "The agent searches compositions, shows trade-offs and only states numbers its tools returned.")
    session = gr.State(None)
    with gr.Row():
        with gr.Column(scale=5):
            chat = gr.Chatbot(height=520, label="Agent")
            box = gr.Textbox(placeholder="e.g. Yield strength at least 1800 MPa and elongation at least 10%",
                             show_label=False, submit_btn=True)
            with gr.Row():
                spec = gr.File(label="Spec sheet (PDF / image)", file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                               type="filepath")
                backend = gr.Radio(list(BACKENDS), value=next(iter(BACKENDS)), label="Model")
        with gr.Column(scale=6):
            # no color encoding: Gradio's color scale intermittently rendered points transparent; OOD is in tooltip + table
            plot = gr.ScatterPlot(scatter_df([]), x="ys", y="el", x_title="Yield strength (MPa)",
                                  y_title="Elongation (%)", tooltip=["id", "ys", "el", "domain"],
                                  title="Candidates: strength vs elongation (mean)", height=300)
            table = gr.Markdown(render_table([]))
            with gr.Accordion("Agent trace", open=True):
                trace = gr.JSON(label="tool → result → verifier")
    gr.Markdown(f"<small>{FOOTER}</small>")

    box.submit(respond, [box, chat, session, backend], [box, chat, session, trace, plot, table])
    spec.upload(upload, spec, box)

if __name__ == "__main__":
    demo.launch()
