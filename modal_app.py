"""Public demo on Modal.

One-time setup (by the repo owner):
    modal setup                                                     # browser login
    modal secret create alloy-agent-keys BASETEN_API_KEY=... MODEL_API_KEY=...
Deploy:
    modal deploy modal_app.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("fastapi[standard]")
    .add_local_dir(".", "/root/app", ignore=[".venv", ".venv-rec", ".git", "data", "demo/out", ".claude",
                                             ".superpowers", "eval/runs_*.jsonl", "runs.jsonl", ".env"])
)
app = modal.App("alloy-agent", image=image)


@app.function(secrets=[modal.Secret.from_name("alloy-agent-keys")],
              env={"LIVE_TURN_LIMIT": "300"},  # caps credit spend from the public URL
              max_containers=1,  # Gradio sessions are in-process state
              scaledown_window=600, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    import os
    import sys

    os.chdir("/root/app")
    sys.path[:0] = ["/root/app", "/root/app/src"]
    import gradio as gr
    from fastapi import FastAPI

    import app as alloy  # trains the predictor (~20 s) on container start

    return gr.mount_gradio_app(FastAPI(), alloy.demo, path="/")
