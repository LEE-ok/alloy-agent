"""Record the 3-minute demo video: Kokoro narration + Playwright screen capture of the app in replay mode.

Run with the recording venv (.venv-rec: kokoro, soundfile, playwright, imageio-ffmpeg):
    .venv/Scripts/python demo/record.py --make-scenario       # once, live LLM calls -> demo/scenario.jsonl
    .venv-rec/Scripts/python demo/record.py                   # -> demo/out/alloy-agent-demo.mp4 (+ .srt)
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "out"
SCENARIO = ROOT / "demo" / "scenario.jsonl"
SPEC = ROOT / "demo" / "spec_landing_gear.pdf"
APP_PY = ROOT / ".venv" / "Scripts" / "python.exe"  # the app runs in the project venv
FOLLOWUP = "I can accept around 1500 MPa yield if elongation goes up. Give me more elongation."
VOICE, RATE, W, H = "am_michael", 24000, 1920, 1080
VW, VH = 1440, 810  # browser viewport; cards are laid out at 1920 and zoomed by VW / W

# (scene id, narration, action). Actions: card:<html>, upload, submit, trace, followup, results, close
SCENES = [
    ("title", "Everything you see here was built this week, from an empty repository, for the Meta Global AI "
              "Developer Hackathon. This is alloy-agent.", "card:title"),
    ("problem", "Designing a steel is slow. Engineers guess a composition, test it, and repeat. alloy-agent lets you "
                "state the properties you need, and an agent built on Muse Spark searches compositions for you, "
                "with honest uncertainty.", "card:problem"),
    ("upload", "Here I upload a supplier spec sheet, as a PDF. Muse Spark 1.3 reads the document and extracts the "
               "requirements: a minimum yield strength, a minimum elongation, and a cap on cobalt.", "upload"),
    ("search", "The agent runs on the open-weight Muse Spark model, served by Baseten. It calls a search tool that "
               "scores every candidate by a conservative margin: the mean, minus two calibrated standard deviations. "
               "No composition passes, and the agent says exactly why: elongation is the binding constraint. So "
               "instead of inventing an alloy, it maps the trade-off with a Pareto tool.", "submit"),
    ("trace", "On the right is the full trace: every tool call, its result, and the verifier. Every number in the "
              "answer must come from a tool. Anything the model makes up is caught, and removed.", "trace"),
    ("followup", "Now I relax the strength target. The agent continues from the previous trade-off front and searches "
                 "around fifteen hundred megapascals. Mean elongation goes up, and its uncertainty shrinks. It "
                 "recommends which compositions to test first, and every candidate stays inside the training domain.",
     "followup"),
    ("results", "We evaluated twenty queries on both backends. Both reached twenty out of twenty task success, and "
                "the open-weight model was faster. The predictor's uncertainty bands cover ninety-five percent of "
                "held-out steels.", "results"),
    ("close", "alloy-agent. An agent that knows what it doesn't know. Open source, under the MIT license.", "close"),
]

CARD_CSS = """body{margin:0;width:1920px;height:1080px;background:#0f1115;color:#e8eaed;font-family:Segoe UI,Arial;
display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center}
h1{font-size:96px;margin:0 0 24px}p{font-size:40px;max-width:1500px;line-height:1.4;color:#b8bcc4}
.tag{font-size:30px;color:#4c9be8;margin-top:40px}img{width:1700px;background:#fff;border-radius:12px}"""


def card(key: str) -> str:
    body = {
        "title": "<h1>alloy-agent</h1><p>Built this week, from an empty repository, for the Meta Global AI "
                 "Developer Hackathon.</p><div class='tag'>Muse Spark 1.3 · Muse Spark Open (Baseten)</div>",
        "problem": "<h1>State the steel you need.</h1><p>The agent searches compositions, shows the trade-offs, "
                   "and only states numbers its tools returned.</p>",
        "results": "<h1 style='font-size:64px'>20 / 20 task success on both backends</h1><img src='data:image/png;base64,"
                   + base64.b64encode((ROOT / "eval" / "compare.png").read_bytes()).decode()
                   + "'><p style='font-size:30px'>±2σ bands cover 95% of held-out steels (5-fold out-of-fold)</p>",
        "close": "<h1>alloy-agent</h1><p>An agent that knows what it doesn't know.</p>"
                 "<div class='tag'>MIT license · open data (Figshare Steel Strength, MIT)</div>",
    }[key]
    return f"<html style='zoom:{VW / W}'><head><style>{CARD_CSS}</style></head><body>{body}</body></html>"


def make_scenario():
    """Run the real agent once (live API) and save its turns for replay."""
    sys.path.insert(0, str(ROOT / "src"))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from alloy_agent.agent import Session, run_turn
    from alloy_agent.intake import constraints_to_prompt, extract_constraints
    from alloy_agent.llm import BACKENDS, make_chat
    from alloy_agent.predictor import OpenPredictor

    first = constraints_to_prompt(extract_constraints(str(SPEC)))
    pred, chat, s = OpenPredictor(), make_chat(*BACKENDS["Muse Spark Open (Baseten)"]), Session()
    SCENARIO.unlink(missing_ok=True)
    for q in (first, FOLLOWUP):
        r = run_turn(s, q, chat, pred, str(SCENARIO))
        print(f"--- {q}\n{r.text}\n{[(t['tool'], t['status'], t['summary']) for t in r.trace]}\n")


def narrate() -> list:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    pipe, clips = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M"), []
    for sid, text, _ in SCENES:
        path = OUT / f"vo_{sid}.wav"
        if not path.exists():
            sf.write(path, np.concatenate([a for _, _, a in pipe(text, voice=VOICE)]), RATE)
        clips.append(sf.read(path)[0])
    return clips


def record(clips) -> tuple[Path, list[float]]:
    from playwright.sync_api import sync_playwright
    env = {**os.environ, "REPLAY": str(SCENARIO), "PYTHONIOENCODING": "utf-8"}
    app = subprocess.Popen([str(APP_PY), "app.py"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):  # the predictor trains at start-up
            try:
                urllib.request.urlopen("http://127.0.0.1:7860", timeout=1)
                break
            except OSError:
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            # record at 1440x810; mux() upscales to 1080p, so app text ends up ~1.33x larger (Playwright pads
            # instead of scaling). en-US keeps Gradio's own UI strings English.
            ctx = browser.new_context(viewport={"width": VW, "height": VH}, record_video_dir=str(OUT),
                                      record_video_size={"width": VW, "height": VH}, color_scheme="dark",
                                      locale="en-US")
            page = ctx.new_page()
            t0, starts = time.time(), []
            box = None
            for (sid, _, action), clip in zip(SCENES, clips):
                starts.append(time.time() - t0)
                end = time.time() + len(clip) / RATE + 0.8
                if action.startswith("card:") or action in ("results", "close"):
                    page.set_content(card(action.split(":")[-1]))
                elif action == "upload":
                    page.goto("http://127.0.0.1:7860")
                    page.wait_for_selector("textarea")
                    box = page.locator("textarea").first
                    page.wait_for_timeout(1500)
                    page.locator("input[type=file]").set_input_files(str(SPEC))  # live Muse Spark 1.3 intake
                    page.wait_for_function("document.querySelector('textarea').value.length > 20", timeout=60000)
                elif action == "submit":
                    box.press("Enter")
                elif action == "trace":
                    page.get_by_text("Agent trace").scroll_into_view_if_needed()
                elif action == "followup":
                    page.mouse.wheel(0, -2000)
                    box.click()
                    box.type(FOLLOWUP, delay=25)
                    box.press("Enter")
                page.wait_for_timeout(max(0, (end - time.time()) * 1000))
            video = Path(page.video.path())
            ctx.close()
            browser.close()
        return video, starts
    finally:
        app.terminate()


def mux(video: Path, clips, starts):
    import imageio_ffmpeg
    import numpy as np
    import soundfile as sf
    total = starts[-1] + len(clips[-1]) / RATE + 1.0
    track = np.zeros(int(total * RATE))
    srt = []
    for i, (clip, st, (_, text, _)) in enumerate(zip(clips, starts, SCENES), 1):
        a = int(st * RATE)
        track[a:a + len(clip)] = clip[: len(track) - a]
        en = st + len(clip) / RATE
        fmt = lambda s: f"{int(s // 3600):02}:{int(s % 3600 // 60):02}:{int(s % 60):02},{int(s * 1000 % 1000):03}"
        srt.append(f"{i}\n{fmt(st)} --> {fmt(en)}\n{text}\n")
    sf.write(OUT / "narration.wav", track, RATE)
    (OUT / "alloy-agent-demo.srt").write_text("\n".join(srt), encoding="utf-8")
    mp4 = OUT / "alloy-agent-demo.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                    "-i", str(OUT / "narration.wav"), "-vf", f"scale={W}:{H}:flags=lanczos",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.2f}", str(mp4)], check=True)
    print(f"wrote {mp4} ({total:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-scenario", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.make_scenario:
        make_scenario()
    else:
        clips = narrate()
        video, starts = record(clips)
        mux(video, clips, starts)
