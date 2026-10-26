# alloy-agent

Describe the steel you want in plain language. A Muse Spark agent turns it into constraints,
calls prediction/search tools, and returns ranked candidate compositions with calibrated ±2σ
bands, predicted stress-strain curves, and reasons. Out-of-domain candidates are flagged, not hidden.

## Built during the hackathon

Everything in this repository was written during the Meta Global AI Developer Hackathon
(Oct 27 – Nov 4, 2026), starting from an empty repo. No pre-existing code or models are included.
Data: see [DATA.md](DATA.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"   # Windows; use .venv/bin/pip elsewhere
cp .env.example .env                    # then fill in MODEL_API_KEY
```

## License

MIT
