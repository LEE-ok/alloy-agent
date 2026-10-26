"""Download and load the Steel Strength dataset (MIT, see DATA.md)."""
import csv
import hashlib
import urllib.request
from pathlib import Path

URL = "https://ndownloader.figshare.com/files/13354694"
SHA256 = "53c995e14371098ff8f6186e43c2a6d833650d352dc0d77118751cf6b06727d7"
PATH = Path(__file__).resolve().parents[2] / "data" / "steel_strength.csv"

ELEMENTS = ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]
TARGETS = ["yield strength", "tensile strength", "elongation"]


def fetch(path: Path = PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=30) as r:
            body = r.read()
        if hashlib.sha256(body).hexdigest() != SHA256:
            raise ValueError("steel_strength.csv checksum mismatch")
        path.write_bytes(body)
    return path


def load() -> list[dict[str, float | None]]:
    """Rows of {element: wt%, target: value or None}; the formula column is dropped."""
    with open(fetch(), newline="") as f:
        return [
            {k: float(row[k]) if row[k] else None for k in ELEMENTS + TARGETS}
            for row in csv.DictReader(f)
        ]
