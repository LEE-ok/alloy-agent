"""Held-out rediscovery (no LLM): hide the strongest 10% of steels, retrain, and check whether `search`
proposes compositions near the hidden ones when asked for their properties.

found = one of the top-5 candidates lies within the training set's median nearest-neighbour distance of the
held-out steel (range-scaled composition space). Reported as is, even if it is 0.
"""
import csv
from pathlib import Path

import numpy as np

from alloy_agent.data import ELEMENTS, load
from alloy_agent.predictor import OpenPredictor
from alloy_agent.tools import Constraint, search

HERE = Path(__file__).resolve().parent


def main():
    rows = [r for r in load() if r["elongation"] is not None]
    cut = np.quantile([r["yield strength"] for r in rows], 0.9)
    held = [r for r in rows if r["yield strength"] >= cut]
    pred = OpenPredictor([r for r in rows if r["yield strength"] < cut])

    Z = pred._scale(pred.train_X)
    d = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    np.fill_diagonal(d, np.inf)
    radius = float(np.median(d.min(1)))

    out, found = [], 0
    for i, h in enumerate(held):
        r = search(pred, [Constraint(target="ys", min=0.9 * h["yield strength"]),
                          Constraint(target="el", min=0.9 * h["elongation"])])
        target = pred._scale(np.array([[h[e] for e in ELEMENTS]]))[0]
        dist = min(np.linalg.norm(pred._scale(np.array([[c["composition"].get(e, 0.0) for e in ELEMENTS]]))[0] - target)
                   for c in r["candidates"])
        hit = dist <= radius
        found += hit
        out.append({"held_out": i, "ys": h["yield strength"], "el": h["elongation"], "feasible": r["feasible_count"],
                    "nearest_dist": round(float(dist), 3), "radius": round(radius, 3), "found": hit})

    with (HERE / "rediscovery.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"rediscovered {found}/{len(held)} (radius {radius:.3f}, held-out YS >= {cut:.0f} MPa)")


if __name__ == "__main__":
    main()
