"""OpenPredictor: per-target bootstrap ensembles with scalar-k sigma calibration and a domain check."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors

from . import data
from .data import ELEMENTS

TARGETS = {"ys": "yield strength", "uts": "tensile strength", "el": "elongation"}


@dataclass
class Prediction:
    mu: dict[str, float]
    sigma: dict[str, float]
    ood: bool
    ood_reasons: list[str]


class OpenPredictor:
    def __init__(self, rows: list[dict] | None = None, n_models: int = 20, seed: int = 0):
        rows = data.load() if rows is None else rows
        X_all = np.array([[r[e] for e in ELEMENTS] for r in rows], dtype=float)
        lo, hi = X_all.min(0), X_all.max(0)
        self.element_ranges = {e: (float(a), float(b)) for e, a, b in zip(ELEMENTS, lo, hi)}
        self.medians = {e: float(m) for e, m in zip(ELEMENTS, np.median(X_all, 0))}
        self._lo, self._span = lo, np.where(hi > lo, hi - lo, 1.0)

        rng = np.random.default_rng(seed)
        self.target_ranges, self.k, self.coverage, self.mae, self._models = {}, {}, {}, {}, {}
        for t, col in TARGETS.items():
            keep = [r[col] is not None for r in rows]
            X, y = X_all[keep], np.array([r[col] for r in rows if r[col] is not None], dtype=float)
            self.target_ranges[t] = (float(y.min()), float(y.max()))
            make = _pick_model(X, y, seed)
            self.k[t], self.coverage[t], self.mae[t] = _calibrate(X, y, make, rng, seed)
            self._models[t] = _ensemble(X, y, make, n_models, rng)

        # Domain check: mean distance to the 5 nearest training rows, in range-scaled space.
        self._nn = NearestNeighbors(n_neighbors=6).fit(self._scale(X_all))
        d, _ = self._nn.kneighbors(self._scale(X_all))
        self._knn_threshold = float(np.percentile(d[:, 1:].mean(1), 95))  # column 0 is the row itself

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self._lo) / self._span

    def normalize(self, comp: dict[str, float]) -> dict[str, float]:
        out = dict(self.medians)
        for k, v in comp.items():
            k = k.strip().lower()
            if k == "fe":
                continue  # balance
            if k not in out:
                raise ValueError(f"unknown element: {k}")
            out[k] = float(v)
        return out

    def predict(self, comps: list[dict[str, float]]) -> list[Prediction]:
        norm = [self.normalize(c) for c in comps]
        X = np.array([[c[e] for e in ELEMENTS] for c in norm])
        per_target = {}
        for t, models in self._models.items():
            preds = np.stack([m.predict(X) for m in models])
            per_target[t] = (preds.mean(0), np.maximum(self.k[t] * preds.std(0), 1e-6))
        knn = self._nn.kneighbors(self._scale(X), n_neighbors=5)[0].mean(1)

        out = []
        for i, c in enumerate(norm):
            reasons = [f"{e}={c[e]:g} outside [{lo:g}, {hi:g}]"
                       for e, (lo, hi) in self.element_ranges.items() if not lo <= c[e] <= hi]
            if knn[i] > self._knn_threshold:
                reasons.append("far from training data")
            out.append(Prediction(
                mu={t: float(mu[i]) for t, (mu, _) in per_target.items()},
                sigma={t: float(s[i]) for t, (_, s) in per_target.items()},
                ood=bool(reasons), ood_reasons=reasons))
        return out


def _candidates(seed: int):
    return [lambda: GradientBoostingRegressor(random_state=seed),
            # single-threaded: parallel tree summation makes predictions differ in the last bits
            lambda: RandomForestRegressor(n_estimators=100, random_state=seed)]


def _pick_model(X, y, seed):
    """The candidate regressor with the lower 5-fold MAE."""
    cv = KFold(5, shuffle=True, random_state=seed)
    return min(_candidates(seed),
               key=lambda make: -cross_val_score(make(), X, y, cv=cv, scoring="neg_mean_absolute_error").mean())


def _ensemble(X, y, make, n, rng):
    models = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        models.append(make().fit(X[idx], y[idx]))
    return models


def _calibrate(X, y, make, rng, seed, n_small=5):
    """Smallest k with mu ± 2k·sigma_raw covering 95% of out-of-fold rows; returns (k, coverage, MAE)."""
    err, sraw = np.empty_like(y), np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        preds = np.stack([m.predict(X[te]) for m in _ensemble(X[tr], y[tr], make, n_small, rng)])
        err[te], sraw[te] = np.abs(y[te] - preds.mean(0)), np.maximum(preds.std(0), 1e-6)
    k = float(np.quantile(err / (2 * sraw), 0.95))
    return k, float(np.mean(err <= 2 * k * sraw)), float(err.mean())
