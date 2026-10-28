"""The three tools the agent can call: predict, search, pareto. Plus their schemas and a safe dispatcher."""
import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .data import ELEMENTS
from .predictor import OpenPredictor, Prediction

Target = Literal["ys", "uts", "el"]
Bounds = dict[str, tuple[float, float]]
FRONT_MAX = 15  # keeps the tool result small enough for the LLM context


class Constraint(BaseModel):
    target: Target
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.min is None and self.max is None:
            raise ValueError("constraint needs min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) > max ({self.max})")
        return self


class Objective(BaseModel):
    target: Target
    goal: Literal["max", "min"] = "max"


class PredictArgs(BaseModel):
    composition: dict[str, float]


class SearchArgs(BaseModel):
    constraints: list[Constraint] = Field(min_length=1)
    bounds: Bounds | None = None
    n: int = Field(5, ge=1, le=10)


class ParetoArgs(BaseModel):
    objectives: list[Objective] = Field(min_length=2)
    constraints: list[Constraint] | None = None
    bounds: Bounds | None = None


def candidate_pool(pred: OpenPredictor, bounds: Bounds | None = None, size: int = 2000, seed: int = 0) -> list[dict[str, float]]:
    """Interpolate random pairs of training steels, jitter, clamp to training ranges; user bounds override."""
    rng = np.random.default_rng(seed)
    X = pred.train_X
    lo, hi = X.min(0), X.max(0)
    i, j = rng.integers(0, len(X), (2, size))
    lam = rng.random((size, 1))
    C = lam * X[i] + (1 - lam) * X[j] + rng.normal(0, 0.05 * (hi - lo), (size, len(ELEMENTS)))
    C = np.clip(C, lo, hi)
    for el, (b_lo, b_hi) in (bounds or {}).items():
        el = el.strip().lower()
        if el == "fe":
            continue
        if el not in ELEMENTS:
            raise ValueError(f"unknown element: {el}")
        C[:, ELEMENTS.index(el)] = rng.uniform(b_lo, b_hi, size)  # may leave the training range → flagged OOD
    return [dict(zip(ELEMENTS, row)) for row in C]


def _margins(p: Prediction, constraints: list[Constraint], pred: OpenPredictor) -> list[float]:
    out = []
    for c in constraints:
        lo, hi = pred.target_ranges[c.target]
        mu, s = p.mu[c.target], p.sigma[c.target]
        if c.min is not None:
            out.append((mu - 2 * s - c.min) / (hi - lo))
        if c.max is not None:
            out.append((c.max - (mu + 2 * s)) / (hi - lo))
    return out


def _cand(i: int, comp: dict[str, float], p: Prediction, **extra) -> dict:
    return {"id": i,
            "composition": {e: round(float(v), 3) for e, v in comp.items() if v >= 0.01},
            "pred": {t: {"mu": round(p.mu[t], 1), "sigma": round(p.sigma[t], 1)} for t in p.mu},
            **{k: round(v, 4) for k, v in extra.items()},
            "ood": p.ood, "ood_reasons": p.ood_reasons}


def predict_tool(pred: OpenPredictor, composition: dict[str, float]) -> dict:
    comp = pred.normalize(composition)
    return _cand(0, comp, pred.predict([comp])[0])


def search(pred: OpenPredictor, constraints: list[Constraint], bounds: Bounds | None = None, n: int = 5, seed: int = 0) -> dict:
    pool = candidate_pool(pred, bounds, seed=seed)
    preds = pred.predict(pool)
    scores = [min(_margins(p, constraints, pred)) for p in preds]
    order = np.argsort(scores)[::-1][:n]
    feasible = sum(s >= 0 for s in scores)

    binding = None
    if not feasible:
        best = preds[order[0]]
        c = min(constraints, key=lambda c: min(_margins(best, [c], pred)))
        mu, s = best.mu[c.target], best.sigma[c.target]
        if c.min is not None and (c.max is None or mu - 2 * s - c.min < c.max - (mu + 2 * s)):
            achieved, bound, value, short = mu - 2 * s, "min", c.min, c.min - (mu - 2 * s)
        else:
            achieved, bound, value, short = mu + 2 * s, "max", c.max, (mu + 2 * s) - c.max
        binding = {"target": c.target, "bound": bound, "value": value,
                   "achieved_conservative": round(achieved, 1), "short_by": round(short, 1)}

    return {"candidates": [_cand(rank + 1, pool[i], preds[i], score=scores[i]) for rank, i in enumerate(order)],
            "feasible_count": int(feasible), "binding": binding}


def pareto(pred: OpenPredictor, objectives: list[Objective], constraints: list[Constraint] | None = None,
           bounds: Bounds | None = None, seed: int = 0) -> dict:
    pool = candidate_pool(pred, bounds, seed=seed)
    preds = pred.predict(pool)
    rows = list(range(len(pool)))
    if constraints:
        scores = {i: min(_margins(preds[i], constraints, pred)) for i in rows}
        rows = [i for i in rows if scores[i] >= 0]
        if not rows:
            return {"front": [], "note": "no candidate meets the constraints conservatively; "
                                         "call pareto without constraints to see the trade-off"}

    def norm(t, v):
        lo, hi = pred.target_ranges[t]
        return (v - lo) / (hi - lo)

    sign = np.array([1.0 if o.goal == "max" else -1.0 for o in objectives])
    F = np.array([[preds[i].mu[o.target] for o in objectives] for i in rows]).reshape(-1, len(objectives)) * sign
    dominated = [bool(np.any(np.all(F >= f, 1) & np.any(F > f, 1))) for f in F]
    front = [i for i, d in zip(rows, dominated) if not d]

    def rank(i):
        sig = np.mean([preds[i].sigma[o.target] / np.ptp(pred.target_ranges[o.target]) for o in objectives])
        gain = scores[i] if constraints else sum(s * norm(o.target, preds[i].mu[o.target]) for s, o in zip(sign, objectives))
        return gain / sig

    front = sorted(front, key=rank, reverse=True)[:FRONT_MAX]
    front.sort(key=lambda i: -sign[0] * preds[i].mu[objectives[0].target])
    return {"front": [_cand(k + 1, pool[i], preds[i], rank=rank(i)) for k, i in enumerate(front)]}


_COMP = {"type": "object", "additionalProperties": {"type": "number"},
         "description": "Element symbol -> wt%. Known: " + ", ".join(e.capitalize() for e in ELEMENTS)
                        + ". Fe is the balance; omit it. Unspecified elements default to the training median."}
_CONSTRAINTS = {"type": "array", "items": {"type": "object", "properties": {
    "target": {"type": "string", "enum": ["ys", "uts", "el"],
               "description": "ys = yield strength (MPa), uts = ultimate tensile strength (MPa), el = elongation (%)"},
    "min": {"type": "number"}, "max": {"type": "number"}}, "required": ["target"]}}
_BOUNDS = {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
           "description": "Optional element -> [low, high] wt% to restrict the search."}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "predict",
        "description": "Predict yield strength, tensile strength (MPa) and elongation (%) of one steel composition, "
                       "with calibrated sigma (mu ± 2 sigma ≈ 95%) and an out-of-domain flag.",
        "parameters": {"type": "object", "properties": {"composition": _COMP}, "required": ["composition"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Search ~2000 candidate compositions and rank them by the conservative margin (mu - 2 sigma "
                       "for minimums, mu + 2 sigma for maximums). score >= 0 means every constraint holds. If none is "
                       "feasible, 'binding' says which constraint falls short and by how much.",
        "parameters": {"type": "object", "properties": {
            "constraints": _CONSTRAINTS, "bounds": _BOUNDS,
            "n": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How many candidates to return (default 5)."}},
            "required": ["constraints"]}}},
    {"type": "function", "function": {
        "name": "pareto",
        "description": "Trade-off front between two or more objectives (e.g. maximize ys and el). Returns up to 15 "
                       "non-dominated candidates; 'rank' is promise per unit uncertainty, useful for picking the next experiments.",
        "parameters": {"type": "object", "properties": {
            "objectives": {"type": "array", "minItems": 2, "items": {"type": "object", "properties": {
                "target": {"type": "string", "enum": ["ys", "uts", "el"]},
                "goal": {"type": "string", "enum": ["max", "min"]}}, "required": ["target"]}},
            "constraints": _CONSTRAINTS, "bounds": _BOUNDS},
            "required": ["objectives"]}}},
]


def dispatch(pred: OpenPredictor, name: str, arguments: str) -> dict:
    """Run a tool call from the LLM. Any failure comes back as {"error": ...} so the model can retry."""
    try:
        args = json.loads(arguments or "{}")
        if name == "predict":
            return predict_tool(pred, **PredictArgs(**args).model_dump())
        if name == "search":
            a = SearchArgs(**args)
            return search(pred, a.constraints, a.bounds, a.n)
        if name == "pareto":
            a = ParetoArgs(**args)
            return pareto(pred, a.objectives, a.constraints, a.bounds)
        return {"error": f"unknown tool: {name}"}
    except Exception as e:  # tool errors are data for the model, not crashes
        return {"error": f"{type(e).__name__}: {e}"[:500]}
