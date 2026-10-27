# alloy-agent 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**목표:** 목표 물성을 받아 강재 조성 후보를 탐색하고, 보정된 ±2σ와 근거를 함께 돌려주는 Muse Spark 에이전트를 만든다. 답에 나오는 숫자는 모두 툴에서 나온 값이어야 한다.

**아키텍처:**
- LLM 에이전트는 하나만 두고, 툴 3개(predict, search, pareto)를 쓴다.
- 검증기는 LLM이 아니라 코드로 만든다.
- 예측기는 공개 데이터 312건으로 학습한 부트스트랩 앙상블이다. σ는 스칼라 k 하나로 보정한다.
- UI는 Gradio로 만든다. 채팅, 트레이스, Pareto 산점도를 넣는다.

**기술 스택:** Python 3.12, numpy, scikit-learn, pydantic 2, openai SDK(OpenAI 호환 엔드포인트), gradio 6, pytest

**스펙:** `docs/superpowers/specs/2026-10-27-alloy-agent-design.md`

## 전역 제약

- 코드, UI 문구, 프롬프트, 툴 설명, 에러 메시지, README는 영어로 쓴다. 에이전트는 사용자가 쓴 언어로 답한다.
- 데이터는 `alloy_agent.data.load()`로만 읽는다. 데이터 파일은 커밋하지 않는다.
- 원소 키는 `data.ELEMENTS`(소문자 13종)를 쓴다. 입력은 대소문자를 구분하지 않는다. `fe`는 무시한다.
- 타깃 키는 `"ys"`, `"uts"`, `"el"` 3개다. 단위는 MPa, MPa, %다.
- σ는 보정된 값이고, 여유는 μ∓2σ를 기준으로 계산한다.
- LLM 호출 시 `max_tokens`는 1000 이상으로 둔다(추론 토큰도 여기서 차감된다). `tool_choice`는 `"auto"`를 쓴다.
- 기본 백엔드: Baseten `https://inference.baseten.co/v1`, 모델 `meta-models/muse-spark-open-1026`. 대안: Meta `https://api.meta.ai/v1`, 모델 `muse-spark-1.3`.
- 호출 상한: 턴당 툴 호출 6회, 재탐색 2회, 턴당 토큰 60,000, 세션당 토큰 400,000.
- API 키는 출력하지도, 커밋하지도 않는다. 테스트는 실제 API를 부르지 않는다(가짜 LLM을 쓴다).
- 커밋 메시지에 Claude 표기를 넣지 않는다.

## 리뷰 포커스

1. **LLM이 원소를 "Ni", "NI", "Fe"처럼 쓰거나 "nickel"처럼 이름으로 쓰는 경우.** 대소문자는 정규화하고 Fe는 무시한다. 모르는 키는 `{"error": ...}`로 돌려줘야 한다. → Task 2 `test_dispatch_element_keys`
2. **LLM이 깨진 JSON 인자나 스키마에 맞지 않는 인자를 보내는 경우.** 예외로 죽지 말고 에러를 툴 결과로 돌려줘야 한다. → Task 2 `test_dispatch_bad_json`
3. **min이 max보다 크거나, 제약 목록이 비어 있는 경우.** 검증 에러를 돌려줘야 한다. → Task 2 `test_constraint_validation`
4. **사용자 범위(bounds)가 학습 범위 밖인 경우.** 결과에 OOD 표시가 붙고 숨겨지지 않아야 한다. → Task 2 `test_search_bounds_outside_training_flagged`
5. **LLM 호출이 네트워크 에러나 타임아웃으로 실패하는 경우.** 한 번 재시도하고, 그래도 실패하면 턴이 크래시 없이 에러 텍스트로 끝나야 한다. → Task 4 `test_llm_error_ends_turn_gracefully`

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/alloy_agent/predictor.py` | OpenPredictor: 학습, 보정, 예측, 범위 검사 |
| `src/alloy_agent/tools.py` | 후보 생성, search, pareto, 툴 스키마, dispatch |
| `src/alloy_agent/verify.py` | 본문에 나온 숫자가 근거 있는지 검증 |
| `src/alloy_agent/llm.py` | 환경변수로 클라이언트와 모델 선택, 재시도가 붙은 chat 함수 |
| `src/alloy_agent/agent.py` | 세션, 턴 루프, 상한, 최종 답 조립, runs.jsonl 기록 |
| `src/alloy_agent/intake.py` | PDF/이미지에서 Constraints 추출 |
| `app.py` | Gradio UI |
| `eval/queries.jsonl`, `eval/run_eval.py`, `eval/rediscovery.py` | 평가 |

---

### Task 1: OpenPredictor

**Files:**
- Create: `src/alloy_agent/predictor.py`
- Test: `tests/test_predictor.py`

**Interfaces:**
- Consumes: `data.load() -> list[dict]`, `data.ELEMENTS`
- Produces:
  - `TARGETS = {"ys": "yield strength", "uts": "tensile strength", "el": "elongation"}`
  - `@dataclass Prediction: mu: dict[str, float]; sigma: dict[str, float]; ood: bool; ood_reasons: list[str]`
  - `class OpenPredictor`
    - `__init__(self, rows: list[dict] | None = None, n_models: int = 20, seed: int = 0)`: 생성과 동시에 학습한다. `rows=None`이면 `data.load()`를 쓴다.
    - 속성:
      - `element_ranges: dict[str, tuple[float, float]]`
      - `target_ranges: dict[str, tuple[float, float]]`
      - `medians: dict[str, float]`
      - `k: dict[str, float]`
      - `coverage: dict[str, float]`
      - `mae: dict[str, float]`
    - `normalize(self, comp: dict[str, float]) -> dict[str, float]`: 키를 소문자로 바꾸고 `fe`를 버린다. 빠진 원소는 중앙값으로 채운다. 모르는 키가 있으면 `ValueError(f"unknown element: {k}")`를 낸다.
    - `predict(self, comps: list[dict[str, float]]) -> list[Prediction]`: 정규화 → 배치 예측 → σ = k·σ_raw → 범위 검사 순서로 처리한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
P = OpenPredictor()  # module-level, fit once

def test_shapes():
    p = P.predict([{"ni": 18, "co": 9, "mo": 5, "ti": 0.7}])[0]
    assert set(p.mu) == set(p.sigma) == {"ys", "uts", "el"}
    assert all(s > 0 for s in p.sigma.values())

def test_fe_and_case_ignored():
    a = P.predict([{"ni": 18, "co": 9}])[0]
    b = P.predict([{"Ni": 18, "CO": 9, "Fe": 70}])[0]
    assert a.mu == b.mu

def test_unknown_element():
    with pytest.raises(ValueError, match="unknown element"):
        P.normalize({"cu": 1})

def test_calibrated_coverage():
    assert all(c >= 0.9 for c in P.coverage.values())

def test_ood_flag():
    assert P.predict([{"ni": 50}])[0].ood
    assert not P.predict([P.medians])[0].ood
```

- [ ] **Step 2: 실패 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_predictor.py -v`
  - Expected: `ModuleNotFoundError`, FAIL

- [ ] **Step 3: `predictor.py` 구현**
  - **모델 선택:** 타깃마다 5-fold로 `GradientBoostingRegressor(random_state=seed)`와 `RandomForestRegressor(n_estimators=100, random_state=seed)`의 MAE를 비교하고, 더 좋은 쪽을 쓴다.
  - **학습 데이터:** 그 타깃 값이 None인 행은 뺀다.
  - **앙상블:** 행을 부트스트랩으로 리샘플해 `n_models`개를 학습한다.
  - **k 보정:**
    - 5-fold의 각 fold에서, 학습 fold로 작은 앙상블(5개)을 만들어 out-of-fold 행의 μ와 σ_raw를 구한다.
    - `|y−μ| ≤ 2kσ_raw` 비율이 0.95 이상이 되는 가장 작은 k를 구한다(정렬된 `|y−μ|/(2σ_raw)`의 95백분위수).
    - 이때의 포함률을 `coverage`, MAE를 `mae`에 저장한다.
  - **범위 검사:**
    - 원소가 `element_ranges`를 벗어나면 `ood_reasons.append(f"{el}={v} outside [{lo}, {hi}]")`
    - 원소 범위로 표준화한 입력에서 kNN(k=5) 평균거리를 구한다. 이 값이 학습 세트 자기 거리(자기 자신 제외)의 95백분위수보다 크면 `"far from training data"`를 추가한다.
    - `ood = bool(ood_reasons)`

- [ ] **Step 4: 통과 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_predictor.py -v`
  - Expected: 5 passed
  - 출력된 `P.coverage`와 `P.mae`는 메모해 둔다(나중에 README에 넣는다).

- [ ] **Step 5: 커밋**

```bash
git add src/alloy_agent/predictor.py tests/test_predictor.py
git commit -m "feat: OpenPredictor with bootstrap ensemble, scalar-k calibration and domain check"
```

---

### Task 2: 툴 (search, pareto, dispatch)

**Files:**
- Create: `src/alloy_agent/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `OpenPredictor.predict`, `.normalize`, `.element_ranges`, `.target_ranges`, `Prediction`
- Produces:
  - `class Constraint(BaseModel)`
    - 필드: `target: Literal["ys","uts","el"]; min: float | None = None; max: float | None = None`
    - validator: min과 max가 둘 다 None이거나 min > max면 에러를 낸다.
  - `class Objective(BaseModel): target: Literal["ys","uts","el"]; goal: Literal["max","min"] = "max"`
  - `Bounds = dict[str, tuple[float, float]]`: 원소별 (lo, hi) wt%
  - `def candidate_pool(pred, bounds: Bounds | None = None, size: int = 2000, seed: int = 0) -> list[dict[str, float]]`
  - `def search(pred, constraints: list[Constraint], bounds: Bounds | None = None, n: int = 5, seed: int = 0) -> dict`
    - 반환: `{"candidates": [Cand...], "feasible_count": int, "binding": dict | None}`
  - `def pareto(pred, objectives: list[Objective], constraints: list[Constraint] | None = None, bounds: Bounds | None = None, seed: int = 0) -> dict`
    - 반환: `{"front": [Cand...]}`. 각 Cand에 `rank` 키가 붙는다.
  - `def predict_tool(pred, composition: dict[str, float]) -> dict`: Cand와 같은 모양이고 score는 없다.
  - Cand 형태:

    ```
    {"id": int,
     "composition": {el: round(wt, 3)},
     "pred": {t: {"mu": round(mu, 1), "sigma": round(s, 1)}},
     "score": round(x, 4),
     "ood": bool,
     "ood_reasons": [...]}
    ```

    composition에는 0.01 이상인 원소만 넣는다.
  - `TOOL_SPECS: list[dict]`: OpenAI function 스키마 3개. 설명은 영어로 쓰고, 단위(MPa, %, wt%)와 "Fe is balance, omit it"을 명시한다.
  - `def dispatch(pred, name: str, arguments: str) -> dict`: JSON을 파싱하고 pydantic으로 검증한 뒤 해당 함수를 호출한다. 모든 예외는 `{"error": str}`로 돌려준다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
P = OpenPredictor()

def test_search_easy_feasible():
    r = search(P, [Constraint(target="ys", min=1200)])
    assert r["feasible_count"] > 0 and r["candidates"][0]["score"] >= 0
    assert r["binding"] is None

def test_search_impossible_binding():
    r = search(P, [Constraint(target="ys", min=3000)])
    assert r["feasible_count"] == 0
    assert r["binding"]["target"] == "ys" and r["binding"]["short_by"] > 0

def test_pareto_non_dominated():
    f = pareto(P, [Objective(target="ys"), Objective(target="el")])["front"]
    pts = [(c["pred"]["ys"]["mu"], c["pred"]["el"]["mu"]) for c in f]
    assert not any(a[0] >= b[0] and a[1] >= b[1] and a != b for a in pts for b in pts)
    assert all("rank" in c for c in f)

def test_constraint_validation():
    with pytest.raises(ValidationError): Constraint(target="ys", min=2000, max=1000)
    with pytest.raises(ValidationError): Constraint(target="ys")
    assert "error" in dispatch(P, "search", '{"constraints": []}')

def test_dispatch_element_keys():
    assert "error" not in dispatch(P, "predict", '{"composition": {"Ni": 18, "Fe": 70}}')
    assert "unknown element" in dispatch(P, "predict", '{"composition": {"nickel": 18}}')["error"]

def test_dispatch_bad_json():
    assert "error" in dispatch(P, "search", "{not json")
    assert "error" in dispatch(P, "nope", "{}")

def test_search_bounds_outside_training_flagged():
    lo, hi = P.element_ranges["ni"]
    r = search(P, [Constraint(target="ys", min=1000)], bounds={"ni": (hi + 5, hi + 10)})
    assert r["candidates"] and all(c["ood"] for c in r["candidates"])
```

- [ ] **Step 2: 실패 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
  - Expected: FAIL (import error)

- [ ] **Step 3: `tools.py` 구현**
  - **`candidate_pool`:**
    - 원소 범위로 표준화해 둔 학습 행에서, size개마다 무작위로 두 행을 고르고 λ∈U(0,1)로 보간한 뒤 각 원소에 ±5%(범위 대비) 가우시안 섭동을 준다.
    - 결과는 학습 min/max로 자른다.
    - `bounds`가 있으면 그 원소를 `U(lo, hi)`로 덮어쓴다. 이렇게 하면 범위 밖 bounds도 그대로 반영되고 OOD로 표시된다.
    - 시드는 `np.random.default_rng(seed)`로 준다.
  - **score와 binding:** 스펙 공식대로 계산하고, range는 `target_ranges`를 쓴다.
    - 최소 목표의 여유: `(mu-2s-min)/range`
    - 최대 목표의 여유: `(max-(mu+2s))/range`
    - feasible한 후보가 없을 때 binding은 최고 점수 후보 기준으로 만든다. 형태는 `{"target", "bound": "min"|"max", "value", "achieved_conservative": mu∓2s, "short_by": MPa 또는 %}`
  - **pareto rank:**
    - constraints가 있으면 rank는 `score/평균 정규화 σ`다.
    - 없으면 목표 μ의 정규화 합 / 평균 정규화 σ다.
  - **정렬:** pareto 전선은 첫 번째 목표 μ 기준 내림차순으로 정렬한다.

- [ ] **Step 4: 통과 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
  - Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
git add src/alloy_agent/tools.py tests/test_tools.py
git commit -m "feat: search, pareto and validated tool dispatch"
```

---

### Task 3: 검증기

**Files:**
- Create: `src/alloy_agent/verify.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Produces:
  - `ALLOWED = {0, 1, 2, 3, 4, 5, 95}`: 순번, "2σ", "95%" 같은 숫자
  - `def numbers(text: str) -> list[float]`: 정규식 `-?\d+(?:,\d{3})*(?:\.\d+)?`로 찾고 쉼표를 제거한다.
  - `def verify(text: str, tool_outputs: list[dict], user_texts: list[str]) -> list[str]`
    - 근거 없는 숫자마다 `f"unsupported number: {x}"`를 넣어 돌려준다.
    - 빈 리스트면 통과다.
    - 근거 판정: `|x−v| ≤ 0.01·max(|v|, 1)`를 만족하는 v가 툴 출력(중첩 dict/list의 모든 숫자) 또는 사용자 텍스트의 숫자 중에 있거나, x가 `ALLOWED`에 있으면 근거가 있는 것으로 본다.
  - `def strip_unsupported(text: str, issues: list[str]) -> str`: 근거 없는 숫자를 `"[?]"`로 바꾼다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
OUT = [{"candidates": [{"pred": {"ys": {"mu": 1523.4, "sigma": 61.2}}}]}]

def test_rounding_ok():
    assert verify("Yield is about 1523 MPa (±2σ, σ≈61).", OUT, []) == []

def test_fabricated_caught():
    assert verify("Yield reaches 1850 MPa.", OUT, []) == ["unsupported number: 1850.0"]

def test_user_numbers_whitelisted():
    assert verify("You asked for 2,000 MPa; best is 1523.", OUT, ["at least 2000 MPa"]) == []

def test_strip():
    t = "Yield reaches 1850 MPa."
    assert strip_unsupported(t, verify(t, OUT, [])) == "Yield reaches [?] MPa."
```

- [ ] **Step 2: 실패 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_verify.py -v`
  - Expected: FAIL

- [ ] **Step 3: `verify.py` 구현**

- [ ] **Step 4: 통과 확인**
  - Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/alloy_agent/verify.py tests/test_verify.py
git commit -m "feat: numeric-provenance verifier"
```

---

### Task 4: LLM 클라이언트와 에이전트 루프

**Files:**
- Create: `src/alloy_agent/llm.py`, `src/alloy_agent/agent.py`
- Modify: `.env.example`
  - `LLM_BASE_URL=https://inference.baseten.co/v1`, `LLM_MODEL=meta-models/muse-spark-open-1026`을 기본값으로 둔다.
  - Meta 값은 주석으로 남긴다.
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `tools.TOOL_SPECS`, `tools.dispatch`, `verify.verify`, `verify.strip_unsupported`, `OpenPredictor`
- Produces:
  - `llm.py`
    - `Chat = Callable[[list[dict], list[dict]], dict]`
      - 반환: `{"content": str | None, "tool_calls": [{"id", "name", "arguments"}], "usage": {"total_tokens": int}}`
    - `def make_chat(base_url: str | None = None, model: str | None = None) -> Chat`
      - base_url과 model은 기본으로 env `LLM_BASE_URL`, `LLM_MODEL`을 읽는다.
      - 키는 `LLM_API_KEY`를 먼저 보고, 없으면 base_url에 "baseten"이 들어 있을 때 `BASETEN_API_KEY`, 아니면 `MODEL_API_KEY`를 쓴다.
      - `max_tokens=4000`, `tool_choice="auto"`, `timeout=120`
      - `openai.APIError` 또는 타임아웃이 나면 2초 뒤 1회 재시도하고, 그래도 실패하면 예외를 그대로 올린다.
  - `agent.py`
    - `@dataclass Session`
      - `messages: list[dict]`
      - `last_front: list[dict]`
      - `candidates: list[dict]`
      - `tokens_used: int = 0`
    - `@dataclass TurnResult`
      - `text: str`
      - `table: list[dict]`: Cand 목록
      - `trace: list[dict]`: `{"step", "tool", "args", "summary", "status"}`
      - `issues: list[str]`
      - `capped: bool`
    - `SYSTEM_PROMPT: str`: 영어로 쓴다. 들어갈 내용:
      - 툴 규칙
      - 데이터 범위(마레이징/초고강도강)
      - "Never state a number no tool returned"
      - "Reply in the user's language"
      - "Fe is balance"
      - "Convert ksi to MPa before calling tools"
    - 상수: `MAX_TOOL_CALLS = 6`, `MAX_REFINE = 2`, `TURN_TOKEN_CAP = 60_000`, `SESSION_TOKEN_CAP = 400_000`
    - `def run_turn(session: Session, user_text: str, chat: Chat, pred: OpenPredictor, log_path: str | None = "runs.jsonl") -> TurnResult`
    - `def render_table(cands: list[dict]) -> str`: Markdown 표. 열은 #, composition, YS μ±2σ, UTS μ±2σ, El μ±2σ, OOD("⚠ out of domain" 또는 "")이다.

- [ ] **Step 1: 실패하는 테스트 작성**

가짜 chat은 미리 정해 둔 응답 목록을 순서대로 하나씩 꺼내 돌려준다.

```python
P = OpenPredictor()

def scripted(*replies):
    it = iter(replies)
    return lambda messages, tools: next(it)

def call(name, args): return {"content": None, "tool_calls": [{"id": "c1", "name": name, "arguments": json.dumps(args)}], "usage": {"total_tokens": 100}}
def say(text): return {"content": text, "tool_calls": [], "usage": {"total_tokens": 50}}

def test_search_then_answer():
    s = Session([], [], [])
    r = run_turn(s, "YS at least 1200 MPa", scripted(call("search", {"constraints": [{"target": "ys", "min": 1200}]}), say("Here are options.")), P, None)
    assert r.table and r.issues == [] and r.trace[0]["tool"] == "search"
    assert "⚠" in render_table([dict(r.table[0], ood=True)])

def test_tool_cap():
    r = run_turn(Session([], [], []), "x", lambda m, t: call("predict", {"composition": {"ni": 18}}), P, None)
    assert r.capped and sum(1 for t in r.trace if t["tool"]) == MAX_TOOL_CALLS

def test_verifier_retry_then_strip():
    r = run_turn(Session([], [], []), "x", scripted(say("Yield 9999 MPa."), say("Still 9999 MPa.")), P, None)
    assert "9999" not in r.text and r.issues

def test_llm_error_ends_turn_gracefully():
    def boom(m, t): raise TimeoutError("timeout")
    r = run_turn(Session([], [], []), "x", boom, P, None)
    assert "error" in r.text.lower()

def test_multiturn_keeps_front():
    s = Session([], [], [])
    run_turn(s, "trade-off", scripted(call("pareto", {"objectives": [{"target": "ys"}, {"target": "el"}]}), say("Front ready.")), P, None)
    assert s.last_front
```

- [ ] **Step 2: 실패 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_agent.py -v`
  - Expected: FAIL

- [ ] **Step 3: `llm.py`, `agent.py` 구현**
  - **세션 시작:**
    - 첫 턴이면 SYSTEM_PROMPT를 넣는다.
    - `last_front`가 있으면 그 요약(id, 조성, μ)을 user 메시지 앞에 짧게 붙인다.
  - **루프:**
    1. `chat`을 호출한다.
    2. tool_calls가 있으면 각각 `dispatch`로 실행하고, 결과는 `role: tool` 메시지로 추가한 뒤 trace에도 남긴다.
    3. search 결과에 `binding`이 있으면 refine 횟수를 1 올린다. `MAX_REFINE`을 넘은 뒤의 search에는 `{"error": "refine limit reached; answer with what you have"}`를 돌려준다.
    4. 툴 호출이 `MAX_TOOL_CALLS`에 닿거나 토큰이 상한에 닿으면 tools 없이 한 번 더 부르거나, 그게 안 되면 마지막 content를 쓴다. 이때 `capped=True`로 표시하고 trace에 `"cap reached"`를 남긴다.
  - **trace 형식:**
    - search 결과에 binding이 있으면 `summary`에 `"binding <target> short by <x>"`를 넣는다. Task 7이 이 값을 쓴다.
    - cap에 걸렸을 때 남기는 항목은 `tool=None`, `status="cap reached"`로 둔다.
  - **표:** 마지막 search의 `candidates`나 pareto의 `front`를 `table`로 쓰고, `session.candidates`와 `last_front`를 갱신한다.
  - **검증:**
    1. 이번 턴 툴 출력과 사용자 텍스트로 검증한다.
    2. 실패하면 `"Verifier: <issues>. Rewrite using only tool numbers."`를 user 역할로 붙여 1회 다시 부른다.
    3. 또 실패하면 `strip_unsupported`를 적용한다.
  - **예외 처리:** chat 예외는 잡아서 `text = f"LLM error: {e}"`로 턴을 끝낸다.
  - **로그:** `log_path`가 있으면 턴의 메시지, trace, 토큰, 시간을 JSON 한 줄로 추가한다.

- [ ] **Step 4: 통과 확인**
  - Run: `.venv/Scripts/python -m pytest -v`
  - Expected: 전체 통과

- [ ] **Step 5: 실제 API 스모크 테스트 (한 번만)**
  - Run:

    ```bash
    .venv/Scripts/python -c "from dotenv import load_dotenv; load_dotenv(); from alloy_agent.agent import *; from alloy_agent.llm import make_chat; from alloy_agent.predictor import OpenPredictor; r=run_turn(Session([],[],[]),'Find steels with yield strength at least 1800 MPa and elongation at least 10%.',make_chat(),OpenPredictor()); print(r.text); print(render_table(r.table)); print(r.trace, r.issues)"
    ```

  - Expected: trace에 search가 있고, 표가 비어 있지 않고, issues가 비어 있거나 strip이 적용되어 있다.
  - Meta 백엔드도 `LLM_BASE_URL=https://api.meta.ai/v1 LLM_MODEL=muse-spark-1.3`으로 한 번 돌려 본다.

- [ ] **Step 6: 커밋**

```bash
git add src/alloy_agent/llm.py src/alloy_agent/agent.py tests/test_agent.py .env.example
git commit -m "feat: agent loop with caps, verifier retry and run logging"
```

---

### Task 5: 스펙 시트 intake (멀티모달)

**Files:**
- Create: `src/alloy_agent/intake.py`
- Create: `demo/spec_landing_gear.pdf`, `demo/spec_table.png`: 직접 만든 데모 파일이다. PDF는 reportlab 없이 Pillow로 그린 PNG를 저장하거나, 짧은 텍스트 PDF로 만든다.
- Test: `tests/test_intake.py`

**Interfaces:**
- Consumes: `tools.Constraint`
- Produces:
  - `class Constraints(BaseModel): targets: list[Constraint]; element_caps: dict[str, float] = {}`
  - `def extract_constraints(path: str, chat_raw: Callable[[list[dict]], str] | None = None) -> Constraints`
    - 기본 chat_raw는 Meta `muse-spark-1.3`에 대한 단발 호출이다.
    - 응답은 JSON만 받는다(시스템 프롬프트로 스키마를 지시한다). 파싱 실패 시 `ValueError`를 낸다.
  - `def constraints_to_prompt(c: Constraints) -> str`: 에이전트에 넘길 영어 요청 문장을 만든다.

- [ ] **Step 1: 스파이크 — Meta API가 PDF와 이미지를 받는 형식 확인**
  - 공식 문서(`https://ai.developer.meta.com/docs/`)에서 이미지와 PDF content part 형식을 확인한다. 후보는 `image_url`에 base64 data URL을 넣는 방식과 `file` part 방식이다.
  - 데모 PNG 1장으로 한 번 호출해서 확인한다. 확인한 형식은 `intake.py` 상단 주석에 한 줄로 남긴다.
  - PDF가 안 되면 PDF는 지원하지 않고 이미지만 받는다(스펙의 절삭 대상 2번).

- [ ] **Step 2: 실패하는 테스트 작성 (LLM 없이)**

```python
def test_parse_valid():
    raw = '{"targets": [{"target": "ys", "min": 1700}, {"target": "el", "min": 8}], "element_caps": {"co": 10}}'
    c = extract_constraints("demo/spec_table.png", chat_raw=lambda m: raw)
    assert c.targets[0].min == 1700 and c.element_caps == {"co": 10}
    assert "1700" in constraints_to_prompt(c)

def test_parse_invalid():
    with pytest.raises(ValueError):
        extract_constraints("demo/spec_table.png", chat_raw=lambda m: "sorry")
```

- [ ] **Step 3: 실패 확인 → 구현 → 통과 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_intake.py -v`
  - Expected: 2 passed

- [ ] **Step 4: 실제 호출 확인**
  - 데모 파일 2개로 실제 호출해서 Constraints가 나오는지 확인한다.

- [ ] **Step 5: 커밋**

```bash
git add src/alloy_agent/intake.py tests/test_intake.py demo/
git commit -m "feat: spec-sheet intake with Muse Spark multimodal"
```

---

### Task 6: Gradio UI

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `Session`, `run_turn`, `render_table`, `make_chat`, `OpenPredictor`, `extract_constraints`, `constraints_to_prompt`
- Produces: `python app.py`를 실행하면 `http://127.0.0.1:7860`이 열린다.

- [ ] **Step 1: `app.py` 구현**
  - `gr.Blocks(title="alloy-agent")`
  - **왼쪽:**
    - `gr.Chatbot`
    - `gr.Textbox`
    - `gr.File(file_types=[".pdf", ".png", ".jpg"])`
    - backend `gr.Radio(["Muse Spark Open (Baseten)", "Muse Spark 1.3 (Meta)"])`
  - **오른쪽:**
    - 트레이스 `gr.JSON`
    - `gr.ScatterPlot(x="YS", y="El", color="OOD")`: 데이터는 table/front에서 만든다.
    - 후보 `gr.Markdown(render_table)`
  - **상태:**
    - `gr.State`에 `Session`을 둔다.
    - predictor는 모듈 로드 시 한 번 학습한다.
  - **업로드:** 파일을 업로드하면 `extract_constraints` → `constraints_to_prompt` 결과를 입력창에 채운다. 실패하면 `gr.Warning`을 띄운다.
  - **Replay 모드:** 스펙의 데모 대비책이다. 환경변수 `REPLAY=runs.jsonl`이 있으면 LLM을 호출하지 않는다. 대신 저장된 턴을 순서대로 다시 보여준다(text, table, trace).
  - **footer:** "Predictions from a model trained on 312 open steels (MIT). ±2σ calibrated to ~95% out-of-fold coverage. Out-of-domain candidates are flagged."

- [ ] **Step 2: 수동 확인**
  - `python app.py`를 실행한 뒤 아래를 확인한다.
    1. "yield ≥ 1800 MPa, elongation ≥ 10%"를 입력하면 표, 트레이스, 산점도가 표시된다.
    2. 후속 질문 "more elongation"을 보내면 이전 전선을 바탕으로 재탐색한다.
    3. 데모 PNG를 업로드하면 입력창이 채워진다.
  - 세 시나리오 스크린샷을 `docs/screens/`에 저장한다.

- [ ] **Step 3: 커밋**

```bash
git add app.py docs/screens/
git commit -m "feat: Gradio UI with trace, Pareto scatter and spec upload"
```

---

### Task 7: 평가

**Files:**
- Create: `eval/queries.jsonl`, `eval/run_eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `run_turn`, `make_chat`, `OpenPredictor`
- Produces:
  - `eval/queries.jsonl`: 한 줄에 `{"id": str, "query": str, "followup": str | null, "expect": "feasible" | "infeasible"}`
    - 영어로 20개를 만든다: feasible 8, infeasible 4, 트레이드오프 4, OOD 유도 2, followup 포함 2.
  - `def score_turn(r: TurnResult, expect: str) -> dict`
    - 반환: `{"success": bool, "tool_calls": int, "ood_rate": float, "verifier_failures": int, "capped": bool}`
    - success 조건:
      - expect=feasible이면 table[0]["score"] ≥ 0이고 issues가 없어야 한다.
      - expect=infeasible이면 binding이 나왔고, 본문에 binding target 이름이 있어야 한다.
  - CLI: `python eval/run_eval.py --backend baseten|meta --limit N` → `eval/results_<backend>.csv`
    - 열: id, success, tool_calls, latency_s, tokens, ood_rate, verifier_failures, capped
  - `python eval/run_eval.py --plot` → 두 CSV를 비교하는 `eval/compare.png`(matplotlib)
    - matplotlib은 `pyproject.toml` dependencies에 추가한다.
  - 툴은 seed로 결정적이므로, 스펙에 있던 툴 출력 캐시는 만들지 않는다. 같은 입력이면 같은 출력이 나온다.

- [ ] **Step 1: 실패하는 테스트 작성** (`score_turn`만 LLM 없이 테스트)

```python
def test_score_feasible():
    r = TurnResult("ok", [{"score": 0.1, "ood": False}], [{"tool": "search"}], [], False)
    assert score_turn(r, "feasible")["success"]

def test_score_infeasible_needs_binding_mention():
    r = TurnResult("Yield is the binding constraint (ys).", [{"score": -0.2, "ood": False}], [{"tool": "search", "summary": "binding ys"}], [], False)
    assert score_turn(r, "infeasible")["success"]
```

- [ ] **Step 2: 실패 확인 → 구현 → 통과 확인**
  - Run: `.venv/Scripts/python -m pytest tests/test_eval.py -v`
  - Expected: 2 passed

- [ ] **Step 3: 소량 실행**
  - `--backend baseten --limit 3`으로 CSV가 만들어지는지 확인한다.
  - 비용은 Baseten 크레딧에서 빠진다.

- [ ] **Step 4: 커밋**

```bash
git add eval/queries.jsonl eval/run_eval.py tests/test_eval.py
git commit -m "feat: eval harness with per-backend CSV and comparison plot"
```

전체 20개 × 백엔드 2개 실행은 11/2에 한다. 결과 CSV와 png는 그때 커밋한다.

---

### Task 8: Rediscovery 벤치마크 (자를 때 1순위)

**Files:**
- Create: `eval/rediscovery.py`

**Interfaces:**
- Consumes: `data.load`, `OpenPredictor(rows=...)`, `search`, `Constraint`
- Produces: `python eval/rediscovery.py`를 실행하면 `eval/rediscovery.csv`가 생기고 요약 한 줄이 출력된다.

- [ ] **Step 1: 구현**
  1. YS 상위 10% 행을 hold-out으로 빼고 나머지로 `OpenPredictor`를 학습한다.
  2. hold-out 행마다 `search(constraints=[ys ≥ 실제 YS의 90%, el ≥ 실제 El의 90%])`를 실행한다.
  3. 상위 5개 후보 중 하나라도 해당 hold-out 행과의 표준화 거리가 학습 세트 최근접 이웃 거리의 중앙값 이하이면 `found=True`로 본다.
  4. 출력은 `rediscovered {found}/{total}`이다. 0이어도 그대로 보고한다.

- [ ] **Step 2: 실행해서 숫자 확인 → 커밋**

```bash
git add eval/rediscovery.py eval/rediscovery.csv
git commit -m "feat: held-out rediscovery benchmark (no LLM)"
```

---

### Task 9: README와 제출 문서

**Files:**
- Modify: `README.md`, `DATA.md`

- [ ] **Step 1: README 갱신**
  - 넣을 내용:
    - 한 줄 소개, 아키텍처 그림(스펙 다이어그램)
    - 실행 방법(`.env` 설정, `python app.py`)
    - 백엔드 전환 방법
    - 예측기 성능: Task 1의 coverage, MAE
    - 평가 표: Task 7 CSV 요약, compare.png
    - rediscovery 결과
    - 한계: 데이터 312건, 마레이징 위주, 실측 검증 없음
    - "Built during the hackathon" 섹션: 날짜별 커밋 범위와 "no pre-existing code"
    - 개발 중 OpenCode + Muse Spark Open을 보조로 썼다면 그 사실
- [ ] **Step 2: 확인**
  - `git ls-files`에 `.env`, `data/`, `runs.jsonl`이 없어야 한다.
- [ ] **Step 3: 커밋**

```bash
git add README.md DATA.md
git commit -m "docs: README with results, limits and built-during-hackathon section"
```

---

## 일정 대응

| 날짜 | Task |
|---|---|
| 10/28 | 1, 2 |
| 10/29 | 3, 4, 5 |
| 10/30 | 6, 7, 8 |
| 11/2 | 평가 전체 실행, 데모 녹화, 9 |

**절삭 순서:** Task 8 → Task 5 → Task 6의 산점도(표로 대체) → Task 7의 질의 수를 10개로 축소
