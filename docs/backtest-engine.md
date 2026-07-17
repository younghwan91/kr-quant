# 백테스팅 엔진 (`kr_quant.engine`) — 설계 결정 기록

성과지표와 시뮬레이션 루프의 유일 구현. 이 문서는 **왜 이렇게 만들었는지**와 **확정된 설계 결정**을
남긴다. 실행 이력은 git log `016f426..eca4325`(Phase 1 ~ Step 6), 원 마이그레이션 절차서는
로컬 스크래치(`.omc/plans/backtest-engine-plan.md`, RALPLAN-DR 2회 컨센서스 검증)에 있다.

## ⚠️ 사용 규칙 (필독)

**새 백테스트 실험은 반드시 `kr_quant.engine`의 지표·시뮬레이션 함수를 재사용하고, 회계 로직
(진입가·벤치마크·비용·연율화)을 새로 구현하지 않는다.** 새 실험은 `examples/pead_sweep_via_recipe.py`
처럼 `engine.recipe.ExperimentConfig` + `run_recipe()`로 파라미터만 정의하는 걸 먼저 시도할 것.
전략 파일을 통째로 읽고 회계 로직을 손으로 베끼는 건 이 엔진을 만든 이유 자체를 무시하는 것이다.

## 패키지 구조

```
src/kr_quant/engine/
    __init__.py            # 공개 API 재export
    metrics.py             # ann_sharpe, cagr, max_drawdown, newey_west_t,
                           # summarize_periods, spearman, quantile_summary,
                           # paired_bootstrap, regime_buckets
    panels.py              # 패널 피벗 헬퍼, ADV 계산, 세션 LRU 캐시
    sim_crosssectional.py  # rank_tilt_backtest, staggered_backtest, rank_ic
    sim_eventdriven.py     # 미너비니식 이벤트드리븐 워크:
                           # position_walk(stop_price 절대값, ma50 명시 입력), trade_runner
    recipe.py              # ExperimentConfig dataclass, run_recipe() 디스패처
```

엔진은 **리프 패키지**다 — numpy/pandas만 import하고, 전략이 엔진을 import한다(역방향 금지).

## ADR

**결정:** 공용 지표 + 두 개의 분리된 시뮬레이션 모듈(횡단면 / 이벤트드리븐) + 패널 캐싱 +
선언적 recipe API를 갖춘 `engine` 패키지를 만들고, 5개 전략 파일을 회귀테스트 우선 원칙으로
하나씩 이전한다.

**결정 동인:**
1. **재구현 없는 재사용** — 모든 지표·시뮬레이션 루프를 파라미터로 호출 가능하게 해서, 미래 연구가
   회계 관례를 다시 유도하지 않게 한다.
2. **수치 무회귀** — README/MULTI_ALPHA의 발표 수치(CAGR +19.3% / Sharpe 1.06 /
   MaxDD −24%, PEAD t-stat 등)는 외부에 인용된다. 조용한 드리프트는 배포 차단 실패다.
3. **점진적 안전성** — 5개 파일 빅뱅 재작성은 허용 불가. 파일 경계마다 테스트가 초록이어야 한다.

**기각한 대안:**
- *지표만 추출* — 위험은 최소지만 시뮬레이션 루프 재사용·recipe API를 못 푼다.
- *지표+패널만, 시뮬레이션 추상화 연기* — 복붙 문제가 시뮬레이션 루프에 그대로 남는다.

**선택 이유:** 스펙 수용 기준을 모두 만족하는 유일안. "잘못된 통합"의 위험은 단일 다형 베이스 대신
**시뮬레이션 모듈을 둘로 분리**하고, 파일별 이전 + 패리티 테스트로 관리한다.

**`sim_eventdriven.py`의 정직한 범위:** `position_walk`의 파라미터(`tennis`, `tennis_window`,
`sell_half_r`, `pe_array`, `breakeven`)는 **미너비니 전용 청산 규칙**이지 범용 이벤트드리븐 인터페이스가
아니다. 현재 다른 전략은 이 함수를 그대로 재사용하지 않는다. 이 모듈은 "미너비니식 이벤트드리븐 워크"로
읽어야 하며, 범용 시뮬레이터가 아니다.

**결과:**
- 엔진은 5개 전략 파일의 강한 의존성이 된다 — 엔진 지표 변경은 전 하류에 영향.
- 스코프 밖 파일(`supply_wave.py`, `multi_signal.py`)은 여전히 `backtest.py`의 재export를 통해
  `spearman`/`forward_returns`를 import한다.

## 확정된 설계 결정 (2026-07-17)

구현 중 "만들면서 정한다"로 미뤄뒀던 6건을 실제 코드 근거로 확정했다. 계획 기본안 유지 2건,
구현 중 결정 4건 — 아무것도 계획을 뒤집지 않았다.

| 질문 | 결정 | 근거 |
|---|---|---|
| 패리티 테스트 영구 보존? | **보존** | `tests/test_parity_*.py` 5개. 엔진이 발표 수치를 안 바꿨음을 증명하는 유일한 장치 — 회귀 보험 값 > 유지 비용 |
| `book_returns`/`benchmark_returns` 엔진 이전? | **`sepa_compare.py` 잔류** | `sepa_compare.py:71`, `:185`. 수익률 시계열 집계는 지표가 아니라 전략 회계 — 최소 diff |
| `build_panels` 캐싱 전략? | **콘텐츠 키 LRU** | `panels.py:92~139`. DataFrame이 unhashable이라 직접 구현 |
| `pivot_fill`/`pivot_fills` 엔진 이전? | **`minervini_sepa.py` 잔류** | `:26`, `:46`. 미너비니 진입 로직과 강결합 — 엔진 `trade_runner`가 호출 |
| `ExperimentConfig` 스키마? | **dataclass** | `recipe.py:51`. 검증·IDE 지원 |
| 재export에 deprecation 경고? | **미추가** | `backtest.py:35~37`, 이유는 `:29~34` 주석. 죽은 리서치 파일에 소음 만들 이유 없음 |

**캐싱 상세:** 키는 `(value 컬럼, shape, pd.util.hash_pandas_object의 sha1 다이제스트)` — 내용이 같고
객체만 다른 프레임도 캐시를 공유한다. `PanelCache`(OrderedDict LRU, maxsize 32, hits/misses 카운터) +
모듈 레벨 `PANEL_CACHE`. `build_panels(..., use_cache=True)`로 스윕에서 재피벗 회피, 세션 간에는
`PANEL_CACHE.clear()`.

## `position_walk` 불변식 6개 (이벤트드리븐 워크)

원래 `minervini_sepa.py`의 `_walk` 클로저가 갖던 동작. **`position_walk`에서 정확히 보존돼야 하며,
"개선"하면 백테스트 수치가 조용히 바뀐다.** 각 항목은 `tests/test_engine_sim_event.py`가 지킨다.
이상해 보이는 코드를 고치기 전에 여기부터 읽을 것.

1. **청산 규칙 우선순위** — 봉마다 `손절(L≤stop)` > `PE 확장` > `테니스공 컬` >
   **`sell_half`/브레이크이븐 상향(청산 아님 — 상태 변경 후 통과)** > `violations`/`climax_run` >
   `time_cap`(루프 후 폴백) 순으로 발화. sell_half는 단독 청산이 아니다 — 한 봉이 sell_half를 트리거
   *하면서* 동시에 violations/climax로 청산될 수 있다.
   *실수:* violations를 손절보다 먼저 검사하거나, sell_half 뒤에 `break`를 넣어 같은 봉의 청산 감지를 막는 것.
2. **violations와 climax_run의 윈도 비대칭** — `violations()`는 최근 윈도(`C[i, max(0, t−ma_exit_window):t+1]`),
   `climax_run()`은 **0번 봉부터의 전체 이력**(`C[i, :t+1]`)을 받는다.
   *실수:* 둘에 같은 윈도를 넘기는 것. (`climax_run`이 두 번 호출되는 건 청산 사유 문자열을
   "climax"/"violation"으로 구분하기 위함 — 단일 캐시 호출로 "최적화"하지 말 것.)
3. **`half_target`은 진입 시 한 번만 계산, 재계산 금지** — 초기 `stop0`에서 산출된다. 이후 stop이
   브레이크이븐으로 상향돼도(`stop = max(stop, raised)`) `half_target`은 원래 값을 유지한다.
   *실수:* stop 상향 후 half_target을 "친절하게" 재계산 → sell_half 트리거 레벨이 바뀌어 수익률이 달라짐.
4. **테니스공 윈도는 NaN 스킵 봉을 포함한 달력 봉을 센다** — `(t − f0) >= tennis_window`는 진입 이후
   모든 봉을 센다(NaN이라 `continue`된 봉 포함). 유한 봉 카운터가 아니라 달력 시간에 발화한다.
   *실수:* 비NaN 봉만 세는 별도 카운터 → 테니스 컬이 지연됨.
5. **갭 관통 손절 체결가 표현식** — `min(OPN, stop) if OPN < stop else stop`. 대수적으론 `min(OPN, stop)`과
   같지만, 갭 관통이 의도적으로 처리됐음을 감사자가 알 수 있게 조건문 형태(또는 동치성 명시 주석)를 유지한다.
6. **스태거드 블렌딩 관례** — 3개 레그(`STAGGERED_STOPS = (0.04, 0.06, 0.08)`)를 돌려 수익률은
   **3개 평균**, `exit_price`/`exit_date`는 **마지막 레그(8% 손절)에서만** 가져오고, `reason`은 어느 규칙이
   실제 발화했든 **리터럴 `"staggered"`로 하드코딩**한다. 자의적 관례지만 그대로 둔다.
   *실수:* 가장 이른 청산일을 쓰거나, 청산가를 평균 내거나, `legs[0]`을 쓰거나, `reason`을 `legs[-1][3]`에서
   읽는 것.

**`ma50`는 명시 입력** — 원본은 전 종목 `ma50`를 한 번 계산해 모든 `_walk` 호출이 공유했다. 엔진에서는
`position_walk`/`trade_runner`의 **명시적 파라미터**여야 한다(호출마다 재계산 금지) — 값 동일성과 성능 둘 다를 위해.

## 후속 과제 (전부 낮은 우선순위)

- `supply_wave.py`·`multi_signal.py`를 엔진으로 이전 (죽은 리서치 파일).
- `research/pead_refinement.py` 스크래치 스크립트를 recipe API로 이전.
- 테스트 DB 픽스처가 생기면 실데이터 골든 아웃풋 CI 테스트 추가.
- 비용 모델이 복잡해지면 `engine/cost.py` 분리 (현재는 `cost_one_way * turnover`).
