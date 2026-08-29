# 백테스팅 엔진 (`kr_quant.engine`) — 설계 결정 기록

성과지표와 시뮬레이션 루프의 유일 구현. 이 문서는 **왜 이렇게 만들었는지**와 **확정된 설계 결정**을
남긴다. 실행 이력은 git log 에 있다.

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
    recipe.py              # ExperimentConfig dataclass, run_recipe() 디스패처
```

엔진이 굴리는 패러다임은 **횡단면(rank-tilt) 하나뿐**이다. 이벤트드리븐 워크가 필요하면
`recipe.py`에 `_run_*` 분기를 추가할 것 — 전략 파일에 회계 루프를 새로 쓰는 게 아니라.

엔진은 **리프 패키지**다 — numpy/pandas만 import하고, 전략이 엔진을 import한다(역방향 금지).

## ADR

**결정:** 공용 지표 + 횡단면 시뮬레이션 모듈 + 패널 캐싱 + 선언적 recipe API를 갖춘
`engine` 패키지를 만들고, 전략 파일을 회귀테스트 우선 원칙으로 하나씩 이전한다.

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
**시뮬레이션 모듈을 분리**하고, 파일별 이전 + 패리티 테스트로 관리한다.

**결과:** 엔진 지표를 바꾸면 전 하류에 영향이 간다. 지금 엔진의 하류는 `strategies/*` 와
`research/` 러너들이다.

## 확정된 설계 결정 (2026-07-17)

구현 중 "만들면서 정한다"로 미뤄뒀던 6건을 실제 코드 근거로 확정했다. 계획 기본안 유지 2건,
구현 중 결정 4건 — 아무것도 계획을 뒤집지 않았다.

| 질문 | 결정 | 근거 |
|---|---|---|
| 패리티 테스트 영구 보존? | **보존** | `tests/test_parity_pead.py`. 엔진이 발표 수치를 안 바꿨음을 증명하는 유일한 장치 — 회귀 보험 값 > 유지 비용 |
| 패널 캐싱 전략? | **콘텐츠 키 LRU** | `panels.py`의 `PanelCache`. DataFrame이 unhashable이라 직접 구현 |
| `ExperimentConfig` 스키마? | **dataclass** | `recipe.py`. 검증·IDE 지원 |
| 재export에 deprecation 경고? | **미추가** | 공개 API 는 `engine/__init__.py` 재export 하나로 모은다 |

**캐싱 상세:** 키는 `(value 컬럼, shape, pd.util.hash_pandas_object의 sha1 다이제스트)` — 내용이 같고
객체만 다른 프레임도 캐시를 공유한다. `PanelCache`(OrderedDict LRU, maxsize 32, hits/misses 카운터) +
모듈 레벨 `PANEL_CACHE`. 스윕에서 같은 프레임을 재피벗하지 않고, 세션 간에는 `PANEL_CACHE.clear()`.

## 후속 과제

- `research/experiments/pead_refinement.py` 스크래치 스크립트를 recipe API로 이전.
- 테스트 DB 픽스처가 생기면 실데이터 골든 아웃풋 CI 테스트 추가.
- 비용 모델이 복잡해지면 `engine/cost.py` 분리 (현재는 `cost_one_way * turnover`).

> **엔진↔래퍼 동치 보증은 one-shot 검수 스크립트가 아니라 테스트가 한다.** `tests/test_parity_pead.py`
> 가 DB 없이 CI 에서 매번 돈다. 한 번 돌리고 마는 검수 스크립트는 곧 아무도 호출하지 않게 되고,
> 그러면 깨진 것도 모른 채 남는다.
