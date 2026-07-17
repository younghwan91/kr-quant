# PEAD 정교화 실험 — 사전등록 + 설계 결정 기록

2026-07-13에 수행한 PEAD 3단계 정교화 실험의 **사전등록 기록**과 회계 관례.
**결과는 `research/PEAD_REFINEMENT_RESULTS.md`(결론: 개선 없음), 스크립트는
`research/pead_refinement.py`.** 이 문서는 "무엇을 데이터 보기 전에 정해뒀는가"를 남긴다 —
결과 문서의 정직성이 여기에 의존한다.

원 계획서는 RALPLAN-DR 컨센서스로 작성됐고, 마이그레이션 절차 등 실행 스크래치는 로컬에만 있다.

## 사전등록된 실험 3개

베이스라인: `staggered_backtest(top_n=40, adv_floor=20000, horizon=60, step=20)`,
전체 확장 유니버스(2,629 DART 종목, 2016Q1~2026).

1. **실적 서프라이즈 크기 필터** — YoY 크기로 사전 필터링 후 스윕.
2. **보유 지평선 스윕** — H60 최적성 점검.
3. **포지션별 트레일링 스탑** — 낙폭 축소 목적.

## Guardrails (사전 확정)

**반드시:** 시점 기준(`avail_date`) 신호만(룩어헤드 금지) · 트레일링 ADV 유동성 하한(as-of, 미래 아님) ·
전체 확장 유니버스 · 아래 회계 조화.

**절대 금지:** `src/kr_quant/` 파일 수정 · 컨센서스 기반 서프라이즈(데이터 미확보) ·
포트폴리오 레벨 낙폭 서킷브레이커(스코프는 포지션별) ·
**인샘플 최적화된 파라미터를 robust한 것처럼 제시하기.**

## ADR — Option C: `staggered_backtest` 위의 수익률 조정 레이어

**결정:** 실험 3개 모두 프로덕션 `staggered_backtest`를 재사용하고, 트레일링만 사후 수익률
조정으로 얹는다. 리밸런스 선정·트랜치 구조·벤치마크·트랜치 평균·`_summarize()` 호출은 전부
프로덕션 기계를 그대로 쓴다.

**기각한 대안:**
- *Option A(커스텀 트레일링 루프)* — 스태거 루프를 독립 재구현하면 `staggered_backtest`의 관례와
  정확히 일치시켜야 하는 **두 번째 코드 경로**가 생긴다. 이게 핵심 회계 불일치 위험.
- *Option B(완전 독립)* — 위 문제의 최대치.

**선택 이유:** Option C는 벤치마크·스태거 구조·연율화를 **구성상(by construction)** 상속한다.
새로 짜는 코드는 포지션별 일봉 워크뿐 — 트레일링 스탑에 필요한 기약 최소치다.

**결과:** 리서치 스크립트가 `kr_quant.strategies.pead` 내부(패널 빌드, eligible/book 로직)에
의존한다. 내부가 바뀌면 깨질 수 있다(스크래치 리서치라 수용). `eligible()`/`book()`은 클로저라
import 불가 → ~20줄 일회성 복사, 감사용 줄 참조 명시.

## 회계 조화 (단일 진실 원천)

세 실험이 직접 비교 가능하려면 아래를 따라야 한다. **어떤 이탈도 버그다.**

> **줄 참조 주의:** 실험 당시 회계 로직은 `pead.py`에 있었으나, 이후 백테스팅 엔진
> 마이그레이션(Step 3, `88aff84`)으로 `engine/sim_crosssectional.py`의 `staggered_backtest`
> 시뮬레이션으로 이전됐다. 아래는 **현재 위치** 기준이다. 공개 진입점은
> `pead.py:114`(엔진으로 얇게 위임). 수치는 패리티 테스트로 불변이 확인됐다.

**(a) 1차 비교는 무비용.** `staggered_backtest`는 `gross`와 `net`에 **같은 값**을 넣는다
(`sim_crosssectional.py:187~188` — 비용 미부과). 세 실험 모두 1차 스윕 표를 무비용으로 보고해
이에 맞춘다.

**(b) 2차 비용조정 컬럼(실험 3 한정).** 트레일 조기청산마다 `cost_one_way = 0.0023`을 부과한
`Sharpe_cost`/`t_cost`/`cum_net_cost`를 별도 보고. 잦은 트레일 청산의 실제 비용 영향을
1차 무비용 비교를 오염시키지 않고 드러내기 위함.

**(c) 진입가 = `close[t]`.** `staggered_backtest`는 `C[:, t+step] / C[:, t] − 1.0`
(`sim_crosssectional.py:179`). 진입은 리밸런스일 **종가**이지 익일 종가가 아니다. 트레일링 일봉
워크도 `close[t_entry]`를 진입가로 써야 한다.

**(d) 조기청산 수익률 회계.** 트레일이 `d`일(`d < step`)에 발화하면:
- 포지션 수익률은 `close[t+d] / close[t] − 1.0` (진입가 대비 부분기간 수익률).
- 나머지 기간(`d+1`~`step`)은 **현금 보유**로 처리 — 재투자·추가 노출 없음.
- **트랜치 벤치마크는 줄이지 않는다** — `staggered_backtest`와 동일한 전체 step 윈도
  유니버스 평균(`sim_crosssectional.py:179~180`, `bench = nanmean(ret[uni])`)을 쓴다. 그래야
  초과수익이 비교 가능하다: 조기청산으로 후반 낙폭을 피했으면 초과수익↑, 후반 랠리를 놓쳤으면
  초과수익↓로 정직하게 나타난다.

**(e) 연율화 분모.** `summarize_periods()`는 항상 `step`으로 호출한다(`horizon`/`max_horizon`
아님) — `staggered_backtest`와 동일(`sim_crosssectional.py:190`). `per_year = 252 / step`.

## 결론

세 실험 모두 **개선 없음**. 상세는 `research/PEAD_REFINEMENT_RESULTS.md`.
후속: 트레일링이 유의미했다면 `staggered_backtest`에 선택적 `trail_pct` 파라미터를 별도 승인 PR로
추가할 예정이었으나, 개선이 없어 진행하지 않는다.
