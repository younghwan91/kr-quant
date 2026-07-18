# 새 알파 실험 — 정규 흐름 (TEMPLATE)

한 알파는 반드시 이 파이프라인을 흐른다. 각 단계는 디렉터리로 강제된다.
**go/no-go는 리포터의 숫자를 읽고 사람이 내린다 — 하드코딩된 verdict가 아니다.**

```
가설 → 신호(research/signals/) → 백테스트 → 검증(kr_quant.validation)
      → 진단(kr_quant.diagnostics) → go/no-go(리포트를 읽고 판단) → 로그(research/logs/)
```

## 5원칙 (비협상)

1. **확률적, 결정론 아님.** 한 백테스트 경로 = 노이즈 분포의 한 표본. 결정론적 극값을 좇지
   말 것. fold/표본 재현성으로 판정. 로버스트 목적함수(부트스트랩 하단), 미니마 전에 멈춤.
2. **평균이 아니라 건당 R-멀티플 분포.** 각 트레이드가 표본. 분포 모양(−1R 왼꼬리 절단,
   오른꼬리 두께)을 본다. 포트폴리오 프레이밍(슬롯·동시보유·연환산-슬롯당) 금지.
3. **취약성은 1급 지표.** monster 의존도(상위5 P&L 비중), 최장 연패, 꼬리제거 민감도.
   양의 기대값이 ~5개 꼬리에 몰리면 통계신뢰 낮음.
4. **no-lookahead는 신성.** 선별 임계값은 TRAIN에서만 학습해 앞으로 적용. TRAIN 경계 =
   진입일 < 2022-01-01. 신호 lag=1.
5. **보수적 배포 게이트.** "유망하니 페이퍼트레이드"는 "실자금 배포"가 아니다. 한국 급등주
   슬리피지는 모델치를 초과할 수 있다.

## 1단계 — 신호 (research/signals/)

`build_*_signal` / `load_data` / (신호 특정이면) 시뮬레이터. 진입시점에 알 수 있는 값만 쓴다.
경계: `src/kr_quant/`는 `research/`를 import하지 않는다 — 반대만 허용.

## 2단계 — 백테스트

신호를 (수익, 진입일) 트레이드 표본으로 변환. `params dict -> (returns, entry_dates)` 콜러블
(`simulate`)을 만들어 검증/최적화에 넘긴다.

```python
def make_sim(prices, flow, cache):
    def simulate(params: dict):
        return simulate_fast(prices, flow, target=0.0, _cache=cache, **params)
    return simulate
```

## 3단계 — 검증 (kr_quant.validation)

```python
from kr_quant.validation.walkforward import FOLDS, oos_fixed, walk_forward, fold_consistency
from kr_quant.validation.sensitivity import sensitivity_table, oos_sensitivity
from kr_quant.validation.optimization import make_objective, mini_bo, TRAIN_HI, TRADE_FLOOR
```

- `FOLDS` — frozen 롤링 6-fold(`rolling_folds()`). **실험마다 새로 만들지 말 것**(fold-shopping).
- `make_objective(simulate, space, train_hi=TRAIN_HI, floor=...)` — 목적함수 = 부트스트랩 2.5%
  하단(`_boot_lower`, sacred). 원시 평균 최적화 금지. TRAIN(<train_hi)만 봄.
- `walk_forward(FOLDS, simulate, fit)` — fold TRAIN서 fit → TEST 평가. IS≫OOS면 과최적.
- `sensitivity_table` / `oos_sensitivity` — 1개씩 흔들어 플래토(강건) vs 스파이크(과최적).

## 4단계 — 진단 (kr_quant.diagnostics)

```python
from kr_quant.diagnostics.r_distribution import (
    r_multiples, dist_shape, selection_curve, conviction_analysis, hold_curve)
from kr_quant.diagnostics.fragility import (
    monster_share, max_loss_streak, tail_removal, fragility_report)
from kr_quant.diagnostics.gate_report import gate_report
```

- `dist_shape(R)` — 왼꼬리 절단·오른꼬리 두께·왜도. `r_multiples(ret, stop)`으로 R 생성.
- `selection_curve` / `conviction_analysis` — 진입시점 확신 신호가 분포를 오른쪽으로 굽나(a-priori).
- `fragility_report(R)` — monster-share·연패·꼬리제거. 소수 꼬리 의존이면 신뢰↓.
- `gate_report(...)` — 배포 준비도 **신호를 REPORT**한다(OOS 부트스트랩 CI, 폴드-벤드 수,
  monster-share, 연패, 엣지가 죽는 비용). **PASS/FAIL을 emit하지 않는다** — 하드코딩된 임계값
  자체가 결정론적 과최적(Principle 1). 숫자를 읽고 사람이 판단.

## 5단계 — go/no-go & 로그 (research/logs/<alpha>/)

`gate_report`의 분포를 읽고 판단한다:
- OOS 부트스트랩 하단 > 0 인가? 폴드 재현(≥과반)인가? monster-share가 낮은가?
- 현실 비용까지 엣지가 사는가? — 아니면 배포하지 않는다.

결과를 `research/logs/<alpha>/VERDICT.md`에 기록(가설·파라미터·판정·날짜·재현 커맨드).
정직한 부정 결과도 산출물이다 — 예: [`logs/contrarian_retail/VERDICT.md`](logs/contrarian_retail/VERDICT.md)
(엣지가 fold-재현되지 않아 NO-GO, sim 추출 취소).

## 참고 실험 (얇은 러너 예시)

- `experiments/contrarian_bo.py` — `make_objective`로 BO(과최적 방지 목적함수는 라이브러리).
- `experiments/contrarian_validate.py` — `FOLDS`·`sensitivity_table`·`mini_bo`.
- `experiments/contrarian_distribution.py` — `dist_shape`·`selection_curve`·`conviction_analysis`.
- `experiments/contrarian_selective.py` — `fold_consistency`(no-lookahead 선별 재현성).
- `experiments/slippage_check.py` — 비용 스윕 게이트(`fold_slices`).
