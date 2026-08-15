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

## 5단계 — 표준 심사 배터리 (research/experiments/*_gate.py)

새 알파는 **반드시** 아래 배터리를 통과해야 한다. 이건 문화적 관례가 아니라 규약이다 —
공용 하버스 [`experiments/prop_gate.py`](experiments/prop_gate.py)가 한 번에 다 낸다.
새 게이트는 이 하버스를 재사용해 **동일 잣대**로 심사한다(자체 배터리 재발명 금지).

- **사전등록 헤더** — VERDICT.md 맨 위에 결과 보기 전 config 1개·합격 바를 확정·커밋
  (GUARDRAILS §6 템플릿). 깃 히스토리가 증인. 결과 본 뒤 수정 금지.
- **건당 R-분포** — 자본곡선·복리 프레이밍이 아니라 `dist_shape(R)` + `fragility_report`.
  각 트레이드가 표본(원칙 2).
- **음성 대조 > 널** — 신호를 랜덤으로 망가뜨린 널(`random_entry_control`)을 같은 관문에
  태워 실제가 자기 널을 명확히 이기는지 본다. **폴드 수 단독 신뢰 금지** — 순수 노이즈도
  5/6을 46% 통과한다. 널 대조가 진짜 판별기다(§7 워크드 예제 미너비니 참고).
- **비용 스윕 + 2배 스트레스** — 왕복비용·2배 비용에도 엣지가 사는지(`cost_edge_dies`).
  특히 한국 급등주는 슬리피지가 모델치를 초과할 수 있다.
- **손 안 댄 창** — 탐색에 한 번도 안 쓴 held-out 최종 구간에서도 양수인지(walk-forward와 별개).

## 6단계 — go/no-go & 로그 (research/logs/<alpha>/)

`gate_report`의 분포를 읽고 판단한다:
- OOS 부트스트랩 하단 > 0 인가? 폴드 재현(≥과반)인가? monster-share가 낮은가?
- 음성대조를 이겼는가? 비용 2배·손 안 댄 창에서도 사는가? — 아니면 배포하지 않는다.

결과를 `research/logs/<alpha>/VERDICT.md`에 기록(가설·파라미터·판정·날짜·재현 커맨드).
정직한 부정 결과도 산출물이다 — 예: [`logs/contrarian_retail/VERDICT.md`](logs/contrarian_retail/VERDICT.md)
(엣지가 fold-재현되지 않아 NO-GO, sim 추출 취소).

> **경계·판정 린트(CI 강제).** `scripts/check_guardrails.py`가 CI 에서 막는 것:
> (a) src→research import, (b) VERDICT 없는 `*_gate.py`, (c) 하드코딩 "PASS"/"FAIL"
> 판정 문자열, (d) `storage` 밖에서 raw `SELECT ... FROM earnings`(정정공시 버전이
> 중복 행으로 샌다), (e) `prop_gate` 에 `config=` 누락(다중검정 원장에 시행이 안 남아
> DSR 이 계산되지 않는다), (f) `*_gate.py` 가 `prop_gate` 하버스를 안 쓰는 것, (g) 가격 테이블 직접 SELECT(정문 `read_prices` 우회 — 유니버스에서 폐지 종목이 조용히 빠진다).
>
> (f)가 이 5단계를 **부탁이 아니라 규약으로** 만든다 — 자체 배터리를 새로 짜면
> 음성대조·비용2배·손안댄창·R분포·fragility 중 무엇이 빠졌는지 아무도 모른다.
> 실제로 `pead_gate.py` 가 그 상태로 오래 있었다(2026-08-15 하버스로 이전).
> 유니버스는 PIT·상장폐지 포함이어야 한다 — `kr_quant.validation.survivorship_report`
> 로 생존편향 스멜을 리포트하고, 명백한 생존필터는 `assert_point_in_time`이 잡는다
> (스멜테스트지 보증 아님 — 폐지수익 정합성은 별도).

## 참고 실험 (얇은 러너 예시)

- `experiments/contrarian_bo.py` — `make_objective`로 BO(과최적 방지 목적함수는 라이브러리).
- `experiments/contrarian_validate.py` — `FOLDS`·`sensitivity_table`·`mini_bo`.
- `experiments/contrarian_distribution.py` — `dist_shape`·`selection_curve`·`conviction_analysis`.
- `experiments/contrarian_selective.py` — `fold_consistency`(no-lookahead 선별 재현성).
- `experiments/slippage_check.py` — 비용 스윕 게이트(`fold_slices`).
