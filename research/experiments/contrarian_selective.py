#!/usr/bin/env python
"""선별형 전략 walk-forward 검증 — "모멘텀 선별이 R-분포를 굽힌다"를 look-ahead 없이 승격.

**얇은 러너**: θ 학습(TRAIN 분위)→TEST 적용의 no-lookahead fold 슬라이스와 굽힘 재현성
집계는 전부 ``kr_quant.validation.walkforward`` 한 곳에서 온다(fold_slices/fold_consistency/
rdist). FOLDS 는 라이브러리 frozen 기본값. 여기선 신호 배선(simulate_detailed)과 print만.

지난 선별 곡선의 함정: "그 기간 상위 X%"는 기간 전체를 본 뒤 랭킹 → 미묘한 look-ahead.
여기선 선별 임계값 θ를 **TRAIN 트레이드에서만** 학습(모멘텀 (1-frac) 분위) → TEST에 그대로 적용.
진입 시점에 이미 아는 θ로 "잡을지 말지"를 결정하므로 배포 가능.

Stochastic 판정(결정론 아님): 한 홀드아웃 극값이 아니라 **여러 롤링 폴드에서 굽힘이 재현되나**.
R-멀티플 = 수익 ÷ 하드손절폭. 복리 없음 — 개별 트레이드가 표본.
실행: uv run python research/experiments/contrarian_selective.py
"""

from __future__ import annotations

import numpy as np

from kr_quant.validation.optimization import TRAIN_HI
from kr_quant.validation.walkforward import FOLDS, fold_consistency
from research.signals.contrarian_retail import _load_env_db, load_data, simulate_detailed

PARAMS = dict(window=8, top_mom=0.80, ext_q=0.85, stop=0.10, trail=0.20, hold=60)
MIN_TRAIN = 40   # θ 학습 최소 트레이드
MIN_SEL = 15     # 선별 후 최소 표본(이하면 폴드 무효 — 희소성 정직 처리)

__all__ = ["PARAMS", "MIN_TRAIN", "MIN_SEL", "TRAIN_HI", "_load_env_db"]


def run(prices, flow):
    stop = PARAMS["stop"]
    cache: dict = {}

    # === Part 1: 대표 강건 설정(상위30% 선별·hold90) 폴드별 상세 ===
    frac0, hold0 = 0.30, 90
    d = simulate_detailed(prices, flow, _cache=cache, **{**PARAMS, "hold": hold0})
    entry, mom, R = d["entry"], d["mom"], d["ret"] / stop
    print(f"=== Part 1: 폴드별 R-분포 — 베이스(전부) vs 선별(모멘텀 상위{frac0:.0%}·θ는 TRAIN학습) "
          f"hold={hold0} ===")
    print(f"  {'TEST연도':>9} {'θ':>6} | {'base_n':>6} {'base_R':>7} {'b≥3R':>5} {'bPO':>5} | "
          f"{'sel_n':>5} {'sel_R':>7} {'s≥3R':>5} {'sPO':>5} | {'굽힘':>4}")
    res = fold_consistency(entry, mom, R, FOLDS, frac0, min_train=MIN_TRAIN, min_sel=MIN_SEL)
    for row in res["rows"]:
        yr = row["fold"].test_lo[:4]
        if row["status"] == "train_short":
            print(f"  {yr:>9}  (TRAIN 부족 — 무효)")
            continue
        if row["status"] == "sparse":
            print(f"  {yr:>9} {row['theta']:>6.3f} | base_n={row['base']['n']:>4} … "
                  f"선별표본<{MIN_SEL} 무효(희소)")
            continue
        b, s = row["base"], row["sel"]
        print(f"  {yr:>9} {row['theta']:>6.3f} | {b['n']:>6} {b['expR']:>+7.3f} {b['tail3']:>5.0%} "
              f"{b['payoff']:>5.2f} | {s['n']:>5} {s['expR']:>+7.3f} {s['tail3']:>5.0%} "
              f"{s['payoff']:>5.2f} | {'▲' if row['bent'] else '▽':>4}")
    valid, bent = res["valid"], res["bent"]
    print(f"  → 굽힘 재현: 유효 {valid}폴드 중 {bent}폴드에서 선별R>베이스R "
          f"({'일관' if valid and bent >= valid - (valid // 3) else '불안정'})")

    # === Part 2: θ·hold 표면 — 굽힘이 플래토인가 스파이크인가(stochastic, 미니마 안 좇음) ===
    print("\n=== Part 2: 선별강도(frac)×hold 표면 — 폴드 전반 굽힘 재현성 ===")
    print("  각 칸: 굽힌폴드/유효폴드 · Δ(선별−베이스 평균기대값R, 폴드평균)")
    holds = (60, 90, 120)
    fracs = (0.50, 0.30, 0.20, 0.10)
    hdr = "frac|hold"
    print(f"  {hdr:>10}" + "".join(f"{h:>16}" for h in holds))
    dcache = {h: simulate_detailed(prices, flow, _cache=cache, **{**PARAMS, "hold": h}) for h in holds}
    for frac in fracs:
        cells = []
        for h in holds:
            dd = dcache[h]
            Rh = dd["ret"] / stop
            r = fold_consistency(dd["entry"], dd["mom"], Rh, FOLDS, frac,
                                 min_train=MIN_TRAIN, min_sel=MIN_SEL)
            # Δ = 유효(ok)폴드 평균의 선별−베이스 기대값R (fold_slices 직접으로도 동일).
            ups = [row["sel"]["expR"] - row["base"]["expR"]
                   for row in r["rows"] if row["status"] == "ok"]
            delta = np.mean(ups) if ups else float("nan")
            cells.append(f"{r['bent']}/{r['valid']} Δ{delta:+.2f}")
        print(f"  {f'상위{frac:.0%}':>10}" + "".join(f"{c:>16}" for c in cells))
    print("  판독: 넓은 영역에서 굽힌폴드≈유효폴드·Δ>0면 플래토(강건). 한 칸만 크면 스파이크(운).")


def main() -> int:
    _load_env_db()
    print("=== 데이터 로드 ===")
    prices, flow = load_data()
    run(prices, flow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
