#!/usr/bin/env python
"""분포 관점 분석 — 복리 자본곡선이 아니라 **개별 트레이드 = 표본**의 R-멀티플 분포.

**얇은 러너**: 분포 모양·선별 곡선·확신 분위·보유상한 스윕의 계산은 전부
``kr_quant.diagnostics.r_distribution`` 한 곳에서 온다(단일 소스, dict 반환). 여기선
신호 배선(simulate_detailed)과 print 표만 한다.

성공한 트레이더와 평균-백테스트의 차이:
  나(백테스트)  : 신호 전부 → 평균 1개 숫자로 붕괴. 엣지 = "모든 신호의 평균".
  성공한 트레이더: 개별 트레이드를 R-멀티플(수익÷진입시 리스크) 표본으로 쌓아 **분포의 모양**을 본다.
                 손절이 왼꼬리를 −1R에 자르고, 선별·승자태우기가 오른꼬리를 살찌우는가.

이건 stochastic이다(deterministic 아님): 한 백테스트 경로는 분포에서 뽑은 한 표본. 판정은
'분포의 모양이 TRAIN·OOS 둘 다에서 같은 방향으로 굽나'로 — 한 경로의 극값이 아니라.
R-멀티플 = 트레이드 수익 ÷ 하드손절폭. 복리·사이징 없음 — 개별 트레이드가 표본.

실행: uv run python research/experiments/contrarian_distribution.py
"""

from __future__ import annotations

import numpy as np

from kr_quant.diagnostics.r_distribution import (
    conviction_analysis,
    dist_shape,
    hold_curve,
    r_multiples,
    selection_curve,
)
from kr_quant.validation.optimization import TRAIN_HI
from research.signals.contrarian_retail import _load_env_db, load_data, simulate_detailed

# 5차/탐색의 플래토 라운드 고정값(경계 아님)
PARAMS = dict(window=8, top_mom=0.80, ext_q=0.85, stop=0.10, trail=0.20, hold=60)


def _print_shape(label: str, sh: dict) -> None:
    """dist_shape dict → 원 스크립트 표. 왼꼬리 절단·오른꼬리 두께 서술."""
    if sh["n"] == 0:
        print(f"  [{label}] 표본 없음")
        return
    print(f"  [{label}] n={sh['n']}  기대값={sh['expectancy_R']:+.3f}R  승률={sh['win_rate']:.0%}  "
          f"손익비={sh['payoff']:.2f}  왜도={sh['skew']:+.2f}")
    print(f"    왼꼬리(≤−0.9R): {sh['left_tail_share']:.0%}  평균패={sh['avg_loss_R']:+.2f}R "
          f"(리스크가 −1R에 절단되면 건강)")
    for thr in (3.0, 5.0, 10.0):
        rt = sh["right_tail"][thr]
        print(f"    오른꼬리(≥+{thr:.0f}R): 빈도={rt['freq']:5.1%}  양수익의 {rt['share']:4.0%} 차지")


def _print_conviction(label: str, res: dict) -> None:
    """conviction_analysis dict → 5분위 R-기대값 표(TRAIN/OOS) + 단조성 판정."""
    print(f"\n  [확신신호: {label}] 5분위 R-기대값 (Q5=최고확신)  — 진입시점 알 수 있는 값")
    print(f"    {'분위':>4} {'TRAIN_R':>8} {'TRAIN_n':>7} {'OOS_R':>8} {'OOS_n':>6} {'OOS≥3R':>7}")
    nan = float("nan")
    for q in range(res["nq"]):
        tr = res["train"][q] if res["train"] else {"expectancy_R": nan, "n": 0}
        oo = res["oos"][q] if res["oos"] else {"expectancy_R": nan, "n": 0, "tail_freq": nan}
        print(f"    Q{q + 1:>2} {tr['expectancy_R']:>+8.3f} {tr['n']:>7} "
              f"{oo['expectancy_R']:>+8.3f} {oo['n']:>6} {oo.get('tail_freq', nan):>7.1%}")
    if res["verdict"] is not None:
        verdict = ("예측력 있음(선별 값어치)" if res["verdict"] == "predictive"
                   else "예측력 없음/불안정 → 꼬리는 대체로 운")
        print(f"    → Q5>Q1: TRAIN={res['train_monotonic']} OOS={res['oos_monotonic']} → {verdict}")


def _print_selection(name: str, rows: list) -> None:
    """selection_curve rows → 선별 곡선 표. 상위%로 갈수록 분포가 굽나."""
    print(f"\n  [선별 곡선: {name}] 모멘텀 확신 상위 X%만 잡을 때 R-분포 이동 (진입시점 선별)")
    print(f"    {'선별':>10} {'n':>5} {'기대값R':>8} {'승률':>5} {'≥3R빈도':>7} {'≥3R점유':>7} {'손익비':>6}")
    for row in rows:
        label = "전부(나)" if row["frac"] == 1.0 else f"상위{row['frac']:.0%}"
        print(f"    {label:>10} {row['n']:>5} {row['expectancy_R']:>+8.3f} {row['win_rate']:>5.0%} "
              f"{row['tail_freq']:>7.1%} {row['tail_share']:>7.0%} {row['payoff']:>6.2f}")
    print("    판독: 상위%로 갈수록 기대값R·오른꼬리 빈도↑면 선별이 분포를 굽힘(트레이더 엣지).")


def main() -> int:
    _load_env_db()
    stop = PARAMS["stop"]
    print("=== 데이터 로드 ===")
    prices, flow = load_data()
    d = simulate_detailed(prices, flow, **PARAMS)
    entry = d["entry"]
    train_m = entry < TRAIN_HI

    print("\n=== 개별표본 R-분포의 모양 (복리 아님, 트레이드=표본) ===")
    _print_shape("TRAIN 전체", dist_shape(r_multiples(d["ret"][train_m], stop)))
    _print_shape("OOS 전체", dist_shape(r_multiples(d["ret"][~train_m], stop)))
    # 청산사유별(OOS) — 오른꼬리는 어디서(트레일/시간청산이 승자를 태우나)
    names = {0: "시간", 1: "손절", 2: "트레일", 3: "목표"}
    print("  청산사유별(OOS):")
    om = ~train_m
    for rc in (0, 1, 2, 3):
        mm = om & (d["reason"] == rc)
        if mm.sum():
            R = d["ret"][mm] / stop
            print(f"    {names[rc]:>5}: n={mm.sum():>5} 기대값={R.mean():+.2f}R 최대={R.max():+.1f}R")

    print("\n=== 꼬리 예측력 (a-priori 확신 신호, TRAIN vs OOS) ===")
    R_all = d["ret"] / stop
    momz = (d["mom"] - np.nanmean(d["mom"])) / np.nanstd(d["mom"])
    dumpz = (d["dump"] - np.nanmean(d["dump"])) / np.nanstd(d["dump"])
    combo = momz + dumpz
    _print_conviction("모멘텀 강도", conviction_analysis(d["mom"], R_all, entry, train_hi=TRAIN_HI))
    _print_conviction("개미 투매 극단도", conviction_analysis(d["dump"], R_all, entry, train_hi=TRAIN_HI))
    _print_conviction("결합(모멘텀+투매 z합)", conviction_analysis(combo, R_all, entry, train_hi=TRAIN_HI))

    # 선별 곡선 — 나(전부) vs 트레이더(선별). TRAIN·OOS 둘 다로 강건성(stochastic 판정)
    _print_selection("TRAIN", selection_curve(R_all[train_m], d["mom"][train_m]))
    _print_selection("OOS", selection_curve(R_all[~train_m], d["mom"][~train_m]))

    # 보유상한 스윕 (OOS) — 더 오래 타면 오른꼬리가 두꺼워지나(승자 태우기)
    print("\n=== 보유상한 스윕 (OOS, 오른꼬리 두께 — 승자 태우기 확인) ===")
    results = {}
    for h in (40, 60, 90, 120):
        dh = simulate_detailed(prices, flow, **{**PARAMS, "hold": h})
        m = dh["entry"] >= TRAIN_HI
        results[h] = {"R": dh["ret"][m] / stop, "reason": dh["reason"][m]}
    print(f"  {'hold':>4} {'기대값R':>8} {'≥3R빈도':>7} {'≥5R빈도':>7} {'시간청산%':>8} {'양수익상위10%점유':>14}")
    for row in hold_curve(results):
        print(f"  {row['hold']:>4} {row['expectancy_R']:>+8.3f} {row['freq_ge_3R']:>7.1%} "
              f"{row['freq_ge_5R']:>7.1%} {row['time_exit_share']:>8.0%} {row['top_share']:>14.0%}")
    print("  판독: hold↑에 기대값R·오른꼬리 빈도↑면 '더 오래 타기'가 오른꼬리를 확장.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
