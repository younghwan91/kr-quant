#!/usr/bin/env python
"""BO 결과 일반화 검증 — 파라미터 민감도(US-B3) + walk-forward(US-B4).

과최적을 잡아낸다:
  --sensitivity: best 파라미터를 1개씩 흔들어 TRAIN목적·OOS기대값 표 → 플래토(강건) vs 스파이크(과최적).
  --walkforward: 롤링 fold마다 train BO→다음 test 성과. IS≫OOS면 과최적. 수동값과도 비교.

실행: python research/bo_validate.py --sensitivity | --walkforward
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.bo_optimize import (  # noqa: E402
    TRAIN_HI,
    _load_env_db,
    mini_bo,
)


# walk-forward folds: train 3년 → test 1년 (test 연도 2020~2025)
FOLDS = [
    ("2017-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
    ("2018-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
    ("2019-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
    ("2020-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
    ("2021-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
    ("2022-01-01", "2025-01-01", "2025-01-01", "2027-01-01"),
]
# 플래토 내부·라운드 고정 파라미터 (경계 아님 — "다 가지 말고 멈춘" 값)
ROBUST = dict(window=8, top_mom=0.80, ext_q=0.85, stop=0.10, trail=0.20, hold=60)


def _fs(prices, flow, cache, params, *, lo=None, hi=None,
        adv_floor=20000.0, adv_window=20, lag=1, target=0.0, cost=0.0046):
    from research.contrarian_retail import fast_stats, simulate_fast
    r, e = simulate_fast(prices, flow, _cache=cache, adv_floor=adv_floor, adv_window=adv_window,
                         lag=lag, target=target, cost_roundtrip=cost, **params)
    m = np.ones(len(e), bool)
    if lo is not None:
        m &= e >= lo
    if hi is not None:
        m &= e < hi
    return fast_stats(r[m], e[m])


def _wf_oos(prices, flow, cache, params, **structural):
    """고정 파라미터를 매 fold TEST에서 평가(재최적화 없음) → OOS 기대값 배열."""
    exps = []
    for _tl, _th, sl, sh in FOLDS:
        s = _fs(prices, flow, cache, params, lo=sl, hi=sh, **structural)
        exps.append(s.get("expectancy", np.nan))
    return np.array(exps)


def run_explore(prices, flow, cache):
    """무심코 고정한 구조 상수를 walk-forward OOS로 탐색(최적화 아님, 강건성 서술)."""
    print("=== 고정상수 탐색 — 플래토 라운드 파라미터 고정, walk-forward OOS 강건성 ===")
    print(f"  고정 전략(플래토 내부·라운드): {ROBUST}")
    base = _wf_oos(prices, flow, cache, ROBUST)
    print(f"  기준(adv 200억·adv_window20·lag1·target0·cost46bp왕복): "
          f"OOS평균 {np.nanmean(base):+.4f}  양수 {int(np.nansum(base > 0))}/{len(base)}  "
          f"folds={np.round(base, 3).tolist()}")

    sweeps = [
        ("adv_floor 유니버스체급", "adv_floor",
         [(5000., "50억"), (10000., "100억"), (20000., "200억"), (50000., "500억"), (100000., "1000억")]),
        ("adv_window ADV룩백", "adv_window", [(10, "10일"), (20, "20일"), (40, "40일")]),
        ("lag 신호시차", "lag", [(1, "1일"), (2, "2일"), (3, "3일")]),
        ("target 고정목표(트레일 병행)", "target", [(0.0, "없음"), (0.30, "+30%"), (0.50, "+50%")]),
        ("cost 왕복비용", "cost", [(0.002, "20bp"), (0.0046, "46bp"), (0.008, "80bp")]),
    ]
    for title, key, vals in sweeps:
        print(f"\n  [{title}]")
        for v, label in vals:
            oos = _wf_oos(prices, flow, cache, ROBUST, **{key: v})
            print(f"    {label:>7}: OOS평균 {np.nanmean(oos):+.4f}  양수 {int(np.nansum(oos > 0))}/{len(oos)}")
    print("\n  판독: 기준 대비 급변하는 상수 = 엣지가 그 선택에 의존(강건성 취약). "
          "완만하면 무심코 박아도 무방. 유니버스체급(adv_floor)이 엣지가 사는 곳을 드러낸다.")


def run_sensitivity(prices, flow, cache):
    best = json.load(open("research/bo_best_params.json"))["best"]
    print("=== 민감도 (best 파라미터 1개씩 스윕, target=0 고정) ===")
    print(f"  center(best): {({k: round(v, 3) if isinstance(v, float) else v for k, v in best.items()})}")
    grids = {
        "window": [3, 5, 8, 12, 16, 20],
        "top_mom": [0.70, 0.78, 0.85, 0.90, 0.95],
        "ext_q": [0.70, 0.78, 0.85, 0.90, 0.95],
        "stop": [0.05, 0.08, 0.10, 0.12, 0.15],
        "trail": [0.10, 0.15, 0.20, 0.25, 0.30],
        "hold": [20, 40, 60, 75, 90],
    }
    for pname, vals in grids.items():
        print(f"\n  [{pname}] (다른 파라미터는 best 고정)")
        print(f"    {'val':>6} {'TRAIN_exp':>9} {'TRAIN_po':>8} {'OOS_exp':>8} {'OOS_po':>7} {'OOS_n':>6}")
        for v in vals:
            p = dict(best)
            p[pname] = v
            tr = _fs(prices, flow, cache, p, hi=TRAIN_HI)
            oo = _fs(prices, flow, cache, p, lo=TRAIN_HI)
            mark = " ←best" if abs(v - best[pname]) < (1e-6 if isinstance(v, float) else 0.5) else ""
            print(f"    {v:>6} {tr.get('expectancy', float('nan')):>+9.4f} {tr.get('payoff', float('nan')):>8.2f} "
                  f"{oo.get('expectancy', float('nan')):>+8.4f} {oo.get('payoff', float('nan')):>7.2f} "
                  f"{oo.get('n', 0):>6}{mark}")
    print("\n  판독: best 주변에서 OOS_exp가 완만(플래토)하면 강건. best만 뾰족하고 옆이 급락하면 과최적.")


def run_walkforward(prices, flow, cache, n_trials):
    manual = dict(window=5, top_mom=0.8, ext_q=0.8, stop=0.10, trail=0.20, hold=60)
    folds = FOLDS
    print(f"=== Walk-forward ({len(folds)} folds, train3년→test1년, fold당 BO {n_trials}trials) ===")
    print(f"  {'test기간':>10} {'BO_IS_exp':>9} {'BO_OOS_exp':>10} {'BO_OOS_po':>9} "
          f"{'수동_OOS_exp':>11} {'BO>수동':>7}")
    bo_oos, man_oos, wins = [], [], 0
    for tl, th, sl, sh in folds:
        best, _ = mini_bo(prices, flow, cache, tl, th, n_trials=n_trials, seed=0)
        is_ = _fs(prices, flow, cache, best, lo=tl, hi=th)
        bo = _fs(prices, flow, cache, best, lo=sl, hi=sh)
        man = _fs(prices, flow, cache, manual, lo=sl, hi=sh)
        be, me = bo.get("expectancy", float("nan")), man.get("expectancy", float("nan"))
        win = be > me
        wins += int(win)
        bo_oos.append(be)
        man_oos.append(me)
        yr = sl[:4] if sh[:4] != str(int(sl[:4]) + 1) else sl[:4]
        print(f"  {yr:>10} {is_.get('expectancy', float('nan')):>+9.4f} {be:>+10.4f} "
              f"{bo.get('payoff', float('nan')):>9.2f} {me:>+11.4f} {'✓' if win else '✗':>7}")
    bo_oos, man_oos = np.array(bo_oos), np.array(man_oos)
    print(f"\n  BO OOS 평균 기대값={np.nanmean(bo_oos):+.4f}  수동 OOS 평균={np.nanmean(man_oos):+.4f}")
    print(f"  BO가 수동을 OOS서 이긴 fold: {wins}/{len(folds)}")
    print("  판독: BO_IS≫BO_OOS 반복이면 과최적. BO_OOS가 수동_OOS를 대부분 못이기면 최적화 이득 없음.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--explore", action="store_true", help="무심코 고정한 구조상수 walk-forward 탐색")
    ap.add_argument("--trials", type=int, default=60, help="walk-forward fold당 BO trials")
    args = ap.parse_args()
    _load_env_db()
    from research.contrarian_retail import load_data
    print("=== 데이터 로드 ===")
    prices, flow = load_data()
    cache: dict = {}
    if args.sensitivity:
        run_sensitivity(prices, flow, cache)
    if args.walkforward:
        run_walkforward(prices, flow, cache, args.trials)
    if args.explore:
        run_explore(prices, flow, cache)
    if not (args.sensitivity or args.walkforward or args.explore):
        print("--sensitivity | --walkforward | --explore 지정")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
