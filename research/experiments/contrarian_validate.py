#!/usr/bin/env python
"""BO 결과 일반화 검증 — 파라미터 민감도(US-B3) + walk-forward(US-B4).

**얇은 러너**: 롤링 fold 정의·no-lookahead OOS 슬라이스·민감도 스윕 기계는 전부
``kr_quant.validation`` 한 곳에서 온다(단일 소스). 여기선 신호 배선(시뮬레이터·탐색공간·
리포팅)만 한다. FOLDS 는 라이브러리의 frozen 기본값 — 실험마다 새로 만들지 않는다(fold-shopping 방지).

과최적을 잡아낸다:
  --sensitivity: best 파라미터를 1개씩 흔들어 TRAIN목적·OOS기대값 표 → 플래토(강건) vs 스파이크(과최적).
  --walkforward: 롤링 fold마다 train BO→다음 test 성과. IS≫OOS면 과최적. 수동값과도 비교.
  --explore: 무심코 고정한 구조상수를 walk-forward OOS로 흔들어 강건성 서술(최적화 아님).

실행: uv run python research/experiments/contrarian_validate.py --sensitivity | --walkforward | --explore
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from kr_quant.validation.optimization import TRAIN_HI, mini_bo
from kr_quant.validation.sensitivity import oos_sensitivity, sensitivity_table
from kr_quant.validation.walkforward import FOLDS
from research.experiments.contrarian_bo import SPACE, make_sim
from research.signals.contrarian_retail import (
    _load_env_db,
    fast_stats,
    load_data,
    simulate_fast,
)

BEST_PARAMS_PATH = "research/logs/contrarian_retail/bo_best_params.json"
# 플래토 내부·라운드 고정 파라미터 (경계 아님 — "다 가지 말고 멈춘" 값)
ROBUST = dict(window=8, top_mom=0.80, ext_q=0.85, stop=0.10, trail=0.20, hold=60)
# explore 가 흔들 구조상수의 기준값(simulate_fast 기본과 일치).
STRUCTURAL = dict(adv_floor=20000.0, adv_window=20, lag=1, target=0.0, cost_roundtrip=0.0046)


def _fs(prices, flow, cache, params, *, lo=None, hi=None, cost=0.0046, **structural):
    """신호 배선 어댑터: 파라미터 1세트를 simulate_fast → 진입일 [lo,hi) 슬라이스 → fast_stats."""
    r, e = simulate_fast(prices, flow, _cache=cache, cost_roundtrip=cost, **structural, **params)
    return fast_stats(r, e, date_lo=lo, date_hi=hi)


def run_sensitivity(prices, flow, cache):
    best = json.load(open(BEST_PARAMS_PATH))["best"]
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
    # 라이브러리 sensitivity_table: center 1개씩 흔들며 TRAIN(<split)/OOS(>=split) stat 표(과최적 탐지).
    rows = sensitivity_table(make_sim(prices, flow, cache), best, grids, split=TRAIN_HI)
    last_param = None
    for row in rows:
        if row["param"] != last_param:
            last_param = row["param"]
            print(f"\n  [{row['param']}] (다른 파라미터는 best 고정)")
            print(f"    {'val':>6} {'TRAIN_exp':>9} {'TRAIN_po':>8} {'OOS_exp':>8} {'OOS_po':>7} {'OOS_n':>6}")
        tr, oo, v = row["train"], row["oos"], row["value"]
        mark = " ←best" if row["is_center"] else ""
        print(f"    {v:>6} {tr.get('expectancy', float('nan')):>+9.4f} {tr.get('payoff', float('nan')):>8.2f} "
              f"{oo.get('expectancy', float('nan')):>+8.4f} {oo.get('payoff', float('nan')):>7.2f} "
              f"{oo.get('n', 0):>6}{mark}")
    print("\n  판독: best 주변에서 OOS_exp가 완만(플래토)하면 강건. best만 뾰족하고 옆이 급락하면 과최적.")


def run_explore(prices, flow, cache):
    """무심코 고정한 구조 상수를 walk-forward OOS로 탐색(최적화 아님, 강건성 서술)."""
    print("=== 고정상수 탐색 — 플래토 라운드 파라미터 고정, walk-forward OOS 강건성 ===")
    print(f"  고정 전략(플래토 내부·라운드): {ROBUST}")
    center = {**ROBUST, **STRUCTURAL}
    # 여기 sim 은 구조상수(target·cost 포함)를 전부 params 로 흘려보낸다 — make_sim 처럼
    # target 을 고정하면 center 의 target 과 충돌하므로 별도 클로저를 쓴다.
    def sim(params: dict):
        return simulate_fast(prices, flow, _cache=cache, **params)
    base = oos_sensitivity(sim, center, {"adv_floor": [STRUCTURAL["adv_floor"]]})[0]
    print(f"  기준(adv 200억·adv_window20·lag1·target0·cost46bp왕복): "
          f"OOS평균 {base['oos_mean']:+.4f}  양수 {base['n_pos']}/{base['n_folds']}  "
          f"folds={np.round(base['folds'], 3).tolist()}")

    # 각 구조상수 그리드 + 표시 라벨(억/일/bp). key 는 simulate_fast 인자명.
    sweeps = [
        ("adv_floor 유니버스체급", "adv_floor",
         [(5000., "50억"), (10000., "100억"), (20000., "200억"), (50000., "500억"), (100000., "1000억")]),
        ("adv_window ADV룩백", "adv_window", [(10, "10일"), (20, "20일"), (40, "40일")]),
        ("lag 신호시차", "lag", [(1, "1일"), (2, "2일"), (3, "3일")]),
        ("target 고정목표(트레일 병행)", "target", [(0.0, "없음"), (0.30, "+30%"), (0.50, "+50%")]),
        ("cost 왕복비용", "cost_roundtrip", [(0.002, "20bp"), (0.0046, "46bp"), (0.008, "80bp")]),
    ]
    for title, key, vals in sweeps:
        print(f"\n  [{title}]")
        rows = oos_sensitivity(sim, center, {key: [v for v, _ in vals]})
        labels = {v: lab for v, lab in vals}
        for row in rows:
            lab = labels.get(row["value"], str(row["value"]))
            print(f"    {lab:>7}: OOS평균 {row['oos_mean']:+.4f}  양수 {row['n_pos']}/{row['n_folds']}")
    print("\n  판독: 기준 대비 급변하는 상수 = 엣지가 그 선택에 의존(강건성 취약). "
          "완만하면 무심코 박아도 무방. 유니버스체급(adv_floor)이 엣지가 사는 곳을 드러낸다.")


def run_walkforward(prices, flow, cache, n_trials):
    manual = dict(window=5, top_mom=0.8, ext_q=0.8, stop=0.10, trail=0.20, hold=60)
    sim = make_sim(prices, flow, cache)  # BO 목적함수용 시뮬(= optimization.make_objective 배선)
    print(f"=== Walk-forward ({len(FOLDS)} folds, train3년→test1년, fold당 BO {n_trials}trials) ===")
    print(f"  {'test기간':>10} {'BO_IS_exp':>9} {'BO_OOS_exp':>10} {'BO_OOS_po':>9} "
          f"{'수동_OOS_exp':>11} {'BO>수동':>7}")
    bo_oos, man_oos, wins = [], [], 0
    for f in FOLDS:
        # fold TRAIN 에서만 BO(no-lookahead) → 고정 best 를 TEST 에서 평가(재최적화 없음).
        best, _ = mini_bo(sim, SPACE, f.train_lo, f.train_hi, n_trials=n_trials, seed=0)
        is_ = _fs(prices, flow, cache, best, lo=f.train_lo, hi=f.train_hi)
        bo = _fs(prices, flow, cache, best, lo=f.test_lo, hi=f.test_hi)
        man = _fs(prices, flow, cache, manual, lo=f.test_lo, hi=f.test_hi)
        be, me = bo.get("expectancy", float("nan")), man.get("expectancy", float("nan"))
        win = be > me
        wins += int(win)
        bo_oos.append(be)
        man_oos.append(me)
        print(f"  {f.test_lo[:4]:>10} {is_.get('expectancy', float('nan')):>+9.4f} {be:>+10.4f} "
              f"{bo.get('payoff', float('nan')):>9.2f} {me:>+11.4f} {'✓' if win else '✗':>7}")
    bo_oos, man_oos = np.array(bo_oos), np.array(man_oos)
    print(f"\n  BO OOS 평균 기대값={np.nanmean(bo_oos):+.4f}  수동 OOS 평균={np.nanmean(man_oos):+.4f}")
    print(f"  BO가 수동을 OOS서 이긴 fold: {wins}/{len(FOLDS)}")
    print("  판독: BO_IS≫BO_OOS 반복이면 과최적. BO_OOS가 수동_OOS를 대부분 못이기면 최적화 이득 없음.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--explore", action="store_true", help="무심코 고정한 구조상수 walk-forward 탐색")
    ap.add_argument("--trials", type=int, default=60, help="walk-forward fold당 BO trials")
    args = ap.parse_args()
    _load_env_db()
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
