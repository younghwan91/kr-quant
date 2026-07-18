#!/usr/bin/env python
"""개미 투매 급등주 전략 — 베이지안 최적화(optuna TPE) + 과최적 방지 + OOS 검증.

이 스크립트는 **얇은 러너**다: 과최적 방지 기계(부트스트랩 하단 목적함수·거래수 페널티·
TRAIN 격리)는 전부 ``kr_quant.validation.optimization`` 한 곳에서 온다 — 여기선 재구현하지
않는다(단일 소스). 신호 특정 배선(시뮬레이터·탐색공간·데이터 로드·리포팅)만 남긴다.

과최적을 구조적으로 막는 라이브러리 계약:
  1. TRAIN 기간(<2022)에서만 최적화. OOS는 목적함수에 절대 안 씀(``train_hi``).
  2. 목적함수 = TRAIN 건당 기대값의 **부트스트랩 2.5% 하단**(``_boot_lower``, sacred).
  3. 거래수 < ``floor`` 시 n에 단조인 페널티(BO가 유효구간으로 climb).
가속: numba 코어 + 윈도별 패널 캐시(윈도 바뀔 때만 재빌드).

실행: uv run python research/experiments/contrarian_bo.py [--trials N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from kr_quant.validation.optimization import (
    TRADE_FLOOR,
    TRAIN_HI,
    make_objective,
)
from research.signals.contrarian_retail import (
    _load_env_db,
    fast_stats,
    load_data,
    simulate_fast,
)

# 탐색 공간 (합리적 범위 — 트레이딩 상식 밖은 배제). int 경계 → suggest_int, else float.
# 신호 특정 config: contrarian_validate 의 walk-forward mini_bo 도 이 공간을 공유(단일 소스).
SPACE = {
    "window": (3, 20),        # 개미강도/모멘텀 윈도(일)
    "top_mom": (0.70, 0.95),  # 급등 분위(상위 5~30%)
    "ext_q": (0.70, 0.95),    # 개미투매 극단분위
    "stop": (0.05, 0.15),     # 하드 손절
    "trail": (0.10, 0.30),    # 트레일링 폭
    "hold": (20, 90),         # 시간 상한(일)
}

BEST_PARAMS_PATH = "research/logs/contrarian_retail/bo_best_params.json"


def make_sim(prices, flow, cache, *, target=0.0):
    """params dict -> (수익, 진입일) 콜러블. 라이브러리 목적함수/검증에 넘길 신호 배선.

    simulate_fast 기본 cost_roundtrip=0.0046(왕복 46bp) 유지 — 원 스크립트 목적함수와 동일."""
    def simulate(params: dict):
        return simulate_fast(prices, flow, target=target, _cache=cache, **params)
    return simulate


def report_params(prices, flow, cache, params, cost=0.0046):
    """주어진 파라미터의 TRAIN/OOS 성과 반환(공정 비교용)."""
    r, e = simulate_fast(prices, flow, target=0.0, cost_roundtrip=cost, _cache=cache, **params)
    out = {}
    for tag, lo, hi in [("TRAIN", None, TRAIN_HI), ("OOS", TRAIN_HI, None), ("ALL", None, None)]:
        m = np.ones(len(e), bool)
        if lo is not None:
            m &= e >= lo
        if hi is not None:
            m &= e < hi
        out[tag] = fast_stats(r[m], e[m])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _load_env_db()
    import optuna

    print("=== 데이터 로드 ===")
    prices, flow = load_data()
    cache: dict = {}
    sim = make_sim(prices, flow, cache)

    print(f"=== BO 최적화 (optuna TPE, {args.trials} trials, TRAIN<{TRAIN_HI}, 목적=부트스트랩 하단) ===")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(make_objective(sim, SPACE, train_hi=TRAIN_HI, floor=TRADE_FLOOR),
                   n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    bt = study.best_trial
    print(f"\nbest 목적값(부트스트랩 하단)={study.best_value:+.5f}  "
          f"n_train={bt.user_attrs.get('n_train')}  exp_train={bt.user_attrs.get('exp_train'):+.5f}")
    print("best params:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})

    # 수동 기본값 vs BO최적 — TRAIN/OOS 공정 비교 (과최적/일반화 1차 점검)
    manual = dict(window=5, top_mom=0.8, ext_q=0.8, stop=0.10, trail=0.20, hold=60)
    print("\n=== 수동기본 vs BO최적 — TRAIN/OOS 성과 (건당 기대값 | 손익비 | n) ===")
    for name, params in [("수동기본", manual), ("BO최적", best)]:
        rep = report_params(prices, flow, cache, params)
        s = " | ".join(f"{tag} exp={rep[tag].get('expectancy', float('nan')):+.4f} "
                       f"po={rep[tag].get('payoff', float('nan')):.2f} n={rep[tag].get('n', 0)}"
                       for tag in ("TRAIN", "OOS"))
        print(f"  {name:8}: {s}")
    print("\n  판독: BO최적의 OOS가 수동기본 OOS보다 확실히 낫고 TRAIN≈OOS면 일반화 성공.")
    print("        BO의 TRAIN≫OOS(급락)면 과최적. 비슷하면 수동값으로 충분(최적화 이득 없음).")

    # best params 저장(다운스트림 민감도/walk-forward용)
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    json.dump({"best": best, "manual": manual, "objective": study.best_value,
               "n_train": bt.user_attrs.get("n_train")}, open(BEST_PARAMS_PATH, "w"), indent=2)
    print(f"\n[saved] {BEST_PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
