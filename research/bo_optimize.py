#!/usr/bin/env python
"""개미 투매 급등주 전략 — 베이지안 최적화(optuna TPE) + 과최적 방지 + OOS 검증.

과최적을 구조적으로 막는다:
  1. TRAIN 기간(2017~2021)에서만 최적화. OOS(2022~2026)는 목적함수에 절대 안 씀.
  2. 목적함수 = TRAIN 건당 기대값의 **부트스트랩 95% 하단**(요행/소표본에 저항).
     원시 기대값 단독 금지 — 소수 대박이 만든 스파이크를 하단이 깎아낸다.
  3. 거래수 하한 미달 시 페널티(n에 단조 → BO가 유효구간으로 climb).
가속: numba 코어 + 윈도별 패널 캐시(윈도 바뀔 때만 재빌드).

실행: python research/bo_optimize.py [--trials N] [--seed S]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 스크립트 직접 실행 시 repo root를 path에 추가(from research.* import 가능하게)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_HI = "2022-01-01"   # TRAIN: entry < 2022 | OOS: entry >= 2022
TRADE_FLOOR = 150         # TRAIN 최소 거래수 (미달시 페널티)

# 탐색 공간 (합리적 범위 — 트레이딩 상식 밖은 배제)
SPACE = {
    "window": (3, 20),      # 개미강도/모멘텀 윈도(일)
    "top_mom": (0.70, 0.95),  # 급등 분위(상위 5~30%)
    "ext_q": (0.70, 0.95),    # 개미투매 극단분위
    "stop": (0.05, 0.15),     # 하드 손절
    "trail": (0.10, 0.30),    # 트레일링 폭
    "hold": (20, 90),         # 시간 상한(일)
}


def _load_env_db():
    if not os.environ.get("KR_QUANT_DB") and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("KR_QUANT_DB"):
                os.environ["KR_QUANT_DB"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def _boot_lower(rets: np.ndarray, *, n_boot=1000, seed=0) -> float:
    """건당 기대값의 부트스트랩 2.5% 하단(보수적 기대값)."""
    if len(rets) < 5:
        return -1.0
    rng = np.random.default_rng(seed)
    means = rng.choice(rets, size=(n_boot, len(rets)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5))


def split_stats(rets, edates, lo=None, hi=None):
    """entry_date로 기간 슬라이스 후 유한수익 반환."""
    r, e = rets, edates
    if lo is not None:
        m = e >= lo
        r, e = r[m], e[m]
    if hi is not None:
        m = e < hi
        r, e = r[m], e[m]
    return r[np.isfinite(r)]


def make_objective(prices, flow, cache, train_lo=None, train_hi=TRAIN_HI, floor=TRADE_FLOOR):
    """train_lo~train_hi 기간의 부트스트랩 기대값 하단을 최대화하는 목적함수(과최적 저항)."""
    from research.contrarian_retail import simulate_fast

    def objective(trial):
        p = {
            "window": trial.suggest_int("window", *SPACE["window"]),
            "top_mom": trial.suggest_float("top_mom", *SPACE["top_mom"]),
            "ext_q": trial.suggest_float("ext_q", *SPACE["ext_q"]),
            "stop": trial.suggest_float("stop", *SPACE["stop"]),
            "trail": trial.suggest_float("trail", *SPACE["trail"]),
            "hold": trial.suggest_int("hold", *SPACE["hold"]),
        }
        r, e = simulate_fast(prices, flow, target=0.0, _cache=cache, **p)
        rt = split_stats(r, e, lo=train_lo, hi=train_hi)
        n = len(rt)
        if n < floor:
            return -1.0 + n / floor * 1e-3   # 페널티(단조 → climb 유도)
        trial.set_user_attr("n_train", int(n))
        trial.set_user_attr("exp_train", float(rt.mean()))
        return _boot_lower(rt)               # 보수적 기대값 하단 최대화

    return objective


def mini_bo(prices, flow, cache, train_lo, train_hi, *, n_trials=60, seed=0, floor=80):
    """지정 기간에서 BO 최적화, best params 반환(walk-forward fold용)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(make_objective(prices, flow, cache, train_lo, train_hi, floor),
                   n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def report_params(prices, flow, cache, params, cost=0.0046):
    """주어진 파라미터의 TRAIN/OOS 성과 반환(공정 비교용)."""
    from research.contrarian_retail import fast_stats, simulate_fast
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
    from research.contrarian_retail import load_data

    print("=== 데이터 로드 ===")
    prices, flow = load_data()
    cache: dict = {}

    print(f"=== BO 최적화 (optuna TPE, {args.trials} trials, TRAIN<{TRAIN_HI}, 목적=부트스트랩 하단) ===")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(make_objective(prices, flow, cache), n_trials=args.trials, show_progress_bar=False)

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
    import json
    outp = "research/bo_best_params.json"
    json.dump({"best": best, "manual": manual, "objective": study.best_value,
               "n_train": bt.user_attrs.get("n_train")}, open(outp, "w"), indent=2)
    print(f"\n[saved] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
