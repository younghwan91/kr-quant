"""Bayesian optimization with a robust objective — the anti-overfit core.

Signal-agnostic extraction of the BO machinery from ``research/bo_optimize.py``.
Every function here operates on GENERIC inputs — a ``simulate`` callable that maps
a parameter dict to ``(returns, entry_dates)`` arrays, a search ``space``, and an
entry-date TRAIN boundary. No signal wiring, no imports from ``research/``.

The objective is deliberately NOT the raw TRAIN mean. It is the **bootstrap 2.5%
lower bound** of per-trade expectancy (``_boot_lower``): a handful of lucky right-
tail trades inflate the mean, and the lower bound discounts exactly that luck.
This is sacred — never optimize the raw mean. Two more guards travel with it:
    - TRAIN-only fit (entry < ``train_hi``); OOS never touches the objective.
    - a monotone trade-count floor penalty so BO climbs toward the valid region
      instead of parking on a 3-trade fluke.

``optuna`` is an OPTIONAL dependency, imported lazily inside ``mini_bo`` (matching
the research script) so this module imports cleanly without it.

Provenance (copied, generalized to take a ``simulate`` callable + ``space``):
    _boot_lower    <- bo_optimize._boot_lower  (unchanged — sacred)
    split_stats    <- bo_optimize.split_stats  (unchanged)
    make_objective <- bo_optimize.make_objective
    mini_bo        <- bo_optimize.mini_bo
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# 진입일 기준 TRAIN/OOS 경계 (entry < TRAIN_HI = TRAIN | 이상 = OOS). no-lookahead.
TRAIN_HI = "2022-01-01"
TRADE_FLOOR = 150  # TRAIN 최소 거래수 (미달시 페널티)

# 타입: params dict -> (returns, entry_dates). 시그널 배선은 호출자(research)가 넣는다.
Simulate = Callable[[dict], "tuple[np.ndarray, np.ndarray]"]
# 탐색공간: 이름 -> (low, high). 두 경계가 모두 int면 suggest_int, 아니면 suggest_float.
Space = "dict[str, tuple]"


def _boot_lower(rets: np.ndarray, *, n_boot: int = 1000, seed: int = 0) -> float:
    """건당 기대값의 부트스트랩 2.5% 하단(보수적 기대값). 원시 평균 대신 이것을 최대화.

    소수 대박이 만든 스파이크를 하단이 깎아낸다. 표본<5면 최적화가 유효구간으로
    올라가도록 페널티 바닥(-1.0)을 반환(요행 소표본 거부)."""
    if len(rets) < 5:
        return -1.0
    rng = np.random.default_rng(seed)
    means = rng.choice(rets, size=(n_boot, len(rets)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5))


def split_stats(rets, edates, lo=None, hi=None) -> np.ndarray:
    """entry_date로 [lo, hi) 슬라이스 후 유한수익만 반환. lo/hi는 ISO 날짜 문자열."""
    r, e = rets, edates
    if lo is not None:
        m = e >= lo
        r, e = r[m], e[m]
    if hi is not None:
        m = e < hi
        r, e = r[m], e[m]
    return r[np.isfinite(r)]


def _suggest(trial, space: Space) -> dict:
    """optuna trial에서 space대로 파라미터 제안. int 경계 → suggest_int, else float."""
    p = {}
    for name, (lo, hi) in space.items():
        if isinstance(lo, int) and isinstance(hi, int):
            p[name] = trial.suggest_int(name, lo, hi)
        else:
            p[name] = trial.suggest_float(name, lo, hi)
    return p


def make_objective(
    simulate: Simulate,
    space: Space,
    *,
    train_lo=None,
    train_hi=TRAIN_HI,
    floor: int = TRADE_FLOOR,
    n_boot: int = 1000,
    boot_seed: int = 0,
):
    """[train_lo, train_hi) 부트스트랩 기대값 하단을 최대화하는 optuna 목적함수(과최적 저항).

    ``simulate(params) -> (returns, entry_dates)`` 만 알면 시그널 무관하게 동작.
    거래수 < floor면 n에 단조인 페널티(BO가 유효구간으로 climb 유도)."""

    def objective(trial):
        params = _suggest(trial, space)
        rets, edates = simulate(params)
        rt = split_stats(rets, edates, lo=train_lo, hi=train_hi)
        n = len(rt)
        if n < floor:
            return -1.0 + n / floor * 1e-3  # 페널티(단조 → climb 유도)
        trial.set_user_attr("n_train", int(n))
        trial.set_user_attr("exp_train", float(rt.mean()))
        return _boot_lower(rt, n_boot=n_boot, seed=boot_seed)  # 보수적 하단 최대화

    return objective


def mini_bo(
    simulate: Simulate,
    space: Space,
    train_lo,
    train_hi,
    *,
    n_trials: int = 60,
    seed: int = 0,
    floor: int = 80,
    n_boot: int = 1000,
) -> "tuple[dict, float]":
    """지정 TRAIN 기간에서 BO 최적화 → (best_params, best_value). walk-forward fold용.

    optuna는 지연 임포트(선택적 의존성) — 이 모듈은 optuna 없이도 로드된다."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        make_objective(simulate, space, train_lo=train_lo, train_hi=train_hi,
                       floor=floor, n_boot=n_boot),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params, study.best_value
