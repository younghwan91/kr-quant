"""Parameter sensitivity sweeps — plateau (robust) vs spike (overfit).

Signal-agnostic extraction from ``research/bo_validate.py`` (run_sensitivity,
run_explore; 원본은 추출과 함께 삭제돼 git 이력에만 있다). One-at-a-time perturbation: hold a center parameter set fixed, walk
one parameter across a grid, and read the TRAIN vs OOS metric shape.

    - A **plateau** around the center (neighbors stay near the center's value) =
      the edge does not hinge on that exact choice → robust.
    - A **spike** (center is high, neighbors collapse) = the edge is a knife-edge
      overfit to that one value.

Two entry points, both take a generic ``simulate(params) -> (returns, entry_dates)``:
    - ``sensitivity_table`` — split-based (TRAIN vs OOS at one entry-date boundary),
      from run_sensitivity.
    - ``oos_sensitivity`` — walk-forward OOS (one metric per perturbation, averaged
      across folds), from run_explore. Reuses ``walkforward.oos_fixed``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .optimization import TRAIN_HI, split_stats
from .walkforward import FOLDS, oos_fixed

Simulate = Callable[[dict], "tuple[np.ndarray, np.ndarray]"]


def _default_stat(rets: np.ndarray) -> dict:
    """건당 기대값·표본수·손익비 최소 지표. sensitivity 표의 기본 stat."""
    r = rets[np.isfinite(rets)] if len(rets) else rets
    if len(r) < 1:
        return {"expectancy": float("nan"), "n": 0, "payoff": float("nan")}
    wins, losses = r[r > 0], r[r < 0]
    payoff = (wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan")
    return {"expectancy": float(r.mean()), "n": int(len(r)), "payoff": float(payoff)}


def _is_center(v, cv) -> bool:
    """v가 center값 cv와 같은가(float은 근사, int은 정확)."""
    if isinstance(cv, float) or isinstance(v, float):
        return abs(v - cv) < 1e-6
    return v == cv


def sensitivity_table(
    simulate: Simulate,
    center: dict,
    grids: "dict[str, list]",
    *,
    split: str = TRAIN_HI,
    stat: Callable[[np.ndarray], dict] = _default_stat,
) -> list:
    """center를 1개씩 흔들며 TRAIN/OOS stat 표. plateau vs spike 판독용(과최적 탐지).

    ``research.bo_validate.run_sensitivity`` 일반화. 다른 파라미터는 center 고정,
    한 파라미터만 grid로 스윕. ``split`` 진입일 경계로 TRAIN(<split)/OOS(>=split) 분리.

    반환: 행 리스트 ``{"param", "value", "is_center", "train", "oos"}`` — 길이는
    ``sum(len(v) for v in grids.values())``. train/oos는 stat이 낸 dict."""
    rows = []
    for pname, vals in grids.items():
        cv = center[pname]
        for v in vals:
            p = dict(center)
            p[pname] = v
            rets, edates = simulate(p)
            tr = split_stats(rets, edates, hi=split)
            oo = split_stats(rets, edates, lo=split)
            rows.append({
                "param": pname,
                "value": v,
                "is_center": _is_center(v, cv),
                "train": stat(tr),
                "oos": stat(oo),
            })
    return rows


def oos_sensitivity(
    simulate: Simulate,
    center: dict,
    grids: "dict[str, list]",
    *,
    folds=FOLDS,
    stat: Callable[[np.ndarray], float] | None = None,
) -> list:
    """center를 1개씩 흔들며 walk-forward OOS 지표(fold 평균·양수 개수) 표.

    ``research.bo_validate.run_explore`` 일반화 — 무심코 고정한 상수(체급·시차·비용 등)나
    전략 파라미터를 흔들어 OOS 강건성을 서술(최적화 아님). fold별 OOS 지표는
    ``walkforward.oos_fixed``로 계산하고 fold 평균·양수개수로 요약.

    반환: 행 리스트 ``{"param", "value", "is_center", "oos_mean", "n_pos", "n_folds", "folds"}``."""
    kw = {} if stat is None else {"stat": stat}
    rows = []
    for pname, vals in grids.items():
        cv = center[pname]
        for v in vals:
            p = dict(center)
            p[pname] = v
            arr = oos_fixed(folds, simulate, p, **kw)
            rows.append({
                "param": pname,
                "value": v,
                "is_center": _is_center(v, cv),
                "oos_mean": float(np.nanmean(arr)) if len(arr) else float("nan"),
                "n_pos": int(np.nansum(arr > 0)),
                "n_folds": len(arr),
                "folds": arr,
            })
    return rows
