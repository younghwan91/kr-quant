"""kr_quant.validation — signal-agnostic out-of-sample validation primitives.

Walk-forward folds, one-at-a-time sensitivity sweeps, and robust Bayesian
optimization, extracted from one-off research scripts so every alpha reuses the
same anti-overfit machinery instead of re-deriving it.

A leaf package like ``engine``: it takes GENERIC inputs — arrays of returns +
entry-dates and a ``simulate(params) -> (returns, entry_dates)`` callable — and
imports NOTHING from ``research/``. The signal-specific wiring stays in research.

Two non-negotiables carried over from the research methodology:
    - **No-lookahead:** ``FOLDS`` is a frozen default (``rolling_folds()``); folds
      fit on TRAIN, evaluate on OOS. Not a per-experiment knob → no fold-shopping.
    - **Robust objective:** ``make_objective`` maximizes the bootstrap 2.5% lower
      bound (``_boot_lower``), never the raw mean. This is sacred.

Provenance:
    walkforward  <- bo_validate.{FOLDS, _wf_oos, run_walkforward}
                    + selective_walkforward.{_fold_slices, _rdist, run}
    sensitivity  <- bo_validate.{run_sensitivity, run_explore}
    optimization <- bo_optimize.{make_objective, mini_bo, _boot_lower, split_stats}
"""

from __future__ import annotations

from .optimization import (
    TRADE_FLOOR,
    TRAIN_HI,
    _boot_lower,
    make_objective,
    mini_bo,
    split_stats,
)
from .sensitivity import (
    oos_sensitivity,
    sensitivity_table,
)
from .universe_hygiene import (
    assert_point_in_time,
    survivorship_report,
)
from .walkforward import (
    FOLDS,
    Fold,
    FoldMask,
    entry_mask,
    fold_consistency,
    fold_slices,
    oos_fixed,
    purge_embargo,
    rdist,
    resolve_exit_dates,
    rolling_folds,
    slice_by_entry,
    walk_forward,
)

__all__ = [
    # walk-forward folds + no-lookahead slicing
    "Fold",
    "FOLDS",
    "rolling_folds",
    "entry_mask",
    "slice_by_entry",
    "oos_fixed",
    "walk_forward",
    # purge + embargo (AFML §7.4 — 인접-누출 차단, opt-in)
    "FoldMask",
    "resolve_exit_dates",
    "purge_embargo",
    # fold-consistency (selective walk-forward)
    "rdist",
    "fold_slices",
    "fold_consistency",
    # sensitivity sweeps
    "sensitivity_table",
    "oos_sensitivity",
    # survivorship / PIT hygiene (reporter + strict smell gate)
    "survivorship_report",
    "assert_point_in_time",
    # robust Bayesian optimization
    "TRAIN_HI",
    "TRADE_FLOOR",
    "_boot_lower",
    "split_stats",
    "make_objective",
    "mini_bo",
]
