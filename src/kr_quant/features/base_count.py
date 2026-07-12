"""Base counting — Minervini's "buy early bases, avoid late ones" filter.

Minervini buys breakouts from the **1st or 2nd base** after a *bear market or
correction* and avoids 4th/5th-stage bases (too obvious, high failure rate) — see
``SEPA_FAITHFUL_DESIGN.md`` §2.3 (❌ absent in the deployed scanner). This module
counts, as of a given bar, how many bases have formed **since the last bear-market-
scale decline**, and flags whether the current one is "early" (stage ≤ 2).

⚠️ **Reset bug (found 2026-07-12 via ablation)**: an earlier version counted every
≥12% pullback since the series' inception with no reset, so ``base_stage`` climbed
unbounded (12–36 over an 8-year history) instead of staying in Minervini's intended
1–4 range — silently defeating the "early base" filter (it kept almost nothing).
The fix below resets the counter whenever price has fallen ``deep_reset`` (≥25%,
matching the 52-week-low trend-template threshold used elsewhere) from its trailing
``long_high_lookback``-day high — a single-stock analog of "the bear market/
correction ended, count resets". The bar where price first recovers out of that
deep drawdown is counted as **Base #1** of the new cycle (not deferred until price
reclaims the stale pre-crash rolling high, which would silently understate the
count for up to ``new_high_lookback`` days after every reset); subsequent
≥``correction`` pullbacks from a short rolling high count as Base #2, #3, ... until
the next deep reset.

Detection is **as-of ``asof_idx``** — only closes up to that bar are used, no
look-ahead. Pure array in → dict out. Feed split-adjusted closes for one code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen SEPA hyperparameters (SEPA_FAITHFUL_DESIGN.md §사전등록 동결표).
CORRECTION = 0.12          # ≥12% pullback from the recent high starts a base
NEW_HIGH_LOOKBACK = 20     # window (days) defining the "recent high"
DEEP_RESET = 0.25          # ≥25% decline from the long-window high = bear-market
                           # analog → resets the base-count clock (matches the
                           # 52-week-low trend-template threshold for consistency)
LONG_HIGH_LOOKBACK = 252   # ~52 weeks, for detecting the deep-reset trigger


def base_count(
    close: np.ndarray,
    asof_idx: int,
    *,
    correction: float = CORRECTION,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
    deep_reset: float = DEEP_RESET,
    long_high_lookback: int = LONG_HIGH_LOOKBACK,
) -> dict:
    """Count bases and the current stage as of ``asof_idx`` (no look-ahead).

    Args:
        close: Per-code split-adjusted close array (abs'd for Kiwoom sign).
        asof_idx: Last bar to use; only ``close[:asof_idx + 1]`` is inspected.
        correction: Pullback depth from the rolling high that starts a new base.
        new_high_lookback: Window (days) over which the "recent high" is measured.
        deep_reset: Decline from the ``long_high_lookback``-day high that resets
            the base-count clock (a bear-market/correction analog).
        long_high_lookback: Window (days) for the deep-reset reference high.

    Returns:
        ``{"base_stage": int, "is_early": bool}`` — ``base_stage`` counts bases
        formed since the last ``deep_reset``-scale decline (each ≥``correction``
        pullback from a rolling high, once per pullback); ``is_early`` is
        ``base_stage ≤ 2`` (Minervini's tradeable window).
    """
    c = np.abs(np.asarray(close[:asof_idx + 1], float))
    if c.size == 0 or not np.all(np.isfinite(c)):
        return {"base_stage": 0, "is_early": True}

    stage = 0
    in_correction = False
    was_deep = False   # tracks the deep-decline → recovery transition
    for i in range(c.size):
        lo_long = max(0, i - long_high_lookback + 1)
        long_high = float(np.max(c[lo_long:i + 1]))
        is_deep = long_high > 0 and (long_high - c[i]) / long_high >= deep_reset
        if is_deep:
            # Bear-market-scale decline: clock resets, suppress base counting
            # until price recovers out of the deep drawdown.
            stage = 0
            in_correction = True
            was_deep = True
            continue
        if was_deep:
            # Just emerged from the deep decline: this recovery attempt IS Base #1
            # (don't wait for a stale pre-crash rolling high — that would understate
            # the count for up to new_high_lookback days after every reset).
            stage = 1
            in_correction = True
            was_deep = False
            continue
        lo = max(0, i - new_high_lookback + 1)
        win_high = float(np.max(c[lo:i + 1]))
        if win_high <= 0:
            continue
        if c[i] >= win_high:                 # new/at recent high → reset pullback
            in_correction = False
        else:
            drawdown = (win_high - c[i]) / win_high
            if drawdown >= correction and not in_correction:
                stage += 1                   # a new base begins on this pullback
                in_correction = True
    return {"base_stage": stage, "is_early": stage <= 2}


def base_count_series(
    close: np.ndarray,
    *,
    correction: float = CORRECTION,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
    deep_reset: float = DEEP_RESET,
    long_high_lookback: int = LONG_HIGH_LOOKBACK,
) -> dict:
    """Vectorized :func:`base_count` for **every** bar in one forward pass.

    ``base_count(close, t)`` re-scans ``close[:t+1]`` from scratch on every call —
    called once per (code, date) candidate in :func:`kr_quant.strategies.
    minervini_sepa.sepa_entries`, this was O(bars) per call, i.e. O(bars²) total
    per code (found 2026-07-12 as the sepa_entries hot-path bottleneck). This
    computes the identical sequence of ``base_stage``/``is_early`` values for
    every bar in one O(bars) pass — same state machine, but the two rolling-max
    references (``long_high``, ``win_high``) are precomputed with pandas
    ``rolling`` instead of an ``np.max`` slice re-taken at every step.

    Returns:
        ``{"base_stage": int array, "is_early": bool array}``, one entry per bar
        of ``close`` — ``base_stage[t]`` equals ``base_count(close, t)["base_stage"]``.
    """
    c = np.abs(np.asarray(close, float))
    n = c.size
    stage_arr = np.zeros(n, dtype=int)
    if n == 0:
        return {"base_stage": stage_arr, "is_early": stage_arr <= 2}
    finite = np.isfinite(c)
    # base_count(close, t) checks `np.all(isfinite(close[:t+1]))` up front and
    # returns stage 0 if that fails — so one NaN anywhere "poisons" every asof_idx
    # from that point on (every later prefix still contains it). Replicate exactly
    # via a running any-NaN-seen-so-far flag, not a per-bar skip.
    poisoned = np.cumsum(~finite) > 0

    s = pd.Series(np.where(finite, c, np.nan))
    long_high = s.rolling(long_high_lookback, min_periods=1).max().to_numpy()
    win_high = s.rolling(new_high_lookback, min_periods=1).max().to_numpy()

    stage = 0
    in_correction = False
    was_deep = False
    for i in range(n):
        if poisoned[i]:
            stage_arr[i] = 0
            continue
        lh = long_high[i]
        is_deep = lh > 0 and (lh - c[i]) / lh >= deep_reset
        if is_deep:
            stage = 0
            in_correction = True
            was_deep = True
            stage_arr[i] = stage
            continue
        if was_deep:
            stage = 1
            in_correction = True
            was_deep = False
            stage_arr[i] = stage
            continue
        wh = win_high[i]
        if wh <= 0:
            stage_arr[i] = stage
            continue
        if c[i] >= wh:
            in_correction = False
        else:
            drawdown = (wh - c[i]) / wh
            if drawdown >= correction and not in_correction:
                stage += 1
                in_correction = True
        stage_arr[i] = stage
    return {"base_stage": stage_arr, "is_early": stage_arr <= 2}
