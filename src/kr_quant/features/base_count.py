"""Base counting — Minervini's "buy early bases, avoid late ones" filter.

Minervini buys breakouts from the **1st or 2nd base** after a market correction and
avoids 4th/5th-stage bases (too obvious, high failure rate) — see
``SEPA_FAITHFUL_DESIGN.md`` §2.3 (❌ absent in the deployed scanner). This module
counts, as of a given bar, how many bases have formed, and flags whether the
current one is "early" (stage ≤ 2).

A base begins when price pulls back at least ``correction`` from its recent
(``new_high_lookback``-day) high; a fresh high resets the pullback state so the
next dip starts a new base. Detection is **as-of ``asof_idx``** — only closes up to
that bar are used, no look-ahead.

Pure array in → dict out. Feed split-adjusted closes for one code.
"""

from __future__ import annotations

import numpy as np

# Frozen SEPA hyperparameters (SEPA_FAITHFUL_DESIGN.md §사전등록 동결표).
CORRECTION = 0.12          # ≥12% pullback from the recent high starts a base
NEW_HIGH_LOOKBACK = 20     # window (days) defining the "recent high"


def base_count(
    close: np.ndarray,
    asof_idx: int,
    *,
    correction: float = CORRECTION,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
) -> dict:
    """Count bases and the current stage as of ``asof_idx`` (no look-ahead).

    Args:
        close: Per-code split-adjusted close array (abs'd for Kiwoom sign).
        asof_idx: Last bar to use; only ``close[:asof_idx + 1]`` is inspected.
        correction: Pullback depth from the rolling high that starts a new base.
        new_high_lookback: Window (days) over which the "recent high" is measured.

    Returns:
        ``{"base_stage": int, "is_early": bool}`` — ``base_stage`` counts bases
        formed up to ``asof_idx`` (each ≥``correction`` pullback from a rolling
        high, once per pullback); ``is_early`` is ``base_stage ≤ 2`` (Minervini's
        tradeable window).
    """
    c = np.abs(np.asarray(close[:asof_idx + 1], float))
    if c.size == 0 or not np.all(np.isfinite(c)):
        return {"base_stage": 0, "is_early": True}

    stage = 0
    in_correction = False
    for i in range(c.size):
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
