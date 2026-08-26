"""Realized-volatility feature — the low-volatility anomaly's raw material.

The low-vol factor ranks stocks by *trailing* return volatility and holds a
dollar-neutral book long the calmest names and short the most volatile
("lottery") names. See :mod:`kr_quant.strategies.lowvol`.

The one correctness property here is **no look-ahead**: the volatility attached
to trading date ``d`` is the standard deviation of daily returns over the
``window`` sessions ending at ``d`` (known at that day's close). The strategy
layer enters at ``t+1``, so the signal is never used before it is observable.

Pure DataFrame in -> DataFrame out (no DB), consistent with the rest of this
package. Ported from scalp-it (``scalp_it.lowvol``, factor-batch note #31);
re-expressed as a kr-quant feature panel so the engine's rank-tilt accounting is
reused instead of a separate decile simulator.
"""

from __future__ import annotations

import pandas as pd

# Frozen default lookback (pre-registered in scalp-it note #31 — do not tune per
# experiment). 60 trading days ~= one quarter of daily returns.
VOL_WINDOW = 60


def realized_vol_panel(
    prices: pd.DataFrame,
    *,
    window: int = VOL_WINDOW,
    close_col: str = "close",
) -> pd.DataFrame:
    """Lookahead-safe (code, date) panel of trailing daily-return volatility.

    For each code, computes daily simple returns from the (abs) close and the
    trailing ``window``-session standard deviation (``ddof=1``), attached to the
    *last* day of each window so no future information leaks.

    Args:
        prices: Long rows with ``code``, ``date`` and ``close_col``. Close may be
            signed (Kiwoom marks the day's direction with the sign); it is abs'd.
        window: Trailing window in trading days (min-periods = ``window``, so a
            code's first ``window`` sessions yield ``NaN``).
        close_col: Close-price column name.

    Returns:
        Long ``code``/``date``/``vol`` DataFrame — ``vol`` is the trailing daily
        return std (``NaN`` during warm-up). One row per input (code, date).
    """
    df = prices[["code", "date", close_col]].copy()
    df["code"] = df["code"].astype(str)
    df["date"] = df["date"].astype(str)
    df[close_col] = df[close_col].abs()
    df = df.sort_values(["code", "date"])
    ret = df.groupby("code")[close_col].pct_change()
    df["vol"] = ret.groupby(df["code"]).transform(
        lambda s: s.rolling(window, min_periods=window).std(ddof=1)
    )
    return df.dropna(subset=["vol"])[["code", "date", "vol"]].reset_index(drop=True)


def lowvol_signal_panel(
    prices: pd.DataFrame,
    *,
    window: int = VOL_WINDOW,
    close_col: str = "close",
) -> pd.DataFrame:
    """Low-vol **signal** panel: ``signal = -vol`` (higher = calmer = long).

    Thin wrapper over :func:`realized_vol_panel` that flips the sign so the
    ranking convention matches the engine's rank-tilt book (top rank = long).
    Feed the result to :func:`kr_quant.strategies.lowvol.lowvol_backtest` (or, as
    a precomputed ``signal_panel``, to any cross-sectional wrapper).

    Returns:
        Long ``code``/``date``/``signal`` DataFrame where ``signal = -vol``.
    """
    vol = realized_vol_panel(prices, window=window, close_col=close_col)
    vol["signal"] = -vol["vol"]
    return vol[["code", "date", "signal"]]
