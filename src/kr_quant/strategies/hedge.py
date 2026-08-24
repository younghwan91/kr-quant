"""Inverse-ETF market hedge — the deployable substitute for an individual short leg.

**Why this exists.** The validated PEAD (and low-vol) edge is a dollar-neutral
long/short book, and robustness testing showed most of PEAD's edge lives in the
*short* leg (raise the liquidity floor and the long-only excess dies:
t≈0.48 at 50억). But a retail Korean account often **cannot short individual
names** — it can only buy an inverse index ETF (KODEX 인버스 = −1× KOSPI200,
252670 = −2×). This module builds the implementable form: hold the long book of
individual stocks and neutralize *market beta* by buying an inverse ETF, instead
of shorting individual stocks.

The trade-off is explicit and must be reported: an inverse ETF removes only the
**market** component of risk/return, so any *stock-specific* short alpha (the
lottery-stock short in low-vol, the negative-surprise short in PEAD) is left on
the table. This is why the hedged form is the honest *deployable* spec while the
individual-short L/S is the theoretical *ceiling*.

⚠️ **Synthetic inverse.** kr-quant's ``daily_bars_adjusted`` holds individual
stocks only — no ETF bars over 2016–2026. So the inverse leg is *synthesized*
from the market proxy: ``r_inverse = −leverage × r_market − fee``. A real KODEX
인버스 also carries futures-roll and tracking drag beyond its ~0.64%/yr TER, so
the synthetic hedge is a mild *upper* bound on the hedged book. See
``docs/combo-book.md``.

Pure Series in -> Series out (no DB), unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# KODEX 인버스 total expense ratio ~0.64%/yr; used as the synthetic inverse's
# holding drag when a per-period fee is derived from an annual figure.
INVERSE_ANNUAL_FEE = 0.0064


def universe_market_return(
    prices: pd.DataFrame,
    *,
    adv_floor: float = 0.0,
    adv_window: int = 20,
    close_col: str = "close",
) -> pd.Series:
    """Equal-weight daily return of the liquid universe — the market proxy.

    With no ETF bars available, the "market" the long book is hedged against (and
    whose −1× the inverse ETF tracks) is proxied by the cross-sectional mean daily
    return across names clearing a trailing-ADV floor at each date.

    Args:
        prices: Long ``code``/``date``/``close`` (+ ``trade_value`` if ``adv_floor>0``).
        adv_floor: Trailing-ADV floor (백만원); ``0`` = whole cross-section.
        adv_window: Trailing window for the ADV filter.
        close_col: Close column (abs'd — close may be signed).

    Returns:
        Date-indexed Series of the equal-weight market daily return.
    """
    df = prices[["code", "date", close_col]].copy()
    df["code"] = df["code"].astype(str)
    df["date"] = df["date"].astype(str)
    df[close_col] = df[close_col].abs()
    df = df.sort_values(["code", "date"])
    df["ret"] = df.groupby("code")[close_col].pct_change()

    if adv_floor > 0 and "trade_value" in prices.columns:
        tv = prices[["code", "date", "trade_value"]].copy()
        tv["code"] = tv["code"].astype(str)
        tv["date"] = tv["date"].astype(str)
        tv["trade_value"] = tv["trade_value"].abs()
        tv = tv.sort_values(["code", "date"])
        tv["adv"] = tv.groupby("code")["trade_value"].transform(
            lambda s: s.rolling(adv_window, min_periods=adv_window).mean())
        df = df.merge(tv[["code", "date", "adv"]], on=["code", "date"], how="left")
        df = df[df["adv"] >= adv_floor]

    return df.dropna(subset=["ret"]).groupby("date")["ret"].mean().sort_index()


def synth_inverse_return(
    market_ret: pd.Series,
    *,
    leverage: float = 1.0,
    fee_per_period: float = 0.0,
) -> pd.Series:
    """Synthetic inverse-ETF return: ``-leverage * market - fee_per_period``.

    Args:
        market_ret: Market return Series (any frequency).
        leverage: 1.0 for KODEX 인버스 (−1×), 2.0 for 252670 (−2×).
        fee_per_period: Holding drag charged per period (e.g. annual TER / periods
            per year), on top of the −leverage tracking.

    Returns:
        Series aligned to ``market_ret``.
    """
    return -leverage * market_ret.astype(float) - fee_per_period


def rolling_beta(
    long_ret: pd.Series,
    market_ret: pd.Series,
    *,
    window: int | None = None,
    min_obs: int = 24,
) -> pd.Series:
    """Lookahead-safe beta of ``long_ret`` on ``market_ret`` (cov/var), lagged one period.

    The beta used to size the hedge at period ``t`` is estimated from data
    strictly **before** ``t`` (expanding by default, or a trailing ``window``),
    then shifted one period so it is knowable at entry.

    Args:
        long_ret: Long-book return Series.
        market_ret: Market return Series (aligned on the same index).
        window: Trailing window; ``None`` = expanding.
        min_obs: Minimum observations before a beta is defined (else ``NaN``).

    Returns:
        Beta Series aligned to ``long_ret`` (``NaN`` during warm-up).
    """
    a = long_ret.astype(float)
    b = market_ret.reindex(a.index).astype(float)
    if window is None:
        cov = a.expanding(min_periods=min_obs).cov(b)
        var = b.expanding(min_periods=min_obs).var()
    else:
        cov = a.rolling(window, min_periods=min_obs).cov(b)
        var = b.rolling(window, min_periods=min_obs).var()
    beta = cov / var.replace(0.0, np.nan)
    return beta.shift(1)  # knowable at entry — no look-ahead


def inverse_hedged_return(
    long_ret: pd.Series,
    market_ret: pd.Series,
    *,
    leverage: float = 1.0,
    beta: float | pd.Series | None = None,
    beta_window: int | None = None,
    fee_per_period: float = 0.0,
    min_obs: int = 24,
) -> tuple[pd.Series, pd.Series]:
    """Long book hedged with a (synthetic) inverse ETF sized to neutralize beta.

    Holds the long book and buys ``beta / leverage`` notional of the inverse ETF,
    so the hedge cancels ``beta × market`` of the long book's return. With
    ``beta=None`` the beta is estimated lookahead-safely
    (:func:`rolling_beta`); pass a scalar for a fixed 1:1 (``beta=1.0``) hedge.

    Args:
        long_ret: Long-book return Series (e.g. long-only PEAD excess-of-cash, or
            a fully-invested Q10 book).
        market_ret: Market return Series (aligned).
        leverage: Inverse ETF leverage (1.0 = −1×, 2.0 = −2×). Higher leverage
            needs less notional (``beta/leverage``) so less fee drag.
        beta: Fixed hedge beta, a beta Series, or ``None`` to estimate.
        beta_window: Trailing window for the estimate (``None`` = expanding).
        fee_per_period: Inverse-ETF holding drag per period.
        min_obs: Warm-up for the beta estimate.

    Returns:
        ``(hedged_ret, beta_used)`` — both Series aligned to ``long_ret``. Periods
        with an undefined beta fall back to the *unhedged* long return (so the
        series is never silently dropped) with ``beta_used`` = ``NaN`` there.
    """
    a = long_ret.astype(float)
    m = market_ret.reindex(a.index).astype(float)
    if beta is None:
        beta_used = rolling_beta(a, m, window=beta_window, min_obs=min_obs)
    elif isinstance(beta, pd.Series):
        beta_used = beta.reindex(a.index).astype(float)
    else:
        beta_used = pd.Series(float(beta), index=a.index)

    inv = synth_inverse_return(m, leverage=leverage, fee_per_period=fee_per_period)
    units = beta_used / leverage
    hedged = a + units * inv
    # Warm-up (no beta yet) → hold unhedged rather than drop the period.
    hedged = hedged.where(beta_used.notna(), a)
    return hedged, beta_used
