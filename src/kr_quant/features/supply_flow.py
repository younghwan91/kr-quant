"""Investor-flow ("수급") feature signals: market-cap-normalized net-buy ratios,
EWMA smoothing/acceleration, and cross-sectional rank.

Kalman-filter-style state estimation was considered and rejected: the actual
collected history is only ~95 trading days, too short for a Kalman filter's
burn-in/convergence to be trustworthy. EWMA (which needs no burn-in period and
degrades gracefully on short/gappy series) and simple cross-sectional ranking
were chosen instead — a deliberate decision, not an oversight.

Like :mod:`kr_quant.strategies.accumulation`, every function here is pure
DataFrame in → DataFrame out: no DB connection, no network call. Callers are
responsible for loading ``supply_demand`` rows and joining in ``market_cap``
(e.g. via :func:`kr_quant.storage.market_cap_asof`) before calling
:func:`add_normalized_ratios`.
"""

from __future__ import annotations

import pandas as pd

from ..storage import INVESTOR_COLUMNS
from ._common import add_ewma_and_change

# Investor-type net-buy columns present in ``supply_demand`` (see
# storage.INVESTOR_COLUMNS): individual, foreign_, institution, fnnc_invt,
# insrnc, invtrt, bank, penfnd_etc, samo_fund, natn, etc_corp.
INVESTOR_TYPES: list[str] = list(INVESTOR_COLUMNS.keys())


def add_normalized_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-cap-normalized net-buy ratio columns for each investor type.

    Args:
        df: Rows with the investor-type net-buy columns (see
            ``INVESTOR_TYPES``) plus a ``market_cap`` column. The caller joins
            ``market_cap`` in (e.g. via ``storage.market_cap_asof``); this
            function does not touch the DB.

    Returns:
        A copy of ``df`` with one new ``{investor_type}_ratio`` column per
        investor type: net-buy amount divided by ``market_cap``. Rows where
        ``market_cap`` is missing or non-positive get ``NaN`` ratios rather
        than raising or dividing by zero.
    """
    out = df.copy()
    safe_cap = out["market_cap"].where(out["market_cap"] > 0)
    for col in INVESTOR_TYPES:
        out[f"{col}_ratio"] = out[col] / safe_cap
    return out


def add_ewma_signal(
    df: pd.DataFrame,
    col: str,
    *,
    halflife: float = 7,
) -> pd.DataFrame:
    """Add a per-code EWMA of ``col`` plus its day-over-day change ("acceleration").

    Args:
        df: Rows with ``code``, ``date``, and the ratio column ``col`` (e.g.
            ``"foreign__ratio"`` from :func:`add_normalized_ratios`).
        col: Name of the per-code, per-date normalized ratio column to smooth.
        halflife: EWMA halflife in trading days. 5-10 day range is the
            intended default (7); shorter reacts faster to recent flow,
            longer smooths more.

    Returns:
        A copy of ``df`` sorted by ``code``, ``date`` with two new columns:
        ``{col}_ewma`` (exponentially-weighted moving average) and
        ``{col}_ewma_diff`` (first difference of the EWMA — the
        "acceleration" signal). Each ``code`` group is smoothed independently
        (via ``groupby``), so no leakage across codes. Gaps (missing dates for
        a code) are tolerated: the EWMA simply operates on whatever rows exist
        for that code, in date order.
    """
    out = df.sort_values(["code", "date"]).copy()
    return add_ewma_and_change(
        out, col, halflife=halflife,
        ewma_col=f"{col}_ewma", change_col=f"{col}_ewma_diff",
    )


def add_avg_cost_gap(
    df: pd.DataFrame,
    investor_col: str,
    *,
    close_col: str = "close",
) -> pd.DataFrame:
    """Add a running volume-weighted average entry price and its gap to ``close``.

    Approximates "is this investor type still averaging down, or already
    sitting on gains" from public flow data alone (no direct cost-basis data
    exists): a cumulative, buy-volume-weighted average price built only from
    days that investor type was a net buyer.

    .. math::

        \\text{avg\\_cost}_{i,t} = \\frac{\\sum_{\\tau \\le t} P_{i,\\tau}
        \\cdot \\max(\\text{netbuy}_{i,\\tau}, 0)}
        {\\sum_{\\tau \\le t} \\max(\\text{netbuy}_{i,\\tau}, 0)}

    Args:
        df: Rows with ``code``, ``date``, ``close_col``, and the raw net-buy
            column ``investor_col`` (e.g. ``"penfnd_etc"`` — the *un*-ratio'd
            column from ``supply_demand``, not the ``_ratio``/``_ewma``
            derived ones, since this needs the actual traded volume/amount to
            weight by, not a market-cap-normalized ratio).
        investor_col: Which investor type's raw net-buy column to use (must
            be a key of :data:`INVESTOR_TYPES`).
        close_col: Price column name.

    Returns:
        A copy of ``df``, sorted by ``code``/``date``, with two new columns:
        ``{investor_col}_avg_cost`` (the running VWAP entry price; ``NaN``
        until that investor type has ever been a net buyer for that code) and
        ``{investor_col}_cost_gap`` (``(close - avg_cost) / avg_cost`` —
        positive means price is above this investor type's average entry,
        i.e. sitting on paper gains; negative means still underwater /
        plausibly still averaging down).
    """
    if investor_col not in INVESTOR_TYPES:
        raise ValueError(f"investor_col must be one of {INVESTOR_TYPES}, got {investor_col!r}")

    out = df.sort_values(["code", "date"]).copy()
    buy_vol = out[investor_col].clip(lower=0)
    weighted_price = out[close_col] * buy_vol

    grouped_wp = weighted_price.groupby(out["code"], sort=False)
    grouped_vol = buy_vol.groupby(out["code"], sort=False)
    cum_wp = grouped_wp.cumsum()
    cum_vol = grouped_vol.cumsum()

    cost_col = f"{investor_col}_avg_cost"
    gap_col = f"{investor_col}_cost_gap"
    out[cost_col] = (cum_wp / cum_vol.where(cum_vol > 0)).to_numpy()
    out[gap_col] = ((out[close_col] - out[cost_col]) / out[cost_col].where(out[cost_col] > 0)).to_numpy()
    return out


def add_cross_sectional_rank(
    df: pd.DataFrame,
    col: str,
    *,
    method: str = "pct",
) -> pd.DataFrame:
    """Add a cross-sectional rank of ``col`` among all codes, per date.

    For each ``date``, ranks all codes by ``col`` (e.g. a market-cap-normalized
    net-buy ratio) — a same-day comparison across the universe, not a rolling
    window over time.

    Args:
        df: Rows with ``date`` and the column to rank, ``col``.
        method: ``"pct"`` (default) returns a percentile rank in ``(0, 1]``,
            highest ``col`` value → rank close to 1. Any other value is passed
            through to ``pandas`` ``rank(pct=False, method=...)`` to return raw
            ranks (1 = lowest) instead.

    Returns:
        A copy of ``df`` with a new ``{col}_rank`` column.
    """
    out = df.copy()
    rank_col = f"{col}_rank"
    if method == "pct":
        out[rank_col] = out.groupby("date")[col].rank(pct=True)
    else:
        out[rank_col] = out.groupby("date")[col].rank(method=method)
    return out
