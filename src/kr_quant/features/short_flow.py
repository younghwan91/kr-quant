"""Short-covering signal: day-over-day reduction in short-sale balance.

A falling ``short_balance`` (outstanding short position) is direct evidence
of covering — someone is closing a short, which is itself a source of buying
pressure independent of ordinary investor-type net-buy flow (it's forced/
incentivized buying, not discretionary accumulation). ``short_selling`` is
collected separately from ``supply_demand`` (see
``kr_quant.collectors.short_credit``), so this is its own small module rather
than added to :mod:`kr_quant.features.supply_flow`.

Pure DataFrame in → DataFrame out, consistent with the rest of this package.
"""

from __future__ import annotations

import pandas as pd


def add_short_covering_signal(
    df: pd.DataFrame,
    *,
    balance_col: str = "short_balance",
) -> pd.DataFrame:
    """Add a normalized day-over-day short-covering signal.

    Args:
        df: Rows with ``code``, ``date``, and ``balance_col`` (outstanding
            short-sale balance, e.g. from the ``short_selling`` table's
            ``short_balance`` column).
        balance_col: Name of the outstanding-short-balance column.

    Returns:
        A copy of ``df``, sorted by ``code``/``date``, with two new columns:

        - ``short_balance_chg``: raw day-over-day change in ``balance_col``
          (negative = balance shrank = covering happened).
        - ``short_covering``: the change normalized by the *prior* day's
          balance (``-chg / prior_balance``), so it reads as "fraction of
          the outstanding short position covered that day" — positive means
          covering (buying pressure), negative means shorts were added
          (selling pressure). ``NaN`` when there's no prior-day balance to
          normalize by (first observation for that code, or prior balance is
          zero/missing).
    """
    out = df.sort_values(["code", "date"]).copy()
    grouped = out.groupby("code", sort=False)[balance_col]
    prior = grouped.shift(1)
    chg = out[balance_col] - prior

    out["short_balance_chg"] = chg
    out["short_covering"] = (-chg / prior.where(prior > 0)).to_numpy()
    return out
