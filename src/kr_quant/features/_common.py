"""Shared low-level primitives for feature modules.

Keeps the per-code EWMA-level + first-difference pattern in one place so
``supply_flow``, ``credit_flow`` (and any future flow feature) compute it
identically instead of re-implementing the same ``groupby(...).transform``
chain. Pure DataFrame in → DataFrame out, like the rest of the package.
"""

from __future__ import annotations

import pandas as pd


def add_ewma_and_change(
    df: pd.DataFrame,
    col: str,
    *,
    halflife: float,
    ewma_col: str,
    change_col: str,
) -> pd.DataFrame:
    """Add a per-``code`` EWMA of ``col`` and its day-over-day change.

    Assumes ``df`` is already sorted by ``code``/``date`` (callers do this).
    Each ``code`` group is smoothed independently (no leakage across codes);
    gaps are tolerated — the EWMA operates on whatever rows exist per code in
    date order.

    Args:
        df: Rows with a ``code`` column and the numeric column ``col``.
        col: Column to smooth.
        halflife: EWMA halflife in trading days.
        ewma_col: Output column name for the smoothed level.
        change_col: Output column name for the first difference of the EWMA.

    Returns:
        ``df`` with ``ewma_col`` and ``change_col`` added (mutates and returns
        the same frame; callers pass a copy).
    """
    grouped = df.groupby("code", sort=False, group_keys=False)[col]
    df[ewma_col] = grouped.transform(
        lambda s: s.ewm(halflife=halflife, adjust=False).mean()
    )
    df[change_col] = df.groupby("code", sort=False, group_keys=False)[ewma_col].transform(
        lambda s: s.diff()
    )
    return df
