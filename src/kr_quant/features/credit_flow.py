"""Credit-balance (신용잔고) signal: leveraged-buying crowding and its trend.

Unlike investor-type net-buy flow (which needs market-cap normalization to
be comparable across stocks), ``credit_balance.balance_rt``(신용잔고율, %) is
*already* a normalized percentage (outstanding credit balance relative to
listed shares) straight from the source data — no extra normalization step
needed here.

Direction is deliberately left for the model to learn, not assumed: rising
credit balance could mean building momentum (more investors willing to lever
up into the move) or crowding risk (a forced-unwind air pocket waiting to
happen) — this module only computes the smoothed level/trend; whether it
turns out to help forward returns positively or negatively is exactly what
the ridge ensemble's learned sign should reveal.

Pure DataFrame in → DataFrame out, consistent with the rest of this package.
"""

from __future__ import annotations

import pandas as pd

from ._common import add_ewma_and_change


def add_credit_signal(
    df: pd.DataFrame,
    *,
    balance_rt_col: str = "balance_rt",
    halflife: float = 7.0,
) -> pd.DataFrame:
    """Add an EWMA-smoothed credit-ratio level and its day-over-day change.

    Args:
        df: Rows with ``code``, ``date``, and ``balance_rt_col`` (from the
            ``credit_balance`` table).
        balance_rt_col: Name of the already-normalized credit-ratio column.
        halflife: EWMA halflife in trading days (same convention as
            :func:`kr_quant.features.supply_flow.add_ewma_signal`).

    Returns:
        A copy of ``df``, sorted by ``code``/``date``, with two new columns:
        ``credit_balance_rt_ewma`` (smoothed credit-ratio level) and
        ``credit_balance_rt_chg`` (its day-over-day change — "is credit
        crowding building or unwinding"). Each ``code`` group is smoothed
        independently; gaps are tolerated the same way
        :func:`kr_quant.features.supply_flow.add_ewma_signal` tolerates them.
    """
    out = df.sort_values(["code", "date"]).copy()
    return add_ewma_and_change(
        out, balance_rt_col, halflife=halflife,
        ewma_col="credit_balance_rt_ewma", change_col="credit_balance_rt_chg",
    )
