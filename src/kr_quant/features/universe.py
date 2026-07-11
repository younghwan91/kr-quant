"""Point-in-time small-mid-cap universe — Minervini's preferred size tier.

The deployed Minervini scanner inverted Minervini's small-mid preference to a
100억+ large-cap universe (``SEPA_FAITHFUL_DESIGN.md`` §1.5, tagged 🔄 reversal).
The faithful arm A restores the small-mid tier: each date, the names ranked
**100–400th by market cap** that also clear a **10억 ADV** floor for tradeability.

Point-in-time by construction: the caller passes as-of market-cap and ADV panels
(from ``shares_outstanding_history`` × price, and trailing 20-day trade value), so
the "small-mid" classification never peeks ahead. ⚠️ PIT removes look-ahead
classification only — delisting survivorship is a separate, unresolved gap
(see design §정직검증 프레임 item 3).

Pure DataFrame in → DataFrame out.
"""

from __future__ import annotations

import pandas as pd

# Frozen SEPA hyperparameters (SEPA_FAITHFUL_DESIGN.md §사전등록 동결표).
CAP_RANK = (100, 400)       # market-cap rank band (exclude mega 1–100, keep 101–400)
ADV_FLOOR = 10000.0         # 20d ADV ≥ 10억 (백만원 units: 10000 = 10억)


def smallmid_universe(
    cap_panel: pd.DataFrame,
    adv_panel: pd.DataFrame,
    *,
    cap_rank: tuple[int, int] = CAP_RANK,
    adv_floor: float = ADV_FLOOR,
) -> pd.DataFrame:
    """Lookahead-safe (code × date) small-mid eligibility flag.

    Args:
        cap_panel: Long ``code``/``date``/``market_cap`` (as-of; e.g. price ×
            shares-outstanding). Provides the per-date cap ranking.
        adv_panel: Long ``code``/``date``/``adv`` (trailing 20-day average trade
            value, same 백만원 units as ``adv_floor``).
        cap_rank: ``(lo, hi)`` cap-rank band — eligible when ``lo < rank ≤ hi``
            (rank 1 = largest). Default excludes mega/large (top 100), keeps 101–400.
        adv_floor: Minimum ADV for tradeability (백만원).

    Returns:
        Long ``code``/``date``/``eligible`` (bool). ``eligible`` is True only where
        the name's per-date cap rank falls in the band **and** ADV clears the floor.
    """
    cap = cap_panel[["code", "date", "market_cap"]].copy()
    cap["date"] = cap["date"].astype(str)
    adv = adv_panel[["code", "date", "adv"]].copy()
    adv["date"] = adv["date"].astype(str)
    df = cap.merge(adv, on=["code", "date"], how="inner")
    # Descending cap rank within each date (1 = largest); ties broken stably.
    df["cap_rank"] = df.groupby("date")["market_cap"].rank(ascending=False, method="first")
    df["eligible"] = (
        (df["cap_rank"] > cap_rank[0])
        & (df["cap_rank"] <= cap_rank[1])
        & (df["adv"] >= adv_floor)
    )
    return df[["code", "date", "eligible"]].reset_index(drop=True)
