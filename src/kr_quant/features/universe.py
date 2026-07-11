"""Point-in-time small-mid-cap universe — Minervini's preferred size tier.

The deployed Minervini scanner inverted Minervini's small-mid preference to a
100억+ large-cap universe (``SEPA_FAITHFUL_DESIGN.md`` §1.5, tagged 🔄 reversal).
The faithful arm A restores the small-mid tier: names sized as **genuinely
small-mid cap**, that also clear an ADV floor for tradeability.

⚠️ **Rank-band pitfall (discovered 2026-07-12):** the original ``cap_rank`` mode
ranks *within whatever universe is passed in* — when that universe is itself a
liquidity-filtered subset (e.g. a top-500-by-trade-value DART backfill), "rank
100–400 of 500" lands at **market-cap percentile ~90 of the full market** (≈1조
median), not small-mid at all. This silently defeated the "faithful re-creation"
experiment (arm A was tested on large/mega caps, not Minervini's actual habitat).
Use ``cap_band`` (absolute market cap, 원 units) instead whenever the input
universe is liquidity-pre-filtered; ``cap_rank`` is only safe against a genuinely
broad/full-market cap panel.

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
# Absolute small-mid band (원): 3천억–2조. Chosen over pure "<3천억" small-cap for
# adequate sample size against a liquidity-pre-filtered backfill (see module note);
# still squarely small-mid — nowhere near the 4.96조 median of the deployed scanner's
# large-cap universe (HONEST_VERDICT.md).
CAP_BAND = (3.0e11, 2.0e12)


def smallmid_universe(
    cap_panel: pd.DataFrame,
    adv_panel: pd.DataFrame,
    *,
    cap_rank: tuple[int, int] | None = CAP_RANK,
    cap_band: tuple[float, float] | None = None,
    adv_floor: float = ADV_FLOOR,
) -> pd.DataFrame:
    """Lookahead-safe (code × date) small-mid eligibility flag.

    Args:
        cap_panel: Long ``code``/``date``/``market_cap`` (as-of; e.g. price ×
            shares-outstanding). Provides the per-date cap ranking/band.
        adv_panel: Long ``code``/``date``/``adv`` (trailing 20-day average trade
            value, same 백만원 units as ``adv_floor``).
        cap_rank: ``(lo, hi)`` cap-rank band — eligible when ``lo < rank ≤ hi``
            (rank 1 = largest). Ignored when ``cap_band`` is given. ⚠️ Only
            meaningful against a broad/full-market cap panel — see module note.
        cap_band: ``(lo, hi)`` **absolute** market cap (원) — eligible when
            ``lo ≤ market_cap < hi``. Takes precedence over ``cap_rank`` when set;
            immune to the rank-band pitfall since it doesn't depend on what
            universe was passed in.
        adv_floor: Minimum ADV for tradeability (백만원).

    Returns:
        Long ``code``/``date``/``eligible`` (bool). ``eligible`` is True only where
        the name's size (band or per-date rank) qualifies **and** ADV clears the floor.
    """
    cap = cap_panel[["code", "date", "market_cap"]].copy()
    cap["date"] = cap["date"].astype(str)
    adv = adv_panel[["code", "date", "adv"]].copy()
    adv["date"] = adv["date"].astype(str)
    df = cap.merge(adv, on=["code", "date"], how="inner")
    if cap_band is not None:
        size_ok = (df["market_cap"] >= cap_band[0]) & (df["market_cap"] < cap_band[1])
    else:
        # Descending cap rank within each date (1 = largest); ties broken stably.
        df["cap_rank"] = df.groupby("date")["market_cap"].rank(ascending=False, method="first")
        size_ok = (df["cap_rank"] > cap_rank[0]) & (df["cap_rank"] <= cap_rank[1])
    df["eligible"] = size_ok & (df["adv"] >= adv_floor)
    return df[["code", "date", "eligible"]].reset_index(drop=True)
