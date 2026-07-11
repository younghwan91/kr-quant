"""Fundamental (실적) features from DART quarterly filings, lookahead-safe.

The one validated alpha in this project is post-earnings-announcement drift
(PEAD): rank stocks by year-over-year net-income growth and hold a low-turnover
dollar-neutral book. See :mod:`kr_quant.strategies.pead`.

The critical correctness property here is **no look-ahead**: a quarter's YoY
figure may only be used on/after its ``avail_date`` (the day the filing became
public = quarter-end + a filing lag, ~45 days for quarterly / ~90 for annual).
:func:`earnings_yoy_panel` enforces this with a backward as-of join: for each
trading date it attaches, per code, the most recent YoY whose ``avail_date`` is
on or before that date.

Pure DataFrame in → DataFrame out, consistent with the rest of this package —
no DB connection, no network. Callers load the raw DART rows (see
``kr_quant.collectors.dart_earnings``) and the trading calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Filing lag in calendar days from period-end to public availability.
# Korean quarterly reports file within ~45 days; annual within ~90.
QUARTER_LAG_DAYS = 45
ANNUAL_LAG_DAYS = 90


def available_date(period_end: pd.Timestamp | str, *, is_annual: bool) -> pd.Timestamp:
    """The date a report for ``period_end`` becomes publicly usable (no look-ahead).

    Args:
        period_end: Quarter/annual period-end date.
        is_annual: Annual reports get the longer :data:`ANNUAL_LAG_DAYS` lag;
            quarterly/half-year reports get :data:`QUARTER_LAG_DAYS`.

    Returns:
        ``period_end`` plus the appropriate filing lag, as a Timestamp.
    """
    lag = ANNUAL_LAG_DAYS if is_annual else QUARTER_LAG_DAYS
    return pd.Timestamp(period_end) + pd.Timedelta(days=lag)


def earnings_yoy_panel(
    earnings: pd.DataFrame,
    trading_dates: list[str],
    *,
    avail_col: str = "avail_date",
    value_col: str = "yoy",
) -> pd.DataFrame:
    """Build a lookahead-safe (code × date) panel of the latest available YoY.

    For each ``code`` and each trading date, selects the most recent earnings
    row whose ``avail_col`` is on or before that date (a backward as-of join),
    so a figure is never used before it was public.

    Args:
        earnings: One row per (code, filing) with columns ``code``, ``avail_col``
            (date the filing became public), and ``value_col`` (the signal, e.g.
            net-income YoY growth). ``avail_col`` may be ``YYYYMMDD`` or
            ``YYYY-MM-DD`` — both are normalized to datetime.
        trading_dates: Sorted trading dates (``YYYY-MM-DD``) to build the panel
            over — typically ``sorted(daily_bars["date"].unique())``.
        avail_col: Availability-date column name.
        value_col: Signal column name to forward-fill from each filing.

    Returns:
        Long DataFrame with columns ``code``, ``date`` (from ``trading_dates``),
        ``yoy`` (latest available value, ``NaN`` before a code's first filing),
        and ``age_days`` (calendar days since that filing became available — the
        PEAD "freshness", used to isolate the post-announcement drift window).
    """
    dates = pd.to_datetime(pd.Series(sorted(trading_dates), name="date"))
    ea = earnings[["code", avail_col, value_col]].copy()
    ea["_avail"] = pd.to_datetime(ea[avail_col].astype(str).str.replace("-", ""), format="%Y%m%d")
    ea = ea.dropna(subset=["_avail", value_col]).sort_values("_avail")

    frames: list[pd.DataFrame] = []
    right_template = pd.DataFrame({"date": dates})
    for code, g in ea.groupby("code", sort=False):
        merged = pd.merge_asof(
            right_template, g.rename(columns={value_col: "yoy"})[["_avail", "yoy"]],
            left_on="date", right_on="_avail", direction="backward",
        )
        merged["code"] = code
        merged["age_days"] = (merged["date"] - merged["_avail"]).dt.days
        frames.append(merged[["code", "date", "yoy", "age_days"]])

    if not frames:
        return pd.DataFrame(columns=["code", "date", "yoy", "age_days"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def earnings_yield_panel(
    annual_earnings: pd.DataFrame,
    market_cap: pd.DataFrame,
    *,
    avail_col: str = "avail_date",
    earnings_col: str = "netinc",
) -> pd.DataFrame:
    """Lookahead-safe (code × date) earnings yield E/P = annual net income / market cap.

    A value factor that complements PEAD (earnings *growth*): it is strongly and
    persistently predictive on its own (4/4 regime buckets in validation) and,
    blended in at a modest weight via :func:`blend_rank`, lifts the combined
    signal's regime persistence without hurting net Sharpe. Uses **annual** net
    income (full-year, filed ~90 days after year-end) so the figure is stable and
    the ``avail_date`` lag is honest.

    Args:
        annual_earnings: One row per (code, fiscal year) with ``code``,
            ``avail_col`` (public date), and ``earnings_col`` (annual net income).
        market_cap: Long ``code``/``date``/``market_cap`` panel (e.g. price ×
            shares-outstanding as-of). Provides the denominator per trading date.
        avail_col: Availability-date column on ``annual_earnings``.
        earnings_col: Annual net-income column on ``annual_earnings``.

    Returns:
        Long ``code``/``date``/``ep`` DataFrame: the latest available annual net
        income (as of each date, no look-ahead) divided by that date's market
        cap. ``NaN`` where earnings or a positive market cap is unavailable.
    """
    dates = sorted(market_cap["date"].astype(str).unique())
    ni = earnings_yoy_panel(
        annual_earnings, dates, avail_col=avail_col, value_col=earnings_col,
    ).rename(columns={"yoy": "netinc"})[["code", "date", "netinc"]]
    mc = market_cap[["code", "date", "market_cap"]].copy()
    mc["date"] = mc["date"].astype(str)
    out = ni.merge(mc, on=["code", "date"], how="inner")
    cap = out["market_cap"].where(out["market_cap"] > 0)
    out["ep"] = out["netinc"] / cap
    return out[["code", "date", "ep"]]


def blend_rank(panels: list[pd.DataFrame], weights: list[float], *, value_cols: list[str]) -> pd.DataFrame:
    """Blend several signal panels into one via per-date cross-sectional rank.

    Each panel is percentile-ranked within each date (so incomparable raw scales
    — growth ratio vs earnings yield — combine fairly), then the ranks are mixed
    by ``weights``. This is how PEAD (growth) and E/P (value) are combined.

    Args:
        panels: Long ``code``/``date``/value DataFrames, one per signal.
        weights: Mixing weights (need not sum to 1; relative scale is what matters).
        value_cols: The value column name in each corresponding panel.

    Returns:
        Long ``code``/``date``/``signal`` DataFrame of the weighted rank blend,
        over the union of (code, date) present in any panel.
    """
    merged: pd.DataFrame | None = None
    for i, (panel, col) in enumerate(zip(panels, value_cols)):
        p = panel[["code", "date", col]].copy()
        p["date"] = p["date"].astype(str)
        p[f"_r{i}"] = p.groupby("date")[col].rank(pct=True)
        p = p[["code", "date", f"_r{i}"]]
        merged = p if merged is None else merged.merge(p, on=["code", "date"], how="outer")

    assert merged is not None
    rank_cols = [f"_r{i}" for i in range(len(panels))]
    w = pd.Series(weights, index=rank_cols)
    merged["signal"] = (merged[rank_cols] * w).sum(axis=1, min_count=1)
    return merged[["code", "date", "signal"]].dropna(subset=["signal"])


def combined_signal(
    earnings: pd.DataFrame,
    market_cap: pd.DataFrame,
    trading_dates: list[str],
    *,
    value_weight: float = 0.25,
) -> pd.DataFrame:
    """The validated tradeable signal: PEAD (YoY growth) ⊕ value (E/P), blended.

    Convenience wrapper over :func:`earnings_yoy_panel`, :func:`earnings_yield_panel`
    and :func:`blend_rank` encoding the validated recipe. Feed the result to
    :func:`kr_quant.strategies.pead.pead_backtest` as ``signal_panel``. The
    validated **tradeable** configuration is a long-only, large-cap book
    (``adv_floor≈20000`` 백만원, ``long_only=True``, ``horizon=40``): IR ≈ 1.0,
    4/4 regime buckets positive, no shorting required.

    Args:
        earnings: DART rows with ``code``, ``avail_date``, ``yoy``, ``netinc``,
            and ``period`` (annual rows identified by ``period`` ending "Q4").
        market_cap: Long ``code``/``date``/``market_cap`` panel (price × shares).
        trading_dates: Trading dates to build the YoY panel over.
        value_weight: Weight on the E/P value leg (growth leg gets the rest).
            0.25 is the validated default — enough to lift regime persistence
            without diluting the growth signal's net Sharpe.

    Returns:
        Long ``code``/``date``/``signal`` DataFrame ready for ``pead_backtest``.
    """
    yoy = earnings_yoy_panel(earnings.dropna(subset=["yoy"]), trading_dates)
    annual = earnings[earnings["period"].astype(str).str.endswith("Q4")]
    ep = earnings_yield_panel(annual, market_cap)
    return blend_rank([yoy, ep], [1.0 - value_weight, value_weight], value_cols=["yoy", "ep"])


def _yoy_vec(cur: pd.Series, prior: pd.Series) -> pd.Series:
    """Vectorized YoY = (cur - prior) / |prior|; NaN where prior is 0/missing."""
    p = prior.where(prior.notna() & (prior != 0))
    return (cur - p) / p.abs()


def _consec_accel(vals: "np.ndarray", i: int, steps: int) -> bool:
    """True iff ``vals[i-steps..i]`` are all finite and strictly increasing —
    ``steps`` consecutive quarter-on-quarter accelerations ending at quarter ``i``."""
    if i < steps:
        return False
    tail = vals[i - steps:i + 1]
    return bool(np.all(np.isfinite(tail)) and np.all(np.diff(tail) > 0))


def code33_panel(
    financials: pd.DataFrame,
    trading_dates: list[str],
    *,
    accel_steps: int = 3,
    avail_col: str = "avail_date",
) -> pd.DataFrame:
    """Lookahead-safe (code × date) Minervini **Code 33** acceleration flag.

    Code 33 = EPS **and** revenue **and** margin all accelerating for the last
    ``accel_steps`` quarters (each quarter's YoY strictly above the previous). This
    is the faithful SEPA fundamental gate (``SEPA_FAITHFUL_DESIGN.md`` §1.3d) that
    the deployed scanner omits. Feed ``fetch_financials`` output (net income +
    revenue + operating income, current & prior).

    Args:
        financials: One row per (code, quarter) with ``code``, ``avail_col``
            (public date), ``netinc``/``netinc_prior``, ``revenue``/
            ``revenue_prior``, ``op_income``/``op_income_prior``.
        trading_dates: Sorted trading dates to build the panel over.
        accel_steps: Consecutive quarterly accelerations required (default 3 = the
            "33" in Code 33 — 3 quarters of accelerating EPS/sales/margin).
        avail_col: Availability-date column (period-end + filing lag, no look-ahead).

    Returns:
        Long ``code``/``date``/``is_code33`` (bool) plus the component quarter YoYs
        ``yoy_eps``/``yoy_rev``/``yoy_margin`` (margin YoY = op-margin change in pp).
        Each date carries the most recent quarter whose ``avail_col`` ≤ date (a
        backward as-of join); dates before a code's first filing are ``is_code33``
        False. A quarter missing revenue/op-income yields NaN component YoYs → that
        quarter (and any window touching it) is ``is_code33`` False, never a crash.
    """
    f = financials.copy()
    f["yoy_eps"] = _yoy_vec(f["netinc"], f["netinc_prior"])
    f["yoy_rev"] = _yoy_vec(f["revenue"], f["revenue_prior"])
    margin = f["op_income"] / f["revenue"].where(f["revenue"].notna() & (f["revenue"] != 0))
    margin_prior = f["op_income_prior"] / f["revenue_prior"].where(
        f["revenue_prior"].notna() & (f["revenue_prior"] != 0))
    f["yoy_margin"] = margin - margin_prior  # margin expansion vs prior year (pp)
    f["_avail"] = pd.to_datetime(f[avail_col].astype(str).str.replace("-", ""), format="%Y%m%d")
    f = f.dropna(subset=["_avail"]).sort_values(["code", "_avail"])

    quarter_rows: list[pd.DataFrame] = []
    for code, g in f.groupby("code", sort=False):
        e = g["yoy_eps"].to_numpy(float)
        r = g["yoy_rev"].to_numpy(float)
        m = g["yoy_margin"].to_numpy(float)
        flags = [
            _consec_accel(e, i, accel_steps)
            and _consec_accel(r, i, accel_steps)
            and _consec_accel(m, i, accel_steps)
            for i in range(len(g))
        ]
        quarter_rows.append(g.assign(is_code33=flags)[
            ["code", "_avail", "is_code33", "yoy_eps", "yoy_rev", "yoy_margin"]])

    cols = ["code", "date", "is_code33", "yoy_eps", "yoy_rev", "yoy_margin"]
    if not quarter_rows:
        return pd.DataFrame(columns=cols)
    quarters = pd.concat(quarter_rows, ignore_index=True).sort_values("_avail")

    dates = pd.to_datetime(pd.Series(sorted(trading_dates), name="date"))
    right_template = pd.DataFrame({"date": dates})
    frames: list[pd.DataFrame] = []
    for code, g in quarters.groupby("code", sort=False):
        merged = pd.merge_asof(
            right_template, g.drop(columns="code"),
            left_on="date", right_on="_avail", direction="backward",
        )
        merged["code"] = code
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    out["is_code33"] = out["is_code33"].fillna(False).astype(bool)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[cols]
