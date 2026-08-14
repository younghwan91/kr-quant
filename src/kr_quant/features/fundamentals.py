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


def _asof_yoy(ea: pd.DataFrame, dates: pd.Series, value_col: str) -> pd.DataFrame:
    """code 별 backward as-of 조인 — (code, date, yoy, age_days) 롱 프레임."""
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
    return pd.concat(frames, ignore_index=True)


def earnings_yoy_panel(
    earnings: pd.DataFrame,
    trading_dates: list[str],
    *,
    avail_col: str = "avail_date",
    value_col: str = "yoy",
    knowledge_col: str | None = None,
    key_cols: tuple[str, ...] = ("code", "period"),
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
        knowledge_col: 정정공시 버전 컬럼(보통 ``knowledge_date``). 주면 **이중
            시간축(bitemporal)** 으로 판단한다 — 어느 분기가 공시됐나(``avail_col``)와
            그 분기의 어느 버전을 그때 알고 있었나(``knowledge_col``)는 서로 독립인
            축이라, 하나의 스칼라 as-of 로는 표현되지 않는다.

            생략하면 기존 동작 그대로다(입력에 버전이 하나뿐이라고 가정). 발표 수치와
            ``tests/test_parity_pead.py`` 가 그 경로에 고정돼 있다.
        key_cols: 버전을 묶는 키. ``knowledge_col`` 을 줄 때만 쓰인다.

    Returns:
        Long DataFrame with columns ``code``, ``date`` (from ``trading_dates``),
        ``yoy`` (latest available value, ``NaN`` before a code's first filing),
        and ``age_days`` (calendar days since that filing became available — the
        PEAD "freshness", used to isolate the post-announcement drift window).

    Note:
        **왜 ``max(avail, knowledge)`` 로 정렬하면 안 되는가.** 그러면 "2024Q4를 이미
        아는 상태에서 2020Q1이 정정된" 경우 정정본의 valid_from 이 더 커서, 옛 분기를
        최신 신호로 집는다. 그래서 버전이 바뀌는 시점으로 타임라인을 쪼개고, 각 구간
        안에서는 그때 알던 스냅샷으로 기존 as-of 조인을 돌린다. 정정이 없으면 구간이
        하나뿐이라 기존 경로와 완전히 동일하다.
    """
    dates = pd.to_datetime(pd.Series(sorted(trading_dates), name="date"))

    cols = ["code", avail_col, value_col]
    if knowledge_col is not None:
        cols += [c for c in (*key_cols, knowledge_col) if c not in cols]
    ea = earnings[cols].copy()
    ea["_avail"] = pd.to_datetime(ea[avail_col].astype(str).str.replace("-", ""), format="%Y%m%d")
    ea = ea.dropna(subset=["_avail", value_col]).sort_values("_avail")

    if knowledge_col is None:
        out = _asof_yoy(ea, dates, value_col)
        if not out.empty:
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out

    ea["_know"] = pd.to_datetime(
        ea[knowledge_col].astype(str).str.replace("-", ""), format="%Y%m%d")
    ea = ea.dropna(subset=["_know"])
    # 이 행을 실제로 쓸 수 있게 된 날. 늦게 수집된 공시(_know > _avail)도 여기서 걸린다.
    ea["_valid_from"] = ea[["_avail", "_know"]].max(axis=1)

    keys = list(key_cols)
    n_versions = ea.groupby(keys, sort=False)["_know"].transform("size")
    # "평범한" 행 = 버전이 하나뿐이고 공시일에 알게 된 것. 이런 행만 있으면 knowledge
    # 제약이 avail 제약에 포함되므로 구간을 쪼갤 이유가 없다.
    plain = (n_versions == 1) & (ea["_know"] <= ea["_avail"])
    events = sorted(ea.loc[~plain, "_valid_from"].unique())

    if not events:
        out = _asof_yoy(ea, dates, value_col)
        if not out.empty:
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out

    bounds = [dates.min()] + [e for e in events if dates.min() < e <= dates.max()]
    frames = []
    for i, lo in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < len(bounds) else None
        seg = dates[(dates >= lo) & ((dates < hi) if hi is not None else True)]
        if seg.empty:
            continue
        # 구간 안에서 버전 선택이 필요한 건 **버전이 여럿이거나 늦게 수집된 키**뿐이다.
        # 평범한 행은 _know == _avail 이라 버전 제약이 분기 제약과 같으므로, 구간 시작
        # 시점으로 거르면 구간 *안에서* 공시되는 최신 분기까지 잘려나간다 — 정정 하나가
        # 무관한 과거 날짜의 값을 바꿔버린다(실측으로 잡음). 그래서 평범한 행은 전부
        # 넘기고 날짜별 게이팅은 아래 merge_asof 에 맡긴다.
        versioned = ea[~plain]
        versioned = versioned[versioned["_know"] <= seg.iloc[0]]
        if not versioned.empty:
            versioned = versioned.sort_values("_know").drop_duplicates(subset=keys, keep="last")
        known = pd.concat([ea[plain], versioned], ignore_index=True)
        seg = seg.reset_index(drop=True)
        if known.empty:
            # 아직 아무 공시도 모르는 구간. 행을 빼면 안 된다 — 기본 경로는 첫 공시
            # 이전 날짜에도 yoy=NaN 행을 남기고, 하류 패널이 그 자리를 기대한다.
            frames.append(pd.DataFrame({
                "code": np.repeat(ea["code"].unique(), len(seg)),
                "date": np.tile(seg.to_numpy(), ea["code"].nunique()),
                "yoy": np.nan, "age_days": np.nan,
            }))
            continue
        frames.append(_asof_yoy(known.sort_values("_avail"), seg, value_col))

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
    """Vectorized YoY = (cur - prior) / |prior|; NaN where prior is 0/missing.

    Private by name but imported by the research gate runners (pead_gate,
    pead_concentrated_gate, pead_refinement, prop_feasibility) — treat it as part
    of this module's surface, not an internal detail free to delete.
    """
    p = prior.where(prior.notna() & (prior != 0))
    return (cur - p) / p.abs()
