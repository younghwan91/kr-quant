"""Accumulation screener logic. Pure DataFrame in → ranked DataFrame out."""

from __future__ import annotations

import pandas as pd

from kiwoom_quant.strategies.accumulation import screen


def _stock_frame(code, name, closes, foreign, inst, indiv, vol=1_000_000):
    """Build a per-stock frame; lists are aligned by trading day."""
    n = len(closes)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "name": [name] * n,
            "market": ["거래소"] * n,
            "sector": ["테스트"] * n,
            "date": [f"202605{d:02d}" for d in range(1, n + 1)],
            "close": closes,
            "acc_trde_qty": [vol] * n,
            "individual": indiv,
            "foreign_": foreign,
            "institution": inst,
        }
    )


def test_sideways_accumulation_is_selected():
    days = 12
    # Sideways price (~100), foreign+institution buying, individuals selling.
    good = _stock_frame(
        "000001", "매집주",
        closes=[100, 101, 99, 100, 102, 100, 101, 99, 100, 101, 100, 102],
        foreign=[5000] * days, inst=[3000] * days, indiv=[-8000] * days,
    )
    result = screen(good, min_days=10, max_range_pct=0.15)
    assert list(result["code"]) == ["000001"]
    assert result.iloc[0]["foreign_cum"] > 0
    assert result.iloc[0]["indiv_cum"] < 0


def test_trending_or_distribution_excluded():
    days = 12
    # Trending price (range too wide) — should be filtered out despite buying.
    trending = _stock_frame(
        "000002", "급등주",
        closes=[100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
        foreign=[5000] * days, inst=[3000] * days, indiv=[-8000] * days,
    )
    # Sideways but smart money is NET SELLING — should be filtered out.
    distributing = _stock_frame(
        "000003", "분산주",
        closes=[100, 101, 99, 100, 102, 100, 101, 99, 100, 101, 100, 102],
        foreign=[-5000] * days, inst=[-3000] * days, indiv=[8000] * days,
    )
    result = screen(pd.concat([trending, distributing]), min_days=10, max_range_pct=0.15)
    assert result.empty


def test_min_days_filter():
    short = _stock_frame(
        "000004", "신규주",
        closes=[100, 101, 100], foreign=[5000] * 3, inst=[3000] * 3, indiv=[-8000] * 3,
    )
    assert screen(short, min_days=10).empty


def test_empty_input():
    assert screen(pd.DataFrame()).empty
