"""Accumulation screener logic. Pure DataFrame in → ranked DataFrame out."""

from __future__ import annotations

import pandas as pd

from kr_quant.strategies.accumulation import screen


def _stock_frame(code, name, closes, foreign, inst, indiv, vol=1_000_000, longterm=None):
    """Build a per-stock frame; lists are aligned by trading day.

    ``longterm`` seeds 연기금(penfnd_etc); invtrt is left at 0.
    """
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
            "penfnd_etc": longterm if longterm is not None else [0] * n,
            "invtrt": [0] * n,
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


def test_require_longterm_inst_filters_program_only_buying():
    days = 12
    closes = [100, 101, 99, 100, 102, 100, 101, 99, 100, 101, 100, 102]
    # 기관 순매수지만 장기자금(연기금/투신)은 0 → 강화 조건에서 제외.
    program_only = _stock_frame(
        "000010", "프로그램주", closes=closes,
        foreign=[5000] * days, inst=[3000] * days, indiv=[-8000] * days,
        longterm=[0] * days,
    )
    # 연기금이 실제로 순매수 → 통과.
    longterm_buy = _stock_frame(
        "000011", "연기금주", closes=closes,
        foreign=[5000] * days, inst=[3000] * days, indiv=[-8000] * days,
        longterm=[2000] * days,
    )
    df = pd.concat([program_only, longterm_buy])
    result = screen(df, min_days=10, require_longterm_inst=True)
    assert list(result["code"]) == ["000011"]


def test_min_avg_vol_filters_illiquid():
    days = 12
    closes = [100, 101, 99, 100, 102, 100, 101, 99, 100, 101, 100, 102]
    illiquid = _stock_frame(
        "000012", "저유동주", closes=closes,
        foreign=[5000] * days, inst=[3000] * days, indiv=[-8000] * days, vol=1000,
    )
    assert screen(illiquid, min_days=10, min_avg_vol=100_000).empty


def test_empty_input():
    assert screen(pd.DataFrame()).empty
