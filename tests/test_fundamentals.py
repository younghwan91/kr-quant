"""Lookahead-safe earnings feature. Pure DataFrame in -> DataFrame out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.fundamentals import (
    available_date,
    blend_rank,
    combined_signal,
    earnings_yield_panel,
    earnings_yoy_panel,
)


def test_available_date_applies_filing_lag():
    assert available_date("2022-03-31", is_annual=False) == pd.Timestamp("2022-05-15")
    assert available_date("2021-12-31", is_annual=True) == pd.Timestamp("2022-03-31")


def test_yoy_not_used_before_available_date():
    # Filing became public 2022-05-15; must be NaN before, set on/after.
    earnings = pd.DataFrame({"code": ["000660"], "avail_date": ["20220515"], "yoy": [1.26]})
    dates = ["2022-05-13", "2022-05-14", "2022-05-15", "2022-05-16"]
    panel = earnings_yoy_panel(earnings, dates).set_index("date")
    assert np.isnan(panel.loc["2022-05-14", "yoy"])
    assert panel.loc["2022-05-15", "yoy"] == 1.26
    assert panel.loc["2022-05-16", "yoy"] == 1.26


def test_latest_available_filing_wins_and_age_tracks_freshness():
    earnings = pd.DataFrame({
        "code": ["A", "A"],
        "avail_date": ["20220515", "20220814"],  # Q1 then Q2
        "yoy": [0.5, 0.9],
    })
    dates = ["2022-05-20", "2022-08-20"]
    panel = earnings_yoy_panel(earnings, dates).set_index("date")
    assert panel.loc["2022-05-20", "yoy"] == 0.5
    assert panel.loc["2022-08-20", "yoy"] == 0.9  # newer filing supersedes
    assert panel.loc["2022-05-20", "age_days"] == 5   # 05-15 -> 05-20
    assert panel.loc["2022-08-20", "age_days"] == 6   # 08-14 -> 08-20


def test_normalizes_dashed_and_plain_avail_dates():
    earnings = pd.DataFrame({"code": ["A"], "avail_date": ["2022-05-15"], "yoy": [1.0]})
    panel = earnings_yoy_panel(earnings, ["2022-05-16"]).set_index("date")
    assert panel.loc["2022-05-16", "yoy"] == 1.0


def test_earnings_yield_is_lookahead_safe_netinc_over_market_cap():
    annual = pd.DataFrame({"code": ["A"], "avail_date": ["20220331"], "netinc": [200.0]})
    mc = pd.DataFrame({
        "code": ["A", "A"],
        "date": ["2022-03-30", "2022-04-01"],
        "market_cap": [1000.0, 1000.0],
    })
    ep = earnings_yield_panel(annual, mc).set_index("date")
    # 03-30 is before the filing became public (03-31) -> NaN (no look-ahead).
    assert np.isnan(ep.loc["2022-03-30", "ep"])
    assert ep.loc["2022-04-01", "ep"] == 0.2  # 200 / 1000


def test_blend_rank_mixes_two_signals_by_percentile():
    # Two codes, one date; signal A ranks A>B, signal B ranks B>A. Equal blend -> tie.
    a = pd.DataFrame({"code": ["A", "B"], "date": ["d1", "d1"], "yoy": [1.0, 0.0]})
    b = pd.DataFrame({"code": ["A", "B"], "date": ["d1", "d1"], "ep": [0.0, 1.0]})
    out = blend_rank([a, b], [0.5, 0.5], value_cols=["yoy", "ep"]).set_index("code")
    assert abs(out.loc["A", "signal"] - out.loc["B", "signal"]) < 1e-9


def test_combined_signal_wires_growth_and_value():
    earnings = pd.DataFrame({
        "code": ["A", "A", "B", "B"],
        "period": ["2021Q4", "2022Q4", "2021Q4", "2022Q4"],
        "avail_date": ["20220331", "20230331", "20220331", "20230331"],
        "yoy": [0.5, 0.8, -0.2, 0.1],
        "netinc": [100.0, 120.0, 50.0, 55.0],
    })
    mc = pd.DataFrame({
        "code": ["A", "B"], "date": ["2023-06-01", "2023-06-01"],
        "market_cap": [1000.0, 1000.0],
    })
    sig = combined_signal(earnings, mc, ["2023-06-01"], value_weight=0.25)
    # Both codes present with a finite blended signal; A (higher growth) ranks above B.
    s = sig.set_index("code")["signal"]
    assert s.notna().all() and s.loc["A"] > s.loc["B"]
