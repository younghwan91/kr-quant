"""Multi-channel supply-wave signal: avg-cost-gap, short-covering, combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kr_quant.features.short_flow import add_short_covering_signal
from kr_quant.features.supply_flow import add_avg_cost_gap
from kr_quant.strategies.multi_signal import (
    build_multi_channel_signal,
    walk_forward_multi_signal_eval,
)


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def test_add_avg_cost_gap_tracks_vwap_of_buy_days():
    # Buys only on day 1 (price 100) and day 3 (price 120); day 2 is a sell
    # (negative netbuy, must be excluded from the cost basis).
    df = pd.DataFrame(
        {
            "code": ["A"] * 4,
            "date": _dates(4),
            "close": [100, 90, 120, 130],
            "penfnd_etc": [10, -5, 10, 0],
        }
    )
    out = add_avg_cost_gap(df, "penfnd_etc")
    # After day1: avg_cost = 100. After day3: avg_cost = (100*10+120*10)/20 = 110.
    assert np.isclose(out.iloc[0]["penfnd_etc_avg_cost"], 100.0)
    assert np.isclose(out.iloc[2]["penfnd_etc_avg_cost"], 110.0)
    # Day 4 has no new buy volume, cost basis carries forward at 110; close=130
    # is above it -> positive gap (sitting on paper gains).
    assert out.iloc[3]["penfnd_etc_cost_gap"] > 0


def test_add_avg_cost_gap_nan_before_first_buy():
    df = pd.DataFrame(
        {"code": ["A"], "date": _dates(1), "close": [100], "penfnd_etc": [-5]}
    )
    out = add_avg_cost_gap(df, "penfnd_etc")
    assert pd.isna(out.iloc[0]["penfnd_etc_avg_cost"])


def test_add_avg_cost_gap_rejects_unknown_investor_col():
    df = pd.DataFrame({"code": ["A"], "date": _dates(1), "close": [100], "x": [1]})
    with pytest.raises(ValueError):
        add_avg_cost_gap(df, "not_a_real_investor_type")


def test_short_covering_positive_when_balance_shrinks():
    df = pd.DataFrame(
        {
            "code": ["A", "A", "A"],
            "date": _dates(3),
            "short_balance": [1000, 800, 800],
        }
    )
    out = add_short_covering_signal(df)
    # Day2: balance dropped 1000->800, a 20% covering of the prior balance.
    assert np.isclose(out.iloc[1]["short_covering"], 0.2)
    # Day3: no change -> covering signal is 0, not NaN.
    assert np.isclose(out.iloc[2]["short_covering"], 0.0)
    # Day1: no prior balance -> NaN.
    assert pd.isna(out.iloc[0]["short_covering"])


def test_short_covering_negative_when_shorts_increase():
    df = pd.DataFrame(
        {"code": ["A", "A"], "date": _dates(2), "short_balance": [1000, 1500]}
    )
    out = add_short_covering_signal(df)
    assert out.iloc[1]["short_covering"] < 0


def _synthetic_multi_frame(n_codes=6, n_days=20, seed=0):
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(n_codes)]
    dates = _dates(n_days)
    rows = []
    short_rows = []
    for code in codes:
        price = 1000.0
        balance = 10000.0
        for date in dates:
            price *= 1 + rng.normal(0, 0.01)
            netbuy_foreign = rng.integers(-5000, 5000)
            netbuy_penfnd = rng.integers(-5000, 5000)
            netbuy_samo = rng.integers(-5000, 5000)
            rows.append(
                {
                    "code": code,
                    "date": date,
                    "close": price,
                    "market_cap": 1_000_000_000,
                    "individual": 0,
                    "foreign_": netbuy_foreign,
                    "institution": 0,
                    "fnnc_invt": 0,
                    "insrnc": 0,
                    "invtrt": 0,
                    "bank": 0,
                    "penfnd_etc": netbuy_penfnd,
                    "samo_fund": netbuy_samo,
                    "natn": 0,
                    "etc_corp": 0,
                }
            )
            balance = max(balance + rng.integers(-500, 500), 0)
            short_rows.append({"code": code, "date": date, "short_balance": balance})
    return pd.DataFrame(rows), pd.DataFrame(short_rows)


def test_build_multi_channel_signal_produces_bounded_composite():
    supply_df, short_df = _synthetic_multi_frame()
    out = build_multi_channel_signal(supply_df, short_df)
    assert "multi_signal" in out.columns
    valid = out["multi_signal"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 1).all()


def test_build_multi_channel_signal_rejects_bad_flow_channel():
    supply_df, short_df = _synthetic_multi_frame(n_codes=2, n_days=3)
    with pytest.raises(ValueError):
        build_multi_channel_signal(supply_df, short_df, flow_channels=("not_real",))


def test_walk_forward_multi_signal_eval_runs_end_to_end():
    supply_df, short_df = _synthetic_multi_frame(n_codes=8, n_days=25)
    splits, summary = walk_forward_multi_signal_eval(
        supply_df, short_df, horizons=(3,), min_formation=8
    )
    assert summary["n_splits"] > 0
    assert 0.0 <= summary["frac_positive"] <= 1.0
    assert set(splits.columns) == {
        "base_date", "eval_date", "horizon", "n", "spearman", "sign",
    }
