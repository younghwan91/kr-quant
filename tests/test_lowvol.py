"""Low-vol backtest. Pure DataFrame in -> DataFrame out (reuses the PEAD engine)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.lowvol import (
    lowvol_backtest,
    lowvol_rank_ic,
    select_lowvol_portfolio,
)


def _synthetic(n_days=400, n_codes=40, seed=0):
    """Low-vol codes drift up, high-vol codes drift down -> long-low/short-high earns.

    Code i gets volatility level rising with i and drift falling with i, so the
    calmest names (low i) both have low realized vol and positive drift: a book
    long the low-vol decile and short the high-vol decile should be net positive.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    codes = [f"{i:06d}" for i in range(n_codes)]
    rows = []
    for i, c in enumerate(codes):
        frac = i / (n_codes - 1)          # 0 (calm) .. 1 (wild)
        sigma = 0.006 + 0.02 * frac       # low-vol names are genuinely calmer
        mu = 0.0010 * (0.5 - frac)        # calm names drift up, wild names down
        rets = mu + rng.normal(0, sigma, n_days)
        price = 1000 * np.cumprod(1 + rets)
        for d, p in zip(dates, price):
            rows.append({"code": c, "date": d, "close": p, "trade_value": 100000})
    return pd.DataFrame(rows), dates


def test_lowvol_long_short_is_net_positive():
    prices, _ = _synthetic()
    periods, summary = lowvol_backtest(
        prices, vol_window=40, horizon=21, adv_floor=0, cost_one_way=0.0,
        min_names=10, start_index=70, borrow_cost_annual=0.0)
    assert summary["n"] > 3
    assert summary["mean_net"] > 0   # long calm / short wild earns the drift gap
    assert summary["sharpe"] > 0


def test_cost_and_borrow_reduce_net():
    prices, _ = _synthetic()
    _, s0 = lowvol_backtest(prices, vol_window=40, horizon=21, adv_floor=0,
                            cost_one_way=0.0, min_names=10, start_index=70,
                            borrow_cost_annual=0.0)
    _, s1 = lowvol_backtest(prices, vol_window=40, horizon=21, adv_floor=0,
                            cost_one_way=0.0034, min_names=10, start_index=70,
                            borrow_cost_annual=0.03)
    assert s1["mean_net"] < s0["mean_net"]  # turnover cost + short borrow bite


def test_rank_ic_confirms_lowvol_predicts_forward_return():
    prices, _ = _synthetic()
    res = lowvol_rank_ic(prices, vol_window=40, horizon=21, adv_floor=0,
                         start_index=70, n_regimes=4)
    assert res["n_days"] > 20
    assert res["ic_mean"] > 0            # -vol rank predicts forward return
    assert res["frac_positive"] > 0.5


def test_select_portfolio_returns_long_low_vol_short_high_vol():
    prices, _ = _synthetic(n_codes=40)
    meta = pd.DataFrame({"code": [f"{i:06d}" for i in range(40)],
                         "name": [f"n{i}" for i in range(40)],
                         "market": ["KOSPI"] * 40})
    book = select_lowvol_portfolio(prices, meta, vol_window=40, adv_floor=0,
                                   n_deciles=10, min_names=10)
    assert set(book["side"]) == {"long", "short"}
    longs = book[book["side"] == "long"]
    shorts = book[book["side"] == "short"]
    # Longs are the calmest decile -> lower annualized vol than the shorts.
    assert longs["vol_ann"].mean() < shorts["vol_ann"].mean()
    # Equal weight within each leg, each leg sums to ~1.
    assert abs(longs["weight"].sum() - 1.0) < 1e-6
    assert abs(shorts["weight"].sum() - 1.0) < 1e-6


def test_select_portfolio_excludes_spac_reit_by_name():
    prices, _ = _synthetic(n_codes=40)
    codes = [f"{i:06d}" for i in range(40)]
    names = [f"n{i}" for i in range(40)]
    names[0] = "머스트스팩"  # calmest name is a SPAC -> must be dropped from longs
    meta = pd.DataFrame({"code": codes, "name": names, "market": ["KOSPI"] * 40})
    book = select_lowvol_portfolio(prices, meta, vol_window=40, adv_floor=0,
                                   n_deciles=10, min_names=10)
    assert "000000" not in set(book["code"])
