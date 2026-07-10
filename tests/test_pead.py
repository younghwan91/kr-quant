"""PEAD backtest. Pure DataFrame in -> DataFrame out (no DB)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.pead import pead_backtest, pead_rank_ic, staggered_backtest


def _synthetic(n_days=400, n_codes=60, drift=0.002, seed=0):
    """Panel where higher earnings-YoY codes drift up -> PEAD should be net+.

    Each code gets a fixed YoY rank; its daily return has a deterministic drift
    proportional to (rank - 0.5) plus noise, so a long-high/short-low book earns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    codes = [f"{i:06d}" for i in range(n_codes)]
    yoy_rank = {c: (i / (n_codes - 1)) for i, c in enumerate(codes)}  # 0..1
    prows, erows = [], []
    for c in codes:
        mu = drift * (yoy_rank[c] - 0.5)
        rets = mu + rng.normal(0, 0.01, n_days)
        price = 1000 * np.cumprod(1 + rets)
        for d, p in zip(dates, price):
            prows.append({"code": c, "date": d, "close": p, "trade_value": 100000})
        erows.append({"code": c, "avail_date": "20191201", "yoy": yoy_rank[c]})
    return pd.DataFrame(prows), pd.DataFrame(erows), dates


def test_pead_is_net_positive_when_earnings_predict_drift():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    periods, summary = pead_backtest(
        prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0, start_index=40,
    )
    assert summary["n"] > 3
    assert summary["mean_net"] > 0          # long-high/short-low earns the drift
    assert summary["sharpe"] > 0
    assert 0.0 <= summary["hit_rate"] <= 1.0


def test_cost_reduces_net_below_gross():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    periods, _ = pead_backtest(prices, panel, horizon=20, adv_floor=0,
                               cost_one_way=0.0023, start_index=40)
    # net = gross - turnover*cost, so net <= gross whenever any turnover occurs.
    assert (periods["net"] <= periods["gross"] + 1e-12).all()
    assert (periods["turnover"] >= 0).all()


def test_rank_ic_confirms_positive_predictive_signal():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    res = pead_rank_ic(prices, panel, horizon=20, adv_floor=0, start_index=40, n_regimes=4)
    assert res["n_days"] > 20
    assert res["ic_mean"] > 0            # YoY rank predicts forward return
    assert res["frac_positive"] > 0.5    # positive on most days
    assert len(res["regimes"]) == 4
    assert all(r["ic_mean"] > 0 for r in res["regimes"])  # persistent across regimes


def test_fresh_days_gate_limits_universe():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    # Earnings became available before the window -> age grows large; a tight
    # fresh_days gate should drop everything and yield no tradeable periods.
    panel = earnings_yoy_panel(earnings, dates)
    _, summary = pead_backtest(prices, panel, horizon=20, adv_floor=0,
                               start_index=40, fresh_days=5)
    assert summary["n"] == 0  # all filings older than 5 days -> nothing eligible


def test_backtest_accepts_precomputed_signal_panel():
    from kr_quant.features.fundamentals import blend_rank, earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    # Blend YoY with itself -> identical ranking -> same result as the raw path.
    blended = blend_rank([panel, panel], [0.5, 0.5], value_cols=["yoy", "yoy"])
    _, s_sig = pead_backtest(prices, panel, signal_panel=blended, horizon=20,
                             adv_floor=0, cost_one_way=0.0, start_index=40)
    _, s_raw = pead_backtest(prices, panel, horizon=20, adv_floor=0,
                             cost_one_way=0.0, start_index=40)
    assert s_sig["n"] == s_raw["n"]
    assert abs(s_sig["mean_net"] - s_raw["mean_net"]) < 1e-9  # blend of yoy⊕yoy == yoy


def test_long_only_reports_excess_and_charges_no_borrow():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    periods, summary = pead_backtest(prices, panel, horizon=20, adv_floor=0,
                                     cost_one_way=0.0, start_index=40,
                                     long_only=True, borrow_cost_annual=0.05)
    assert summary["n"] > 3
    assert summary["mean_net"] > 0  # long high-earnings tilt beats the universe
    # long-only has no short book, so borrow must not bite: net == gross at 0 cost.
    assert (abs(periods["net"] - periods["gross"]) < 1e-12).all()


def test_staggered_backtest_runs_and_reports_payoff():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    periods, summary = staggered_backtest(prices, panel, horizon=60, step=20,
                                          top_n=5, adv_floor=0, start_index=40)
    assert summary["n"] > 3
    assert "payoff_ratio" in summary
    assert summary["mean_net"] > 0  # captures the drift as positive excess
    # staggered turnover is 1/n_tranches (60/20 = 3 tranches -> ~0.33)
    assert abs(periods["turnover"].iloc[0] - 1.0 / 3) < 1e-9


def test_top_n_concentrates_and_reports_payoff_ratio():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    _, summary = pead_backtest(prices, panel, horizon=20, adv_floor=0,
                               cost_one_way=0.0, start_index=40,
                               long_only=True, top_n=5)
    assert summary["n"] > 3
    assert "payoff_ratio" in summary and "avg_win" in summary and "avg_loss" in summary
    # concentrated top-5 still captures the drift as positive expectancy
    assert summary["mean_net"] > 0


def test_borrow_cost_reduces_long_short_net():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    _, s0 = pead_backtest(prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0,
                          start_index=40, borrow_cost_annual=0.0)
    _, s1 = pead_backtest(prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0,
                          start_index=40, borrow_cost_annual=0.03)
    assert s1["mean_net"] < s0["mean_net"]  # borrow drag on the short book


def test_liquidity_floor_excludes_illiquid_names():
    from kr_quant.features.fundamentals import earnings_yoy_panel
    prices, earnings, dates = _synthetic(n_codes=40)
    # Make half the names illiquid (trade_value below the floor).
    prices.loc[prices["code"] >= "000020", "trade_value"] = 10.0
    panel = earnings_yoy_panel(earnings, dates)
    _, summary = pead_backtest(prices, panel, horizon=20, adv_floor=1000,
                               start_index=40, min_names=5)
    assert summary["n"] > 0  # still trades the liquid half
