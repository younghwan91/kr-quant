"""Parity pins: old ``pead.*`` == new ``engine.sim_crosssectional.*``, exactly.

Feeds identical synthetic inputs to the legacy strategy functions and the new
engine functions and asserts field-by-field EXACT float equality (no tolerance).
Guards the byte-identical numeric parity principle for the PEAD alpha whose
published Sharpe/t depend on exact reproduction. Runs both before the pead.py
delegation swap (proving the engine reproduces the old code) and after (proving
the thin wrapper changes nothing).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from kr_quant.engine.panels import panel_pivot, resolve_signal
from kr_quant.engine.sim_crosssectional import (
    rank_ic,
    rank_tilt_backtest,
    staggered_tranche_backtest,
)
from kr_quant.features.fundamentals import blend_rank, earnings_yoy_panel
from kr_quant.strategies.pead import pead_backtest, pead_rank_ic, staggered_backtest


def _synthetic(n_days=400, n_codes=60, drift=0.002, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    codes = [f"{i:06d}" for i in range(n_codes)]
    yoy_rank = {c: (i / (n_codes - 1)) for i, c in enumerate(codes)}
    prows, erows = [], []
    for c in codes:
        mu = drift * (yoy_rank[c] - 0.5)
        rets = mu + rng.normal(0, 0.01, n_days)
        price = 1000 * np.cumprod(1 + rets)
        for d, p in zip(dates, price):
            prows.append({"code": c, "date": d, "close": p, "trade_value": 100000})
        erows.append({"code": c, "avail_date": "20191201", "yoy": yoy_rank[c]})
    return pd.DataFrame(prows), pd.DataFrame(erows), dates


def _arrays(prices, earnings_panel, signal_panel=None):
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    sig, age = resolve_signal(earnings_panel, signal_panel, codes, dates)
    return C, V, sig, age, dates, codes


def _cap_array(cap_panel, codes, dates):
    return (
        cap_panel.pivot_table(index="code", columns="date", values="market_cap", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )


def _deep_exact_equal(a, b):
    """Recursive EXACT equality with NaN==NaN, for summary/result structures."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), (a, b)
        for k in a:
            _deep_exact_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            _deep_exact_equal(x, y)
    elif isinstance(a, float):
        assert isinstance(b, float)
        if math.isnan(a):
            assert math.isnan(b), (a, b)
        else:
            assert a == b, (a, b)  # EXACT, no tolerance
    else:
        assert a == b, (a, b)


def _assert_backtest_parity(old, new):
    periods_old, summary_old = old
    periods_new, summary_new = new
    pd.testing.assert_frame_equal(periods_old, periods_new, check_exact=True)
    _deep_exact_equal(summary_old, summary_new)


# --- rank_tilt_backtest / pead_backtest parity --------------------------------

def test_parity_pead_default_long_short():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_backtest(prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0, start_index=40)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=0,
                             cost_one_way=0.0, start_index=40, age=age)
    _assert_backtest_parity(old, new)


def test_parity_pead_with_cost():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_backtest(prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0023, start_index=40)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=0,
                             cost_one_way=0.0023, start_index=40, age=age)
    _assert_backtest_parity(old, new)


def test_parity_pead_long_only_excess():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_backtest(prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0,
                        start_index=40, long_only=True, borrow_cost_annual=0.05)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=0, cost_one_way=0.0,
                             start_index=40, long_only=True, borrow_cost_annual=0.05, age=age)
    _assert_backtest_parity(old, new)


def test_parity_pead_top_n_concentrated():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_backtest(prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0,
                        start_index=40, long_only=True, top_n=5)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=0, cost_one_way=0.0,
                             start_index=40, long_only=True, top_n=5, age=age)
    _assert_backtest_parity(old, new)


def test_parity_pead_borrow_cost():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_backtest(prices, panel, horizon=20, adv_floor=0, cost_one_way=0.0,
                        start_index=40, borrow_cost_annual=0.03)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=0, cost_one_way=0.0,
                             start_index=40, borrow_cost_annual=0.03, age=age)
    _assert_backtest_parity(old, new)


def test_parity_pead_fresh_days_empty():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_backtest(prices, panel, horizon=20, adv_floor=0, start_index=40, fresh_days=5)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=0,
                             start_index=40, fresh_days=5, age=age)
    _assert_backtest_parity(old, new)


def test_parity_pead_adv_floor_filters():
    prices, earnings, dates = _synthetic(n_codes=40)
    prices.loc[prices["code"] >= "000020", "trade_value"] = 10.0
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_backtest(prices, panel, horizon=20, adv_floor=1000, start_index=40, min_names=5)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=1000,
                             start_index=40, min_names=5, age=age)
    _assert_backtest_parity(old, new)


def test_parity_pead_signal_panel():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    blended = blend_rank([panel, panel], [0.5, 0.5], value_cols=["yoy", "yoy"])
    C, V, sig, age, dcols, _ = _arrays(prices, panel, signal_panel=blended)
    old = pead_backtest(prices, panel, signal_panel=blended, horizon=20,
                        adv_floor=0, cost_one_way=0.0, start_index=40)
    new = rank_tilt_backtest(C, V, sig, dcols, horizon=20, adv_floor=0,
                             cost_one_way=0.0, start_index=40, age=age)
    _assert_backtest_parity(old, new)


# --- staggered_tranche_backtest / staggered_backtest parity -------------------

def test_parity_staggered_basic():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, _, dcols, _ = _arrays(prices, panel)
    old = staggered_backtest(prices, panel, horizon=60, step=20, top_n=5,
                             adv_floor=0, start_index=40)
    new = staggered_tranche_backtest(C, V, sig, dcols, horizon=60, step=20, top_n=5,
                                     adv_floor=0, start_index=40)
    _assert_backtest_parity(old, new)


def test_parity_staggered_cap_tier():
    prices, earnings, dates = _synthetic(n_codes=60)
    panel = earnings_yoy_panel(earnings, dates)
    cap = pd.DataFrame([
        {"code": c, "date": d, "market_cap": (int(c) + 1) * 1e9}
        for c in prices["code"].unique() for d in dates
    ])
    C, V, sig, _, dcols, codes = _arrays(prices, panel)
    caparr = _cap_array(cap, codes, dcols)
    old = staggered_backtest(prices, panel, horizon=60, step=20, top_n=5, adv_floor=0,
                             start_index=40, cap_panel=cap, cap_rank=(20, 40), min_names=5)
    new = staggered_tranche_backtest(C, V, sig, dcols, horizon=60, step=20, top_n=5,
                                     adv_floor=0, start_index=40, cap_array=caparr,
                                     cap_rank=(20, 40), min_names=5)
    _assert_backtest_parity(old, new)


# --- rank_ic / pead_rank_ic parity --------------------------------------------

def test_parity_rank_ic():
    prices, earnings, dates = _synthetic()
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_rank_ic(prices, panel, horizon=20, adv_floor=0, start_index=40, n_regimes=4)
    new = rank_ic(C, V, sig, dcols, horizon=20, adv_floor=0, start_index=40,
                  n_regimes=4, age=age)
    _deep_exact_equal(old, new)


def test_parity_rank_ic_with_adv_floor():
    prices, earnings, dates = _synthetic(n_codes=40)
    prices.loc[prices["code"] >= "000020", "trade_value"] = 10.0
    panel = earnings_yoy_panel(earnings, dates)
    C, V, sig, age, dcols, _ = _arrays(prices, panel)
    old = pead_rank_ic(prices, panel, horizon=20, adv_floor=1000, start_index=40, n_regimes=4)
    new = rank_ic(C, V, sig, dcols, horizon=20, adv_floor=1000, start_index=40,
                  n_regimes=4, age=age)
    _deep_exact_equal(old, new)
