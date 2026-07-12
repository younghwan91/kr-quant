"""Parity: old `minervini_sepa.sepa_trades` (the `_walk` closure) vs the engine.

Builds the exact same panel inputs `sepa_trades` derives, drives them through
`engine.sim_eventdriven.trade_runner`, and asserts the resulting trades DataFrame
is identical field-by-field. Run before the `sepa_trades` body is swapped to
delegate (a genuine old-vs-new check); it remains a regression guard afterward
because it reconstructs the engine inputs independently of the strategy glue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.panels import panel_pivot
from kr_quant.engine.sim_eventdriven import trade_runner
from kr_quant.strategies.minervini_sepa import pivot_fills, sepa_trades


def _engine_trades(prices, entries, *, pe_panel=None, ma_exit_window=50, **kw):
    """Replicate sepa_trades' panel construction, then call the engine directly."""
    fills = pivot_fills(entries, prices)
    close = panel_pivot(prices, "close")
    codes, dates = list(close.index), list(close.columns)
    C = close.to_numpy(float)
    OPN = panel_pivot(prices, "open").reindex(index=codes, columns=dates).to_numpy(float)
    H = panel_pivot(prices, "high").reindex(index=codes, columns=dates).to_numpy(float)
    L = panel_pivot(prices, "low").reindex(index=codes, columns=dates).to_numpy(float)
    V = prices.pivot_table(index="code", columns="date", values="volume", aggfunc="first").reindex(
        index=codes, columns=dates).to_numpy(float)
    ma50 = pd.DataFrame(C.T).rolling(ma_exit_window, min_periods=1).mean().to_numpy()
    PE = (
        pe_panel.pivot_table(index="code", columns="date", values="pe", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
        if pe_panel is not None else None
    )
    return trade_runner(
        C, OPN, H, L, V, ma50, dates, codes, fills,
        ma_exit_window=ma_exit_window, pe_array=PE, **kw)


def _one_code(close, *, open_=None, high=None, low=None, code="X", start="2020-01-01"):
    n = len(close)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    rows = []
    for k, (d, c) in enumerate(zip(dates, close)):
        rows.append({
            "code": code, "date": d,
            "open": open_[k] if open_ is not None else c,
            "high": high[k] if high is not None else c,
            "low": low[k] if low is not None else c,
            "close": c, "volume": 1000.0,
        })
    return pd.DataFrame(rows), list(dates)


def _assert_parity(prices, entries, **kw):
    old = sepa_trades(prices, entries, **kw)
    new = _engine_trades(prices, entries, **{k: v for k, v in kw.items()})
    pd.testing.assert_frame_equal(old, new)
    return old


def test_parity_sell_half_and_breakeven():
    close = [100.0] * 11 + [106.0, 111.0, 102.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]
    prices, dates = _one_code(close)
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])
    _assert_parity(prices, entries, time_cap=20, sell_half=True, breakeven=True)
    _assert_parity(prices, entries, time_cap=20, sell_half=False)


def test_parity_staggered():
    lows = {12: 96.0, 13: 94.0, 14: 92.0}
    n = 18
    close = [100.0] * n
    low = [lows.get(k, 100.0) for k in range(n)]
    prices, dates = _one_code(close, open_=[100.0] * n, high=[100.0] * n, low=low)
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])
    _assert_parity(prices, entries, time_cap=30, staggered=True, tennis=False, sell_half=False)
    _assert_parity(prices, entries, time_cap=30, staggered=False, tennis=False, sell_half=False)


def test_parity_tennis_cull():
    close = [100.0] * 11 + [99.0] * 15
    prices, dates = _one_code(close)
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])
    _assert_parity(prices, entries, time_cap=30, tennis=True, tennis_window=10)
    _assert_parity(prices, entries, time_cap=30, tennis=False)


def test_parity_pe_expansion():
    close = [100.0] * 11 + [101.0, 102.0, 103.0, 103.0, 103.0]
    prices, dates = _one_code(close)
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])
    pe = pd.DataFrame([{"code": "X", "date": d, "pe": (10.0 if k < 12 else 30.0)}
                       for k, d in enumerate(dates)])
    old = sepa_trades(prices, entries, time_cap=20, pe_panel=pe)
    new = _engine_trades(prices, entries, time_cap=20, pe_panel=pe)
    pd.testing.assert_frame_equal(old, new)


def test_parity_stop_and_ride_multi_code():
    base = np.full(260, 100.0)
    win = np.concatenate([base, 100 * 1.002 ** np.arange(25)])
    lose = np.concatenate([base, [100, 93, 92, 92, 92]])
    lose = np.concatenate([lose, np.full(20, 92.0)])
    win = np.concatenate([win, np.full(len(lose) - len(win), win[-1])]) if len(win) < len(lose) else win
    n = len(lose)
    dates = pd.bdate_range("2019-01-01", periods=n).strftime("%Y-%m-%d")
    rows = []
    for code, series in {"WIN": win[:n], "LOSE": lose}.items():
        for d, c in zip(dates, series):
            rows.append({"code": code, "date": d, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 1000.0})
    prices = pd.DataFrame(rows)
    entries = pd.DataFrame([
        {"code": "WIN", "date": dates[259], "pivot": 100.0},
        {"code": "LOSE", "date": dates[259], "pivot": 100.0},
    ])
    _assert_parity(prices, entries, time_cap=20)
