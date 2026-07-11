"""SEPA arm-A harness — end-to-end composition smoke tests on synthetic data.

Proves the Phase 0/1 components compose into a runnable signal→entry→exit pipeline
(the Phase 2 skeleton), without needing the real DART earnings backfill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.rs_rating import rs_rating_panel
from kr_quant.strategies.minervini_sepa import sepa_entries, sepa_trades


def _prices(series: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long OHLCV frame from per-code close arrays (open/high/low derived)."""
    n = len(next(iter(series.values())))
    dates = pd.bdate_range("2019-01-01", periods=n).strftime("%Y-%m-%d")
    rows = []
    for code, close in series.items():
        for d, c in zip(dates, close):
            rows.append({"code": code, "date": d, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 1000.0})
    return pd.DataFrame(rows)


def _eligible_all(prices: pd.DataFrame) -> pd.DataFrame:
    e = prices[["code", "date"]].copy()
    e["eligible"] = True
    return e


def _ramp(anchors: list[float], steps: int) -> np.ndarray:
    out = [anchors[0]]
    for a, b in zip(anchors[:-1], anchors[1:]):
        for k in range(1, steps + 1):
            out.append(a + (b - a) * k / steps)
    return np.asarray(out)


def test_sepa_entries_base_count_filters_late_bases():
    # Long warm-up rise then several short (<20d) −15% base cycles = a late-stage
    # base at the final breakout; base_count should filter those signals out.
    warm = np.linspace(50, 100, 200)
    cyc = _ramp([100, 130, 110, 140, 118, 150, 128, 160, 136, 175], 8)  # 8-bar legs
    stair = np.concatenate([warm, cyc[1:]])
    prices = _prices({"STAIR": stair, "FLAT": np.full(len(stair), 100.0)})
    rs = rs_rating_panel(prices)
    kw = dict(code33=pd.DataFrame(columns=["code", "date", "is_code33"]),
              use_vcp=False, use_code33=False)
    on = sepa_entries(prices, _eligible_all(prices), rs, use_base_count=True, **kw)
    off = sepa_entries(prices, _eligible_all(prices), rs, use_base_count=False, **kw)
    on_set = set(zip(on["code"], on["date"]))
    off_set = set(zip(off["code"], off["date"]))
    assert on_set <= off_set          # base count only removes, never adds
    assert len(on) < len(off)         # late bases actually filtered


def test_sepa_trades_sell_half_and_breakeven():
    from kr_quant.strategies.minervini_sepa import sepa_trades
    # entry 100, stop 95, +2R target = 110. Price hits 111 (sells half at 110, raises
    # stop to break-even), then falls back through it → remainder ≈ break-even.
    close = [100.0] * 11 + [106.0, 111.0, 102.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]
    dates = pd.bdate_range("2020-01-01", periods=len(close)).strftime("%Y-%m-%d")
    prices = pd.DataFrame([{"code": "X", "date": d, "open": c, "high": c, "low": c,
                            "close": c, "volume": 1000.0} for d, c in zip(dates, close)])
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])

    half = sepa_trades(prices, entries, time_cap=20, sell_half=True, breakeven=True).iloc[0]
    assert half["reason"].endswith("+half")            # a half was banked at 2R
    assert abs(half["ret"] - 0.045) < 0.01             # ½·(+10%) + ½·(≈break-even)

    whole = sepa_trades(prices, entries, time_cap=20, sell_half=False).iloc[0]
    assert "+half" not in whole["reason"]              # no partial exit
    assert whole["ret"] < half["ret"]                  # rode it down instead of banking half


def test_sepa_trades_staggered_stops():
    from kr_quant.strategies.minervini_sepa import sepa_trades
    # Close stays flat (no violations), lows wick down to hit each tranche (96/94/92)
    # on successive bars → staggered loss = mean(−4,−6,−8) = −6% vs single −5%.
    lows = {12: 96.0, 13: 94.0, 14: 92.0}
    n = 18
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    prices = pd.DataFrame([{"code": "X", "date": dates[k], "open": 100.0, "high": 100.0,
                            "low": lows.get(k, 100.0), "close": 100.0, "volume": 1000.0}
                           for k in range(n)])
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])
    stag = sepa_trades(prices, entries, time_cap=30, staggered=True, tennis=False, sell_half=False).iloc[0]
    single = sepa_trades(prices, entries, time_cap=30, staggered=False, tennis=False, sell_half=False).iloc[0]
    assert stag["reason"] == "staggered"
    assert abs(stag["ret"] - (-0.06)) < 1e-6       # mean of −4/−6/−8
    assert abs(single["ret"] - (-0.05)) < 1e-6     # single 5% stop
    assert single["ret"] > stag["ret"]


def test_sepa_trades_tennis_ball_cull():
    from kr_quant.strategies.minervini_sepa import sepa_trades
    # Enters at 100 then drifts at 99 (above the stop, but never a new high) — a
    # broken egg → tennis-ball cull after tennis_window; without tennis it rides on.
    close = [100.0] * 11 + [99.0] * 15
    dates = pd.bdate_range("2020-01-01", periods=len(close)).strftime("%Y-%m-%d")
    prices = pd.DataFrame([{"code": "X", "date": d, "open": c, "high": c, "low": c,
                            "close": c, "volume": 1000.0} for d, c in zip(dates, close)])
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])
    culled = sepa_trades(prices, entries, time_cap=30, tennis=True, tennis_window=10).iloc[0]
    assert culled["reason"] == "tennis"
    held = sepa_trades(prices, entries, time_cap=30, tennis=False).iloc[0]
    assert held["reason"] == "time_cap"


def test_sepa_trades_pe_expansion_exit():
    from kr_quant.strategies.minervini_sepa import sepa_trades
    # Gentle rise (no stop/half/climax), but P/E triples from entry → valuation sell.
    close = [100.0] * 11 + [101.0, 102.0, 103.0, 103.0, 103.0]
    dates = pd.bdate_range("2020-01-01", periods=len(close)).strftime("%Y-%m-%d")
    prices = pd.DataFrame([{"code": "X", "date": d, "open": c, "high": c, "low": c,
                            "close": c, "volume": 1000.0} for d, c in zip(dates, close)])
    entries = pd.DataFrame([{"code": "X", "date": dates[9], "pivot": 100.0}])
    pe = pd.DataFrame([{"code": "X", "date": d, "pe": (10.0 if k < 12 else 30.0)}
                       for k, d in enumerate(dates)])   # P/E 10 → 30 (3×)
    tr = sepa_trades(prices, entries, time_cap=20, pe_panel=pe).iloc[0]
    assert tr["reason"] == "pe_expansion"
    # Without the P/E panel it would ride to the time cap instead.
    tr2 = sepa_trades(prices, entries, time_cap=20).iloc[0]
    assert tr2["reason"] == "time_cap"


def test_sepa_entries_gates_uptrend_only():
    # Strong uptrend passes the trend template + RS; a flat name fails both.
    n = 300
    up = 100 * (1.004 ** np.arange(n))
    flat = np.full(n, 100.0)
    prices = _prices({"UP": up, "FLAT": flat})
    rs = rs_rating_panel(prices)
    ent = sepa_entries(
        prices, _eligible_all(prices), rs,
        code33=pd.DataFrame(columns=["code", "date", "is_code33"]),
        use_vcp=False, use_code33=False,   # A₋VCP-style relaxation for a constructible smoke
    )
    assert list(ent.columns) == ["code", "date", "pivot", "score"]   # score = RS at signal
    assert not ent.empty
    assert set(ent["code"]) == {"UP"}          # only the trend-template + high-RS name
    assert (ent["pivot"] > 0).all()


def test_sepa_trades_stop_and_ride():
    # WIN: fills then rides up to the time cap (positive). LOSE: gaps below the
    # 5% stop the day after entry (exits ~ -5%).
    base = np.full(260, 100.0)
    win = np.concatenate([base, 100 * 1.002 ** np.arange(25)])   # steady rise post-signal
    lose = np.concatenate([base, [100, 93, 92, 92, 92]])         # drops through 95 stop
    lose = np.concatenate([lose, np.full(20, 92.0)])
    win = np.concatenate([win, np.full(len(lose) - len(win), win[-1])]) if len(win) < len(lose) else win
    prices = _prices({"WIN": win[:len(lose)], "LOSE": lose})
    entries = pd.DataFrame([
        {"code": "WIN", "date": prices["date"].unique()[259], "pivot": 100.0},
        {"code": "LOSE", "date": prices["date"].unique()[259], "pivot": 100.0},
    ])
    tr = sepa_trades(prices, entries, time_cap=20).set_index("code")
    assert tr.at["LOSE", "reason"] == "stop"
    assert tr.at["LOSE", "ret"] <= -0.05 + 1e-9     # stop (or gap below it)
    assert tr.at["WIN", "ret"] > 0                  # rode the uptrend
