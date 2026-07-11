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
