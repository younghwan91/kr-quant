"""Credit-balance (신용잔고) EWMA level/trend signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.credit_flow import add_credit_signal


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def test_add_credit_signal_tracks_rising_balance():
    df = pd.DataFrame(
        {
            "code": ["A"] * 6,
            "date": _dates(6),
            "balance_rt": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )
    out = add_credit_signal(df, halflife=2.0)
    # EWMA should be monotonically increasing as the raw series rises steadily.
    ewma = out["credit_balance_rt_ewma"].to_numpy()
    assert np.all(np.diff(ewma) > 0)
    # Day-over-day change should be positive throughout (crowding building).
    chg = out["credit_balance_rt_chg"].dropna()
    assert (chg > 0).all()


def test_add_credit_signal_independent_per_code():
    df = pd.DataFrame(
        {
            "code": ["A", "A", "B", "B"],
            "date": _dates(2) + _dates(2),
            "balance_rt": [1.0, 5.0, 10.0, 10.0],
        }
    )
    out = add_credit_signal(df, halflife=1.0)
    # Code B's flat series shouldn't be influenced by code A's jump.
    b_rows = out[out["code"] == "B"]
    assert np.isclose(b_rows.iloc[-1]["credit_balance_rt_chg"], 0.0, atol=1e-6)


def test_add_credit_signal_handles_gaps_without_crashing():
    df = pd.DataFrame(
        {
            "code": ["A", "A", "A"],
            "date": ["20260101", "20260105", "20260110"],  # gappy
            "balance_rt": [1.0, 2.0, 1.5],
        }
    )
    out = add_credit_signal(df)
    assert not out["credit_balance_rt_ewma"].isna().any()
