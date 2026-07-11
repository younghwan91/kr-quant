"""IBD RS rating — pure panel in → percentile out (no DB/network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.rs_rating import rs_rating_panel


def _prices(series_by_code: dict[str, np.ndarray]) -> pd.DataFrame:
    dates = [f"2020-{1 + i // 21:02d}-{1 + i % 21:02d}" for i in range(len(next(iter(series_by_code.values()))))]
    rows = []
    for code, arr in series_by_code.items():
        for d, px in zip(dates, arr):
            rows.append({"code": code, "date": d, "close": px})
    return pd.DataFrame(rows)


def test_strong_uptrend_outranks_flat():
    n = 300
    strong = 100 * (1.003 ** np.arange(n))   # steady +0.3%/day
    stronger = 100 * (1.005 ** np.arange(n))  # steeper
    flat = np.full(n, 100.0)                  # sideways
    out = rs_rating_panel(_prices({"UP1": strong, "UP2": stronger, "FLAT": flat}))
    last = out[out["date"] == out["date"].max()].set_index("code")["rs_rating"]
    assert last["UP2"] > last["UP1"] > last["FLAT"]


def test_rating_within_0_100():
    n = 300
    rng = np.random.default_rng(0)
    series = {f"C{i}": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)) for i in range(8)}
    out = rs_rating_panel(_prices(series))
    assert out["rs_rating"].min() >= 0.0
    assert out["rs_rating"].max() <= 100.0


def test_insufficient_history_dropped():
    # Only 260 bars: dates before the 252nd have no full RS window → no rows there.
    n = 260
    series = {"A": 100 * (1.002 ** np.arange(n)), "B": 100 * (1.001 ** np.arange(n))}
    out = rs_rating_panel(_prices(series))
    valid_dates = sorted(out["date"].unique())
    assert len(valid_dates) == n - 252          # only the last (260-252) dates rank
    assert out["rs_rating"].notna().all()
