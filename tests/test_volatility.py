"""Realized-vol feature. Pure DataFrame in -> DataFrame out, lookahead-safe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.volatility import lowvol_signal_panel, realized_vol_panel


def _prices(n_days=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    rows = []
    for i, c in enumerate(["AAA", "BBB"]):
        sigma = 0.005 * (i + 1)  # BBB twice as volatile as AAA
        rets = rng.normal(0, sigma, n_days)
        price = 1000 * np.cumprod(1 + rets)
        for d, p in zip(dates, price):
            rows.append({"code": c, "date": d, "close": p})
    return pd.DataFrame(rows), dates


def test_vol_is_lookahead_safe_warmup_is_nan_free_output():
    prices, dates = _prices(n_days=80)
    vol = realized_vol_panel(prices, window=20)
    # A code's first `window` sessions are warm-up -> dropped; nothing before it.
    first_aaa = vol[vol["code"] == "AAA"]["date"].min()
    assert first_aaa == dates[20]  # 21st session (index 20) is the first defined vol
    assert vol["vol"].notna().all()
    assert (vol["vol"] > 0).all()


def test_higher_vol_code_has_higher_realized_vol():
    prices, _ = _prices(n_days=120)
    vol = realized_vol_panel(prices, window=30)
    mean_vol = vol.groupby("code")["vol"].mean()
    assert mean_vol["BBB"] > mean_vol["AAA"]  # BBB was built twice as volatile


def test_signal_is_negative_vol():
    prices, _ = _prices(n_days=80)
    vol = realized_vol_panel(prices, window=20).set_index(["code", "date"])["vol"]
    sig = lowvol_signal_panel(prices, window=20).set_index(["code", "date"])["signal"]
    aligned = sig.reindex(vol.index)
    assert np.allclose(aligned.to_numpy(), -vol.to_numpy())
