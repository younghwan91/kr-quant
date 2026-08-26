"""Inverse-ETF market hedge. Pure Series in -> Series out, lookahead-safe beta."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.hedge import (
    inverse_hedged_return,
    rolling_beta,
    synth_inverse_return,
    universe_market_return,
)


def _prices(n_days=30):
    dates = pd.bdate_range("2021-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    rows = []
    for c, base in (("AAA", 1000.0), ("BBB", 2000.0)):
        price = base * np.cumprod(1 + np.linspace(0.001, 0.002, n_days))
        for d, p in zip(dates, price):
            rows.append({"code": c, "date": d, "close": p, "trade_value": 100000})
    return pd.DataFrame(rows)


def test_universe_market_return_is_cross_sectional_mean():
    prices = _prices()
    m = universe_market_return(prices, adv_floor=0)
    assert isinstance(m, pd.Series)
    assert m.index.is_monotonic_increasing
    assert m.notna().all()


def test_synth_inverse_is_negative_leverage_minus_fee():
    market = pd.Series([0.01, -0.02, 0.0], index=["d1", "d2", "d3"])
    inv = synth_inverse_return(market, leverage=1.0, fee_per_period=0.001)
    assert np.allclose(inv.to_numpy(), [-0.011, 0.019, -0.001])
    inv2 = synth_inverse_return(market, leverage=2.0, fee_per_period=0.0)
    assert np.allclose(inv2.to_numpy(), [-0.02, 0.04, 0.0])


def test_rolling_beta_is_lagged_no_lookahead():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 200))
    long = 1.5 * m + rng.normal(0, 0.002, 200)  # true beta ~1.5
    beta = rolling_beta(long, m, min_obs=24)
    # Warm-up NaN, then converges near the true beta; shifted so it's knowable.
    assert beta.iloc[:24].isna().all()
    assert abs(beta.dropna().iloc[-1] - 1.5) < 0.3


def test_beta_one_hedge_removes_market_and_cuts_variance():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0.0005, 0.012, 300))
    long = 1.0 * m + rng.normal(0.0003, 0.003, 300)  # beta 1 + idiosyncratic alpha
    hedged, beta_used = inverse_hedged_return(long, m, leverage=1.0, beta=1.0,
                                              fee_per_period=0.0)
    # A 1:1 inverse hedge cancels the market term -> only idiosyncratic left.
    assert hedged.var() < long.var()
    assert np.allclose(beta_used.to_numpy(), 1.0)
    # hedged == long + (beta/lev) * (-lev*m) == long - m
    assert np.allclose(hedged.to_numpy(), (long - m).to_numpy())


def test_estimated_beta_hedge_falls_back_unhedged_during_warmup():
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0, 0.01, 100))
    long = 0.8 * m + rng.normal(0, 0.002, 100)
    hedged, beta_used = inverse_hedged_return(long, m, beta=None, min_obs=24)
    # No beta yet in warm-up -> hedged equals the raw long return there.
    assert np.allclose(hedged.iloc[:24].to_numpy(), long.iloc[:24].to_numpy())
    assert beta_used.iloc[:24].isna().all()
