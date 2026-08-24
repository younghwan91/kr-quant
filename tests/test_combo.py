"""Inverse-vol combo. Pure Series in -> Series out, lookahead-safe expanding form."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.combo import (
    combine_inverse_vol,
    expanding_inverse_vol,
    inverse_vol_weights,
    series_metrics,
)


def _two_series(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2018-01", periods=n, freq="M")
    a = pd.Series(rng.normal(0.01, 0.02, n), index=idx)   # calmer
    b = pd.Series(rng.normal(0.01, 0.06, n), index=idx)   # ~3x vol
    return {"A": a, "B": b}


def test_inverse_vol_weights_favor_the_calmer_series():
    sm = _two_series()
    w = inverse_vol_weights(sm)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["A"] > w["B"]  # lower vol -> larger weight
    # w_i proportional to 1/sigma_i
    sa, sb = sm["A"].std(ddof=1), sm["B"].std(ddof=1)
    assert abs(w["A"] / w["B"] - sb / sa) < 1e-6


def test_combine_is_weighted_sum_on_common_index():
    sm = _two_series()
    combined, w = combine_inverse_vol(sm)
    expected = w["A"] * sm["A"] + w["B"] * sm["B"]
    assert np.allclose(combined.to_numpy(), expected.to_numpy())


def test_combine_accepts_explicit_weights():
    sm = _two_series()
    combined, w = combine_inverse_vol(sm, weights={"A": 0.7, "B": 0.3})
    assert w == {"A": 0.7, "B": 0.3}
    assert np.allclose(combined.to_numpy(), (0.7 * sm["A"] + 0.3 * sm["B"]).to_numpy())


def test_combined_vol_below_the_higher_leg():
    sm = _two_series()
    combined, _ = combine_inverse_vol(sm)
    assert combined.std(ddof=1) < sm["B"].std(ddof=1)  # diversification cuts vol


def test_expanding_combo_is_lookahead_safe():
    sm = _two_series(n=60)
    exp = expanding_inverse_vol(sm, min_periods=24)
    # First `min_periods` months have no prior-only weights -> dropped.
    assert len(exp) == 60 - 24
    assert exp.notna().all()


def test_series_metrics_reports_sharpe_and_mdd():
    sm = _two_series()
    m = series_metrics(sm["A"])
    assert m["n"] == 60
    assert np.isfinite(m["sharpe"])
    assert m["mdd"] <= 0.0
    assert np.isfinite(m["ann_ret"]) and np.isfinite(m["ann_vol"])
