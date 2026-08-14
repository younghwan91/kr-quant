"""Engine recipe API (Step 5) — declarative experiments produce the same output
as calling the underlying simulation functions directly.

Cross-sectional (PEAD-style) recipes route to ``rank_tilt_backtest``, pinned
against the direct call so the recipe layer adds no divergence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.panels import panel_pivot, resolve_signal
from kr_quant.engine.recipe import (
    ArmSpec,
    ExperimentConfig,
    run_recipe,
)
from kr_quant.engine.sim_crosssectional import rank_tilt_backtest


def _cross_synth():
    """Synthetic PEAD panel: prices (code/date/close/trade_value) + a YoY signal."""
    n = 260
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    rng = np.random.default_rng(7)
    codes = [f"{i:06d}" for i in range(40)]
    prow, erow = [], []
    for j, code in enumerate(codes):
        drift = 0.0004 * (j - 20)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
        for d, c in zip(dates, close):
            prow.append({"code": code, "date": d, "close": c, "trade_value": 1e5})
        # standing YoY signal (correlated with the drift so the book is non-trivial)
        for d in dates:
            erow.append({"code": code, "date": d, "yoy": drift * 1000})
    return pd.DataFrame(prow), pd.DataFrame(erow), list(dates)


def _direct_rank_tilt(prices, earnings, **kw):
    """Call rank_tilt_backtest exactly as the pead wrapper would (no recipe)."""
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    sig, age = resolve_signal(earnings, None, codes, dates)
    return rank_tilt_backtest(C, V, sig, dates, age=age, **kw)


def test_cross_sectional_recipe_matches_direct_call():
    prices, earnings, _ = _cross_synth()
    kw = dict(horizon=20, adv_floor=0.0, start_index=60, min_names=5, long_only=True)
    config = ExperimentConfig(
        experiment_type="cross_sectional",
        arms=[ArmSpec(name="pead", backtest_kwargs=kw)])
    table, summaries = run_recipe(config, prices, earnings)

    _, direct = _direct_rank_tilt(prices, earnings, **kw)
    # The recipe's per-arm summary is the byte-identical summarize_periods dict
    # (assert_equal treats NaN==NaN, unlike plain ==).
    np.testing.assert_equal(summaries["pead"], direct)
    assert table.loc["pead", "sharpe"] == direct["sharpe"]
    assert table.loc["pead", "n"] == direct["n"]


def test_cross_sectional_recipe_multi_arm_surprise_filter_sweep():
    """A PEAD surprise-filter sweep: several arms, one config, one run_recipe call."""
    prices, earnings, _ = _cross_synth()
    base = dict(horizon=20, adv_floor=0.0, start_index=60, min_names=5, long_only=True)
    arms = [ArmSpec(name=f"top{tn}", backtest_kwargs={**base, "top_n": tn})
            for tn in (5, 10, 0)]
    table, summaries = run_recipe(
        ExperimentConfig(experiment_type="cross_sectional", arms=arms), prices, earnings)

    assert list(table.index) == ["top5", "top10", "top0"]
    for tn in (5, 10, 0):
        _, direct = _direct_rank_tilt(prices, earnings, **{**base, "top_n": tn})
        np.testing.assert_equal(summaries[f"top{tn}"], direct)


def test_signal_panel_overrides_earnings_yoy():
    """A precomputed ``signal_panel`` is used instead of the YoY earnings panel."""
    prices, earnings, dates = _cross_synth()
    rng = np.random.default_rng(3)
    codes = sorted(prices["code"].unique())
    sig_panel = pd.DataFrame(
        [{"code": c, "date": d, "signal": rng.normal()} for c in codes for d in dates])
    kw = dict(horizon=20, adv_floor=0.0, start_index=60, min_names=5, long_only=True)
    config = ExperimentConfig(
        experiment_type="cross_sectional",
        arms=[ArmSpec(name="blend", backtest_kwargs=kw)],
        signal_panel=sig_panel)
    _, summaries = run_recipe(config, prices, earnings)

    # Direct call with the same precomputed signal (not the YoY path).
    close = panel_pivot(prices, "close")
    codes_l, dates_l = list(close.index), list(close.columns)
    C = close.to_numpy(float)
    V = panel_pivot(prices, "trade_value").reindex(index=codes_l, columns=dates_l).to_numpy(float)
    sig, age = resolve_signal(earnings, sig_panel, codes_l, dates_l)
    _, direct = rank_tilt_backtest(C, V, sig, dates_l, age=age, **kw)
    np.testing.assert_equal(summaries["blend"], direct)


def test_unknown_experiment_type_raises():
    prices, earnings, _ = _cross_synth()
    config = ExperimentConfig(experiment_type="nonsense", arms=[])
    try:
        run_recipe(config, prices, earnings)
    except ValueError as e:
        assert "nonsense" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown experiment_type")
