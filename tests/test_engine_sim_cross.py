"""Unit tests for ``engine.sim_crosssectional`` — cross-sectional rank-tilt sim.

Drives the engine functions directly on synthetic ``code × date`` numpy panels
(adapted from ``test_pead``'s fixture): higher-signal codes drift up, so a
long-high/short-low book earns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from kr_quant.engine.sim_crosssectional import (
    rank_ic,
    rank_tilt_backtest,
    staggered_tranche_backtest,
)


def _panels(n_days=400, n_codes=60, drift=0.002, seed=0, trade_value=100000.0):
    """Return (C, V, sig, dates): higher-rank codes get a positive drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    C = np.empty((n_codes, n_days))
    sig = np.empty((n_codes, n_days))
    for i in range(n_codes):
        rank = i / (n_codes - 1)
        mu = drift * (rank - 0.5)
        rets = mu + rng.normal(0, 0.01, n_days)
        C[i] = 1000 * np.cumprod(1 + rets)
        sig[i] = rank
    V = np.full((n_codes, n_days), float(trade_value))
    return C, V, sig, dates


def test_rank_tilt_net_positive_when_signal_predicts_drift():
    C, V, sig, dates = _panels()
    periods, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                          cost_one_way=0.0, start_index=40)
    assert summary["n"] > 3
    assert summary["mean_net"] > 0
    assert summary["sharpe"] > 0
    assert 0.0 <= summary["hit_rate"] <= 1.0


def test_rank_tilt_summary_schema():
    C, V, sig, dates = _panels()
    _, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0, start_index=40)
    for key in ("n", "sharpe", "t_stat", "mean_net", "hit_rate", "cum_net",
                "avg_turnover", "avg_win", "avg_loss", "payoff_ratio", "best", "worst"):
        assert key in summary


def test_rank_tilt_cost_reduces_net_below_gross():
    C, V, sig, dates = _panels()
    periods, _ = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                    cost_one_way=0.0023, start_index=40)
    assert (periods["net"] <= periods["gross"] + 1e-12).all()
    assert (periods["turnover"] >= 0).all()


def test_rank_tilt_dollar_neutral_periods_columns():
    C, V, sig, dates = _panels()
    periods, _ = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                    cost_one_way=0.0, start_index=40)
    assert list(periods.columns) == ["date", "gross", "turnover", "net"]
    # dates come from the label list, not integer indices
    assert periods["date"].iloc[0] in dates


def test_rank_tilt_long_only_excess_no_borrow_at_zero_cost():
    C, V, sig, dates = _panels()
    periods, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                          cost_one_way=0.0, start_index=40,
                                          long_only=True, borrow_cost_annual=0.05)
    assert summary["n"] > 3
    assert summary["mean_net"] > 0
    # long-only has no short book -> borrow must not bite: net == gross at 0 cost.
    assert (abs(periods["net"] - periods["gross"]) < 1e-12).all()


def test_rank_tilt_borrow_cost_reduces_long_short_net():
    C, V, sig, dates = _panels()
    _, s0 = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                               cost_one_way=0.0, start_index=40, borrow_cost_annual=0.0)
    _, s1 = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                               cost_one_way=0.0, start_index=40, borrow_cost_annual=0.03)
    assert s1["mean_net"] < s0["mean_net"]


def test_rank_tilt_top_n_concentrates_and_reports_payoff():
    C, V, sig, dates = _panels()
    _, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                    cost_one_way=0.0, start_index=40,
                                    long_only=True, top_n=5)
    assert summary["n"] > 3
    assert "payoff_ratio" in summary and "avg_win" in summary and "avg_loss" in summary
    assert summary["mean_net"] > 0


def test_rank_tilt_fresh_days_gate_empties_when_age_all_stale():
    C, V, sig, dates = _panels()
    age = np.full_like(sig, 999.0)  # every name is far too stale
    _, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                    start_index=40, fresh_days=5, age=age)
    assert summary["n"] == 0


def test_rank_tilt_fresh_days_gate_keeps_fresh_names():
    C, V, sig, dates = _panels()
    age = np.zeros_like(sig)  # all names freshly filed
    _, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                    start_index=40, fresh_days=5, age=age)
    assert summary["n"] > 3


def test_rank_tilt_age_none_defaults_to_no_gate():
    C, V, sig, dates = _panels()
    # age=None with fresh_days=0 must trade the full universe (no gate).
    _, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                    start_index=40, age=None)
    assert summary["n"] > 3


def test_rank_tilt_adv_floor_excludes_illiquid_names():
    C, V, sig, dates = _panels(n_codes=40)
    V = V.copy()
    V[20:, :] = 10.0  # make half illiquid
    _, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=1000,
                                    start_index=40, min_names=5)
    assert summary["n"] > 0


def test_rank_tilt_min_names_skips_thin_universe():
    C, V, sig, dates = _panels(n_codes=40)
    # min_names above the whole universe -> nothing ever eligible.
    _, summary = rank_tilt_backtest(C, V, sig, dates, horizon=20, adv_floor=0,
                                    start_index=40, min_names=100)
    assert summary["n"] == 0


def test_staggered_runs_and_reports_turnover():
    C, V, sig, dates = _panels()
    periods, summary = staggered_tranche_backtest(C, V, sig, dates, horizon=60, step=20,
                                                  top_n=5, adv_floor=0, start_index=40)
    assert summary["n"] > 3
    assert "payoff_ratio" in summary
    assert summary["mean_net"] > 0
    # 60/20 = 3 tranches -> turnover 1/3
    assert abs(periods["turnover"].iloc[0] - 1.0 / 3) < 1e-9


def test_staggered_cap_tier_restricts_universe():
    C, V, sig, dates = _panels(n_codes=60)
    cap = np.tile(np.arange(1, 61, dtype=float)[:, None] * 1e9, (1, len(dates)))
    _, summary = staggered_tranche_backtest(C, V, sig, dates, horizon=60, step=20, top_n=5,
                                            adv_floor=0, start_index=40,
                                            cap_array=cap, cap_rank=(20, 40), min_names=5)
    assert summary["n"] > 3


def test_rank_ic_confirms_positive_predictive_signal():
    C, V, sig, dates = _panels()
    res = rank_ic(C, V, sig, dates, horizon=20, adv_floor=0, start_index=40, n_regimes=4)
    assert res["n_days"] > 20
    assert res["ic_mean"] > 0
    assert res["frac_positive"] > 0.5
    assert len(res["regimes"]) == 4
    assert all(r["ic_mean"] > 0 for r in res["regimes"])


def test_rank_ic_empty_when_universe_too_thin():
    C, V, sig, dates = _panels(n_codes=10)  # fewer than the 20-name IC floor
    res = rank_ic(C, V, sig, dates, horizon=20, adv_floor=0, start_index=40, n_regimes=4)
    assert res["n_days"] == 0
    assert math.isnan(res["ic_mean"])
    assert res["regimes"] == []
