"""Engine metrics — unit + source-equivalence tests (Step 0 of the migration).

Each metric is checked against a known answer plus its edge cases (empty, all-NaN,
single element, zero-vol), and each is asserted byte-identical to the strategy
function it was copied from — the pinning guarantee that the extraction is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.metrics import (
    ann_sharpe,
    cagr,
    max_drawdown,
    newey_west_t,
    paired_bootstrap,
    quantile_summary,
    regime_buckets,
    spearman,
    summarize_periods,
)

# Source functions the engine copied from (equivalence pins).
from kr_quant.strategies.backtest import _quantile_summary as _src_quantile_summary
from kr_quant.strategies.backtest import spearman as _src_spearman
from kr_quant.strategies.pead import _newey_west_t as _src_newey_west_t
from kr_quant.strategies.pead import _summarize as _src_summarize
from kr_quant.strategies.sepa_compare import _ann_sharpe as _src_ann_sharpe
from kr_quant.strategies.sepa_compare import _cagr as _src_cagr
from kr_quant.strategies.sepa_compare import _max_drawdown as _src_max_drawdown
from kr_quant.strategies.sepa_compare import paired_bootstrap as _src_paired_bootstrap
from kr_quant.strategies.sepa_compare import regime_buckets as _src_regime_buckets

_MONTHS = [f"20{y:02d}-{m:02d}" for y in range(18, 24) for m in range(1, 13)]  # 72 months


def _series(mean: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, vol, len(_MONTHS)), index=_MONTHS, name="ret")


# --- ann_sharpe ----------------------------------------------------------------

def test_ann_sharpe_known_value():
    r = np.array([0.01, -0.01, 0.02, 0.00, 0.03])
    expected = r.mean() / r.std() * np.sqrt(12)
    assert abs(ann_sharpe(r) - expected) < 1e-12


def test_ann_sharpe_zero_vol_is_nan():
    assert np.isnan(ann_sharpe(np.full(12, 0.01)))


def test_ann_sharpe_empty_is_nan():
    assert np.isnan(ann_sharpe(np.array([])))


def test_ann_sharpe_single_element_is_nan():
    assert np.isnan(ann_sharpe(np.array([0.05])))


def test_ann_sharpe_drops_nan_and_honors_ppy():
    r = np.array([0.01, np.nan, -0.01, 0.02])
    assert abs(ann_sharpe(r, ppy=252) - ann_sharpe(np.array([0.01, -0.01, 0.02]), ppy=252)) < 1e-12


def test_ann_sharpe_matches_source():
    r = _series(0.01, 0.03, seed=7).to_numpy(float)
    assert ann_sharpe(r) == _src_ann_sharpe(r)


# --- cagr -----------------------------------------------------------------------

def test_cagr_known_value():
    r = np.full(12, 0.01)  # +1%/mo, 12 months
    assert abs(cagr(r) - (1.01 ** 12 - 1)) < 1e-9


def test_cagr_empty_is_nan():
    assert np.isnan(cagr(np.array([])))


def test_cagr_all_nan_is_nan():
    assert np.isnan(cagr(np.array([np.nan, np.nan])))


def test_cagr_single_element():
    # one +10% period annualizes to (1.1)^(12/1) - 1
    assert abs(cagr(np.array([0.10])) - (1.10 ** 12 - 1.0)) < 1e-9


def test_cagr_matches_source():
    r = _series(0.005, 0.02, seed=8).to_numpy(float)
    assert cagr(r) == _src_cagr(r)


# --- max_drawdown ---------------------------------------------------------------

def test_max_drawdown_known():
    # +10%, -50%, +10%: peak 1.1 then trough 0.55 -> dd = 0.55/1.1 - 1 = -0.5
    assert abs(max_drawdown(np.array([0.1, -0.5, 0.1])) - (-0.5)) < 1e-12


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown(np.array([0.01, 0.02, 0.03])) == 0.0


def test_max_drawdown_empty_is_nan():
    assert np.isnan(max_drawdown(np.array([])))


def test_max_drawdown_matches_source():
    r = _series(0.0, 0.05, seed=9).to_numpy(float)
    assert max_drawdown(r) == _src_max_drawdown(r)


# --- newey_west_t ---------------------------------------------------------------

def test_newey_west_t_zero_lag_matches_plain_t():
    x = np.array([0.02, 0.01, -0.01, 0.03, 0.00, 0.02])
    mu, t = newey_west_t(x, lag=0)
    se = x.std(ddof=0) / np.sqrt(len(x))  # var = (d@d)/n is the population var
    assert abs(mu - x.mean()) < 1e-12
    assert abs(t - x.mean() / se) < 1e-9


def test_newey_west_t_insufficient_returns_nan():
    mu, t = newey_west_t(np.array([0.01, 0.02]), lag=5)
    assert np.isnan(mu) and np.isnan(t)


def test_newey_west_t_drops_nan():
    x = np.array([0.02, np.nan, 0.01, -0.01, 0.03])
    m1 = newey_west_t(x, lag=1)
    m2 = newey_west_t(np.array([0.02, 0.01, -0.01, 0.03]), lag=1)
    assert m1 == m2


def test_newey_west_t_matches_source():
    rng = np.random.default_rng(3)
    x = rng.normal(0.01, 0.02, 100)
    assert newey_west_t(x, 10) == _src_newey_west_t(x, 10)


# --- summarize_periods ----------------------------------------------------------

def test_summarize_empty_returns_zero_n():
    s = summarize_periods(pd.DataFrame(columns=["net", "turnover"]), horizon=20)
    assert s["n"] == 0
    assert np.isnan(s["sharpe"])


def test_summarize_known_fields():
    periods = pd.DataFrame({"net": [0.02, -0.01, 0.03, 0.00], "turnover": [0.5, 0.5, 0.5, 0.5]})
    s = summarize_periods(periods, horizon=20)
    assert s["n"] == 4
    assert abs(s["mean_net"] - 0.01) < 1e-12
    assert abs(s["hit_rate"] - 0.5) < 1e-12
    assert abs(s["avg_turnover"] - 0.5) < 1e-12


def test_summarize_payoff_ratio():
    periods = pd.DataFrame({"net": [0.04, -0.02], "turnover": [1.0, 1.0]})
    s = summarize_periods(periods, horizon=20)
    assert abs(s["payoff_ratio"] - (0.04 / 0.02)) < 1e-12
    assert abs(s["best"] - 0.04) < 1e-12
    assert abs(s["worst"] - (-0.02)) < 1e-12


def test_summarize_matches_source():
    periods = pd.DataFrame({"net": [0.02, -0.01, 0.03, -0.02, 0.01], "turnover": [0.4] * 5})
    assert summarize_periods(periods, 40) == _src_summarize(periods, 40)


# --- spearman -------------------------------------------------------------------

def test_spearman_monotonic():
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([10, 20, 30, 40])
    assert spearman(a, b) == 1.0
    assert spearman(a, -b) == -1.0


def test_spearman_single_element_is_nan():
    assert np.isnan(spearman(pd.Series([1.0]), pd.Series([2.0])))


def test_spearman_matches_source():
    a = pd.Series([3.0, 1.0, 4.0, 1.5, 5.0])
    b = pd.Series([2.0, 7.0, 1.0, 8.0, 2.0])
    assert spearman(a, b) == _src_spearman(a, b)


# --- quantile_summary -----------------------------------------------------------

def _merged(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"score": rng.normal(0, 1, n), "fwd_ret": rng.normal(0, 0.05, n)})


def test_quantile_summary_buckets():
    out = quantile_summary(_merged(50), 5)
    assert list(out.columns) == ["quantile", "n", "mean_fwd", "hit_rate"]
    assert len(out) == 5
    assert out["n"].sum() == 50


def test_quantile_summary_too_few_returns_empty():
    out = quantile_summary(_merged(3), 5)
    assert out.empty
    assert list(out.columns) == ["quantile", "n", "mean_fwd", "hit_rate"]


def test_quantile_summary_matches_source():
    m = _merged(40)
    a = quantile_summary(m, 4)
    b = _src_quantile_summary(m, 4)
    pd.testing.assert_frame_equal(a, b)


# --- paired_bootstrap -----------------------------------------------------------

def test_paired_bootstrap_detects_clear_winner():
    strong = _series(0.015, 0.03, seed=1)
    weak = _series(0.000, 0.03, seed=2)
    res = paired_bootstrap(strong, weak, n_boot=500, seed=0)
    assert res["d_sharpe_ci"][0] > 0
    assert res["prob_a_better_sharpe"] > 0.9


def test_paired_bootstrap_ties_include_zero():
    a = _series(0.005, 0.03, seed=3)
    b = _series(0.005, 0.03, seed=4)
    lo, hi = paired_bootstrap(a, b, n_boot=500, seed=0)["d_sharpe_ci"]
    assert lo < 0 < hi


def test_paired_bootstrap_too_short_returns_nan():
    a = pd.Series([0.01, 0.02, 0.03])
    res = paired_bootstrap(a, a, block=6, n_boot=100, seed=0)
    assert np.isnan(res["d_sharpe_ci"][0]) and res["n"] == 3


def test_paired_bootstrap_matches_source():
    a = _series(0.012, 0.03, seed=11)
    b = _series(0.004, 0.03, seed=12)
    assert paired_bootstrap(a, b, n_boot=300, seed=0) == _src_paired_bootstrap(a, b, n_boot=300, seed=0)


# --- regime_buckets -------------------------------------------------------------

def test_regime_buckets_sign_count():
    r = pd.Series([0.02] * 18 + [-0.01] * 18 + [0.03] * 18 + [0.01] * 18, index=_MONTHS)
    regs = regime_buckets(r, n=4)
    assert len(regs) == 4
    assert sum(x["positive"] for x in regs) == 3


def test_regime_buckets_too_few_returns_empty():
    assert regime_buckets(pd.Series([0.01, 0.02]), n=4) == []


def test_regime_buckets_matches_source():
    r = _series(0.006, 0.02, seed=13)
    assert regime_buckets(r, n=4) == _src_regime_buckets(r, n=4)
