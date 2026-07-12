"""Parity: the recipe-backed ``run_experiment`` / ``robustness_sweep`` reproduce the
pre-migration hand-rolled implementations exactly (Step 5 scaffolding).

The ``_legacy_*`` functions below are verbatim copies of the pre-Step-5 bodies
(build panels → build arms → compare_arms; sweep stops × concentrations). The
paired bootstrap is seeded (deterministic), so table + verdicts must match to the
float on identical synthetic input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.panels import build_panels
from kr_quant.strategies.minervini_sepa import sepa_entries, sepa_trades
from kr_quant.strategies.sepa_compare import (
    _ann_sharpe,
    _cagr,
    benchmark_returns,
    book_returns,
    compare_arms,
)
from kr_quant.strategies.sepa_experiment import (
    N_CONCENTRATED,
    N_DIVERSIFIED,
    robustness_sweep,
    run_experiment,
)


def _synth():
    """Same synthetic generator as tests/test_sepa_experiment.py."""
    n = 320
    dates = pd.bdate_range("2019-01-01", periods=n).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    codes = [f"{i:06d}" for i in range(1, 9)]
    prow, srow = [], []
    for j, code in enumerate(codes):
        drift = 0.0006 * (j - 3)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        shares_out = 1_000_000 * (j + 1)
        for d, c in zip(dates, close):
            prow.append({"code": code, "date": d, "open": c, "high": c * 1.02,
                         "low": c * 0.98, "close": c, "volume": 1e5,
                         "trade_value": c * 1e5})
        srow.append({"code": code, "date": dates[0], "shares_outstanding": shares_out})
    prices = pd.DataFrame(prow)
    shares = pd.DataFrame(srow)

    avails = ["2019-05-15", "2019-08-15", "2019-11-15", "2020-03-30", "2020-05-15"]
    erow = []
    for code in codes:
        for k, av in enumerate(avails):
            g = 1.0 + 0.1 * k
            erow.append({"code": code, "period": f"Q{k}", "avail_date": av,
                         "netinc": 100 * g, "netinc_prior": 100.0,
                         "revenue": 1000 * g, "revenue_prior": 1000.0,
                         "op_income": 100 * g * 1.1, "op_income_prior": 100.0})
    earnings = pd.DataFrame(erow)
    return prices, earnings, shares


def _legacy_run_experiment(prices, earnings, shares, *, adjust=True, **boot_kwargs):
    """Verbatim pre-Step-5 run_experiment body."""
    p = build_panels(prices, earnings, shares, adjust=adjust)
    px, rs, c33 = p["prices"], p["rs"], p["code33"]
    pe = p["pe"]
    ent_a = sepa_entries(px, p["smallmid"], rs, c33, use_vcp=True, use_code33=True)
    trades_a = sepa_trades(px, ent_a, pe_panel=pe)
    ent_avcp = sepa_entries(px, p["smallmid"], rs, c33, use_vcp=False, use_code33=True)
    ent_b = sepa_entries(px, p["largecap"], rs, c33, use_vcp=False, use_code33=False,
                         use_base_count=False, rs_min=0.0)
    arms = {
        "A": book_returns(px, trades_a, n_slots=N_CONCENTRATED, sized=True),
        "A-diversified": book_returns(px, trades_a, n_slots=N_DIVERSIFIED, sized=False),
        "A-noVCP": book_returns(px, sepa_trades(px, ent_avcp, pe_panel=pe), n_slots=N_CONCENTRATED, sized=True),
        "B-shell": book_returns(px, sepa_trades(px, ent_b), n_slots=N_DIVERSIFIED, sized=False),
        "C-bench": benchmark_returns(px, p["cap"]),
    }
    months = sorted(set().union(*[r.index for r in arms.values()]))
    idx = pd.Index(months, name="month")
    arms = {k: r.reindex(idx).fillna(0.0) for k, r in arms.items()}
    return compare_arms(arms, deployed="B-shell", benchmark="C-bench", **boot_kwargs)


def _legacy_robustness_sweep(prices, earnings, shares, *, adjust=True,
                             concentrations=(4, 6, 8), stops=(0.04, 0.05, 0.08),
                             use_vcp=True):
    """Verbatim pre-Step-5 robustness_sweep body."""
    p = build_panels(prices, earnings, shares, adjust=adjust)
    ent = sepa_entries(p["prices"], p["smallmid"], p["rs"], p["code33"],
                       use_vcp=use_vcp, use_code33=True)
    rows = []
    for stop in stops:
        trades = sepa_trades(p["prices"], ent, stop_pct=stop)
        for n in concentrations:
            r = book_returns(p["prices"], trades, n_slots=n, sized=True)
            rows.append({"concentration": n, "stop": stop,
                         "sharpe": _ann_sharpe(r.to_numpy(float)),
                         "cagr": _cagr(r.to_numpy(float)), "n_trades": len(trades)})
    return pd.DataFrame(rows)


def test_run_experiment_parity_table():
    prices, earnings, shares = _synth()
    old_t, _ = _legacy_run_experiment(prices, earnings, shares, adjust=False, n_boot=200)
    new_t, _ = run_experiment(prices, earnings, shares, adjust=False, n_boot=200)
    pd.testing.assert_frame_equal(old_t, new_t)


def test_run_experiment_parity_verdicts():
    prices, earnings, shares = _synth()
    _, old_v = _legacy_run_experiment(prices, earnings, shares, adjust=False, n_boot=200)
    _, new_v = run_experiment(prices, earnings, shares, adjust=False, n_boot=200)
    assert old_v.keys() == new_v.keys()
    for arm in old_v:
        assert old_v[arm]["beats_b_ci"] == new_v[arm]["beats_b_ci"]
        assert old_v[arm]["beats_c_ci"] == new_v[arm]["beats_c_ci"]
        for ref in ("vs_deployed", "vs_benchmark"):
            # assert_equal treats NaN==NaN (CIs are NaN when an arm has no trades).
            np.testing.assert_equal(old_v[arm][ref]["d_sharpe_ci"],
                                    new_v[arm][ref]["d_sharpe_ci"])
            np.testing.assert_equal(old_v[arm][ref]["d_cagr_ci"],
                                    new_v[arm][ref]["d_cagr_ci"])


def test_robustness_sweep_parity():
    prices, earnings, shares = _synth()
    old = _legacy_robustness_sweep(prices, earnings, shares, adjust=False, use_vcp=False)
    new = robustness_sweep(prices, earnings, shares, adjust=False, use_vcp=False)
    pd.testing.assert_frame_equal(
        old.reset_index(drop=True), new.reset_index(drop=True))


def test_robustness_sweep_parity_custom_grid():
    prices, earnings, shares = _synth()
    kw = dict(adjust=False, use_vcp=False, concentrations=(6,), stops=(0.05, 0.08))
    old = _legacy_robustness_sweep(prices, earnings, shares, **kw)
    new = robustness_sweep(prices, earnings, shares, **kw)
    pd.testing.assert_frame_equal(
        old.reset_index(drop=True), new.reset_index(drop=True))
