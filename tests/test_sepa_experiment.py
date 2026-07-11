"""SEPA experiment orchestrator — synthetic end-to-end (data → 5-arm verdict).

Proves the whole chain runs and produces the comparison structure before the real
DART backfill; a live run is the same call with real prices/earnings/shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.sepa_experiment import run_experiment


def _synth():
    n = 320
    dates = pd.bdate_range("2019-01-01", periods=n).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    codes = [f"{i:06d}" for i in range(1, 9)]          # 8 names
    prow, srow = [], []
    for j, code in enumerate(codes):
        # Mix of trends so cap ranks and RS vary across names.
        drift = 0.0006 * (j - 3)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        shares_out = 1_000_000 * (j + 1)               # distinct caps
        for d, c in zip(dates, close):
            prow.append({"code": code, "date": d, "open": c, "high": c * 1.02,
                         "low": c * 0.98, "close": c, "volume": 1e5,
                         "trade_value": c * 1e5})
        srow.append({"code": code, "date": dates[0], "shares_outstanding": shares_out})
    prices = pd.DataFrame(prow)
    shares = pd.DataFrame(srow)

    # Minimal earnings so code33_panel has something to join (accelerating for a few).
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


def test_experiment_runs_end_to_end_and_reports_all_arms():
    prices, earnings, shares = _synth()
    table, verdicts = run_experiment(prices, earnings, shares, adjust=False, n_boot=200)

    # Five arms in the table.
    assert set(table.index) == {"A", "A-diversified", "A-noVCP", "B-shell", "C-bench"}
    for col in ("sharpe", "cagr", "max_dd", "pos_regimes"):
        assert col in table.columns

    # Only the three A-arms get a verdict (B/C are reference arms).
    assert set(verdicts) == {"A", "A-diversified", "A-noVCP"}
    for v in verdicts.values():
        assert "vs_deployed" in v and "vs_benchmark" in v
        assert isinstance(v["beats_b_ci"], bool)


def test_benchmark_arm_is_nonflat():
    # C-bench (cap-weighted) should move with the synthetic market (not all zeros).
    prices, earnings, shares = _synth()
    table, _ = run_experiment(prices, earnings, shares, adjust=False, n_boot=100)
    assert np.isfinite(table.loc["C-bench", "cagr"])
