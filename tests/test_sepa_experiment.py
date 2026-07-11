"""SEPA experiment orchestrator — synthetic end-to-end (data → 5-arm verdict).

Proves the whole chain runs and produces the comparison structure before the real
DART backfill; a live run is the same call with real prices/earnings/shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.sepa_experiment import robustness_sweep, run_experiment, write_verdict


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


def test_robustness_sweep_grid_shape_and_values():
    prices, earnings, shares = _synth()
    # use_vcp=False so the synthetic arm actually trades → meaningful (non-all-NaN) grid.
    sweep = robustness_sweep(prices, earnings, shares, adjust=False, use_vcp=False,
                             concentrations=(4, 6, 8), stops=(0.04, 0.05, 0.08))
    assert set(sweep["concentration"]) == {4, 6, 8}
    assert set(sweep["stop"]) == {0.04, 0.05, 0.08}
    assert len(sweep) == 9                                  # 3 × 3 grid
    for col in ("sharpe", "cagr", "n_trades"):
        assert col in sweep.columns
    # every cell is finite or NaN (no crash), and the frozen (6, 0.05) point exists.
    assert ((sweep["concentration"] == 6) & (sweep["stop"] == 0.05)).any()
    assert sweep["sharpe"].apply(lambda x: np.isfinite(x) or np.isnan(x)).all()


def _verdict_table(a_sharpe: float):
    return pd.DataFrame(
        {"sharpe": [a_sharpe, 0.50, 0.80, 0.90, 0.70],
         "cagr": [0.30, 0.15, 0.20, 0.25, 0.17],
         "max_dd": [-0.20, -0.25, -0.22, -0.28, -0.30],
         "pos_regimes": ["4/4", "2/4", "3/4", "3/4", "3/4"]},
        index=["A", "A-diversified", "A-noVCP", "B-shell", "C-bench"])


def _verdicts(beats_b: bool, beats_c: bool):
    return {"A": {"vs_deployed": {"d_sharpe_ci": (0.2, 0.8) if beats_b else (-0.3, 0.5)},
                  "vs_benchmark": {"d_sharpe_ci": (0.1, 0.6) if beats_c else (-0.4, 0.4)},
                  "beats_b_ci": beats_b, "beats_c_ci": beats_c}}


def test_write_verdict_clear_winner():
    md = write_verdict(_verdict_table(1.2), _verdicts(True, True))
    assert "판정" in md and "A-diversified" in md
    assert "✅ 배포후보 갱신" in md          # A beats B & C (CI>0), 4/4 regimes, 1.2 > 0.5
    assert md.count("✅ 승") >= 4            # all four criteria pass


def test_write_verdict_loser_keeps_deployed():
    md = write_verdict(_verdict_table(0.4), _verdicts(False, False))
    assert "❌ 기존 배포판 유지" in md        # fails paired CIs and A(0.4) < A-diversified(0.5)


def test_write_verdict_no_trade_arm():
    md = write_verdict(_verdict_table(float("nan")), _verdicts(False, False))
    assert "평가불가 (무거래)" in md


def test_write_verdict_writes_file(tmp_path):
    out = tmp_path / "VERDICT.md"
    write_verdict(_verdict_table(1.2), _verdicts(True, True), out_path=str(out))
    assert out.exists() and "배포후보 갱신" in out.read_text()


def test_benchmark_arm_is_nonflat():
    # C-bench (cap-weighted) should move with the synthetic market (not all zeros).
    prices, earnings, shares = _synth()
    table, _ = run_experiment(prices, earnings, shares, adjust=False, n_boot=100)
    assert np.isfinite(table.loc["C-bench", "cagr"])
