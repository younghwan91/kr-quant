"""SEPA arm comparison harness — eval primitives + paired bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.sepa_compare import (
    _ann_sharpe,
    _cagr,
    _max_drawdown,
    compare_arms,
    monthly_book_returns,
    paired_bootstrap,
    regime_buckets,
)

_MONTHS = [f"20{y:02d}-{m:02d}" for y in range(18, 24) for m in range(1, 13)]  # 72 months


def _series(mean: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, vol, len(_MONTHS)), index=_MONTHS, name="ret")


def test_eval_primitives():
    r = np.full(12, 0.01)                       # +1%/mo, no vol
    assert abs(_cagr(r) - (1.01 ** 12 - 1)) < 1e-9
    assert np.isnan(_ann_sharpe(r))             # zero vol → undefined
    assert _max_drawdown(np.array([0.1, -0.5, 0.1])) < 0    # a drawdown exists


def test_monthly_book_returns_books_at_exit_month():
    trades = pd.DataFrame([
        {"exit_date": "2018-03-15", "ret": 0.10},
        {"exit_date": "2018-03-28", "ret": 0.20},   # same month → averaged
        {"exit_date": "2018-05-01", "ret": -0.05},
    ])
    s = monthly_book_returns(trades, _MONTHS)
    assert abs(s["2018-03"] - 0.15) < 1e-9      # mean(0.10, 0.20)
    assert s["2018-05"] == -0.05
    assert s["2018-04"] == 0.0                  # no exits → flat


def test_regime_buckets_sign_count():
    r = pd.Series([0.02] * 18 + [-0.01] * 18 + [0.03] * 18 + [0.01] * 18, index=_MONTHS)
    regs = regime_buckets(r, n=4)
    assert len(regs) == 4
    assert sum(x["positive"] for x in regs) == 3   # bucket 2 negative


def test_paired_bootstrap_detects_clear_winner():
    strong = _series(0.015, 0.03, seed=1)       # higher mean, same vol
    weak = _series(0.000, 0.03, seed=2)
    res = paired_bootstrap(strong, weak, n_boot=500, seed=0)
    assert res["d_sharpe_ci"][0] > 0            # CI excludes 0 → A clearly beats B
    assert res["prob_a_better_sharpe"] > 0.9


def test_paired_bootstrap_ties_include_zero():
    a = _series(0.005, 0.03, seed=3)
    b = _series(0.005, 0.03, seed=4)            # same distribution
    res = paired_bootstrap(a, b, n_boot=500, seed=0)
    lo, hi = res["d_sharpe_ci"]
    assert lo < 0 < hi                          # CI spans 0 → not a win


def test_compare_arms_table_and_verdicts():
    arms = {
        "A": _series(0.015, 0.03, seed=1),
        "A_noconc": _series(0.010, 0.03, seed=5),
        "B": _series(0.000, 0.03, seed=2),
        "C": _series(0.006, 0.03, seed=6),
    }
    table, verdicts = compare_arms(arms, deployed="B", benchmark="C", n_boot=400)
    assert set(table.index) == {"A", "A_noconc", "B", "C"}
    assert "vs_deployed" in verdicts["A"] and "vs_benchmark" in verdicts["A"]
    assert verdicts["A"]["beats_b_ci"] is True   # A clearly beats the shell B
    assert "B" not in verdicts and "C" not in verdicts   # reference arms not judged
