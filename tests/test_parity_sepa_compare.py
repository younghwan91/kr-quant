"""Parity pins: old ``sepa_compare`` metric fns == new ``engine.metrics`` fns.

Step 1 of the backtest-engine migration (`.omc/plans/backtest-engine-plan.md`).
These assert **exact** float equality (atol=0, rtol=0 / direct ==) between the
private metric helpers in :mod:`kr_quant.strategies.sepa_compare` and their
:mod:`kr_quant.engine.metrics` equivalents, using the same synthetic fixtures as
``test_sepa_compare.py``. They must pass BEFORE the migration (engine holds an
identical copy) and stay green after (the old names become re-exports).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from kr_quant.engine import metrics as em
from kr_quant.strategies import sepa_compare as sc

_MONTHS = [f"20{y:02d}-{m:02d}" for y in range(18, 24) for m in range(1, 13)]  # 72 months


def _series(mean: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, vol, len(_MONTHS)), index=_MONTHS, name="ret")


def _exact(old: float, new: float) -> None:
    """Exact scalar equality, treating NaN==NaN as a match."""
    if isinstance(old, float) and math.isnan(old):
        assert isinstance(new, float) and math.isnan(new)
        return
    assert old == new


def test_parity_ann_sharpe():
    for seed in range(6):
        r = _series(0.01, 0.03, seed).to_numpy(float)
        _exact(sc._ann_sharpe(r), em.ann_sharpe(r))
    # degenerate / zero-vol / short inputs → NaN on both
    _exact(sc._ann_sharpe(np.full(12, 0.01)), em.ann_sharpe(np.full(12, 0.01)))
    _exact(sc._ann_sharpe(np.array([0.02])), em.ann_sharpe(np.array([0.02])))
    _exact(sc._ann_sharpe(np.array([])), em.ann_sharpe(np.array([])))
    # non-default ppy
    r = _series(0.01, 0.03, 1).to_numpy(float)
    _exact(sc._ann_sharpe(r, ppy=252), em.ann_sharpe(r, ppy=252))


def test_parity_cagr():
    for seed in range(6):
        r = _series(0.01, 0.03, seed).to_numpy(float)
        _exact(sc._cagr(r), em.cagr(r))
    _exact(sc._cagr(np.full(12, 0.01)), em.cagr(np.full(12, 0.01)))
    _exact(sc._cagr(np.array([])), em.cagr(np.array([])))
    r = _series(0.01, 0.03, 2).to_numpy(float)
    _exact(sc._cagr(r, ppy=4), em.cagr(r, ppy=4))


def test_parity_max_drawdown():
    for seed in range(6):
        r = _series(0.01, 0.03, seed).to_numpy(float)
        _exact(sc._max_drawdown(r), em.max_drawdown(r))
    _exact(sc._max_drawdown(np.array([0.1, -0.5, 0.1])), em.max_drawdown(np.array([0.1, -0.5, 0.1])))
    _exact(sc._max_drawdown(np.array([])), em.max_drawdown(np.array([])))


def _assert_bootstrap_equal(old: dict, new: dict) -> None:
    assert old.keys() == new.keys()
    _exact(old["d_sharpe_ci"][0], new["d_sharpe_ci"][0])
    _exact(old["d_sharpe_ci"][1], new["d_sharpe_ci"][1])
    _exact(old["d_cagr_ci"][0], new["d_cagr_ci"][0])
    _exact(old["d_cagr_ci"][1], new["d_cagr_ci"][1])
    _exact(old["prob_a_better_sharpe"], new["prob_a_better_sharpe"])
    assert old["n"] == new["n"]


def test_parity_paired_bootstrap_clear_winner():
    a = _series(0.015, 0.03, seed=1)
    b = _series(0.000, 0.03, seed=2)
    _assert_bootstrap_equal(
        sc.paired_bootstrap(a, b, n_boot=500, seed=0),
        em.paired_bootstrap(a, b, n_boot=500, seed=0),
    )


def test_parity_paired_bootstrap_tie():
    a = _series(0.005, 0.03, seed=3)
    b = _series(0.005, 0.03, seed=4)
    _assert_bootstrap_equal(
        sc.paired_bootstrap(a, b, n_boot=500, seed=0),
        em.paired_bootstrap(a, b, n_boot=500, seed=0),
    )


def test_parity_paired_bootstrap_degenerate_short_series():
    # n < block + 1 → early-return NaN branch on both.
    a = pd.Series([0.01, 0.02, 0.03], index=_MONTHS[:3])
    b = pd.Series([0.00, 0.01, 0.02], index=_MONTHS[:3])
    _assert_bootstrap_equal(
        sc.paired_bootstrap(a, b, block=6, n_boot=100, seed=0),
        em.paired_bootstrap(a, b, block=6, n_boot=100, seed=0),
    )


def test_parity_regime_buckets():
    r = pd.Series([0.02] * 18 + [-0.01] * 18 + [0.03] * 18 + [0.01] * 18, index=_MONTHS)
    old = sc.regime_buckets(r, n=4)
    new = em.regime_buckets(r, n=4)
    assert len(old) == len(new) == 4
    for o, n in zip(old, new):
        assert o["start"] == n["start"]
        _exact(o["mean"], n["mean"])
        assert o["positive"] == n["positive"]
    # too-short series → empty on both
    assert sc.regime_buckets(r.iloc[:2], n=4) == em.regime_buckets(r.iloc[:2], n=4) == []
