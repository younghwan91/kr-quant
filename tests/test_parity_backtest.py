"""Byte-parity pinning tests for the backtest.py -> engine migration (Step 2).

These pin the pre-migration behavior of ``backtest.spearman`` /
``backtest._quantile_summary`` / ``backtest.forward_returns`` as embedded
``_legacy_*`` reference functions (copied byte-for-byte from the original
``backtest.py``) and assert the engine implementations reproduce them exactly.

Embedding the legacy code (rather than referencing the live ``backtest`` module)
keeps the test a genuine regression guard: after migration ``backtest.spearman``
IS ``engine.metrics.spearman``, so comparing them would be a tautology. The
golden reference must be independent of the code under test.

Synthetic data reuses the ``_stock_frame`` fixture from ``test_backtest.py``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from kr_quant.engine.metrics import quantile_summary, spearman
from kr_quant.engine.panels import forward_returns
from kr_quant.strategies.backtest import backtest


# --- Legacy references (byte-identical copies of pre-migration backtest.py) -----


def _legacy_spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation (Pearson on ranks — no scipy dependency)."""
    if len(a) < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def _legacy_forward_returns(df: pd.DataFrame, base_date: str, eval_date: str) -> pd.Series:
    piv = df.pivot_table(index="code", columns="date", values="close", aggfunc="first").abs()
    return (piv[eval_date] / piv[base_date] - 1.0).rename("fwd_ret")


def _legacy_quantile_summary(merged: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """Mean forward return + hit rate per score quantile (Q1 = highest score)."""
    cols = ["quantile", "n", "mean_fwd", "hit_rate"]
    if len(merged) < quantiles:
        return pd.DataFrame(columns=cols)
    # rank=False so Q1 is the top score bucket; labels 1..quantiles.
    q = pd.qcut(merged["score"].rank(method="first", ascending=False), quantiles, labels=False) + 1
    out = (
        merged.assign(_q=q)
        .groupby("_q")
        .agg(n=("fwd_ret", "size"), mean_fwd=("fwd_ret", "mean"), hit_rate=("fwd_ret", lambda s: (s > 0).mean()))
        .reset_index()
        .rename(columns={"_q": "quantile"})
    )
    return out[cols]


# --- Synthetic fixtures (mirrors tests/test_backtest.py) ------------------------


def _stock_frame(code, closes, foreign, inst, indiv, vol=1_000_000):
    n = len(closes)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "name": [code] * n,
            "market": ["거래소"] * n,
            "sector": ["테스트"] * n,
            "date": [f"202605{d:02d}" for d in range(1, n + 1)],
            "close": closes,
            "acc_trde_qty": [vol] * n,
            "individual": indiv,
            "foreign_": foreign,
            "institution": inst,
            "penfnd_etc": [0] * n,
            "invtrt": [0] * n,
        }
    )


def _merged_fixture() -> pd.DataFrame:
    """A realistic score/fwd_ret table (enough rows for quantile bucketing)."""
    return pd.DataFrame(
        {
            "code": [f"{i:06d}" for i in range(10)],
            "score": [9.0, 8.5, 7.0, 6.5, 5.0, 4.5, 3.0, 2.5, 1.0, 0.5],
            "fwd_ret": [0.12, -0.03, 0.08, 0.01, -0.05, 0.02, -0.09, 0.04, -0.01, 0.06],
        }
    )


# --- spearman parity ------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        (pd.Series([1, 2, 3, 4]), pd.Series([10, 20, 30, 40])),
        (pd.Series([1, 2, 3, 4]), pd.Series([-10, -20, -30, -40])),
        (pd.Series([3, 1, 4, 1, 5, 9, 2, 6]), pd.Series([2, 7, 1, 8, 2, 8, 1, 8])),
        (pd.Series([1.0]), pd.Series([2.0])),  # len < 2 -> nan
    ],
)
def test_spearman_parity(a, b):
    old = _legacy_spearman(a, b)
    new = spearman(a, b)
    if math.isnan(old):
        assert math.isnan(new)
    else:
        assert new == old


# --- forward_returns parity -----------------------------------------------------


def test_forward_returns_parity():
    closes = [100, 101, 99, 100, 102, 100, 101, 99, 100, 101, 100, 102, 100, 120]
    days = len(closes)
    df = _stock_frame(
        "000001", closes,
        foreign=[5000] * days, inst=[3000] * days, indiv=[-8000] * days,
    )
    old = _legacy_forward_returns(df, "20260512", "20260514")
    new = forward_returns(df, "20260512", "20260514")
    pd.testing.assert_series_equal(new, old, check_exact=True)


def test_forward_returns_parity_signed_close():
    df = pd.DataFrame(
        {
            "code": ["A", "A", "B", "B"],
            "date": ["20260101", "20260110", "20260101", "20260110"],
            "close": [-100, 110, 200, -220],
        }
    )
    old = _legacy_forward_returns(df, "20260101", "20260110")
    new = forward_returns(df, "20260101", "20260110")
    pd.testing.assert_series_equal(new, old, check_exact=True)


# --- quantile_summary parity ----------------------------------------------------


@pytest.mark.parametrize("quantiles", [2, 5])
def test_quantile_summary_parity(quantiles):
    merged = _merged_fixture()
    old = _legacy_quantile_summary(merged, quantiles)
    new = quantile_summary(merged, quantiles)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def test_quantile_summary_parity_too_few_rows():
    merged = _merged_fixture().head(3)
    old = _legacy_quantile_summary(merged, 5)  # len < quantiles -> empty
    new = quantile_summary(merged, 5)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def test_quantile_summary_parity_from_backtest_flow():
    """Feed a merged frame produced by the real backtest() through both paths."""
    days = 14
    frames = []
    for code, phase, fmag in [("A", 0, 5000), ("B", 1, 4000), ("C", 0, 3000),
                              ("D", 1, 2000), ("E", 0, 6000), ("F", 1, 3500)]:
        closes = [100 + (2 if (d + phase) % 2 == 0 else -2) for d in range(days)]
        frames.append(_stock_frame(
            code, closes,
            foreign=[fmag] * days, inst=[fmag // 2] * days, indiv=[-fmag] * days,
        ))
    df = pd.concat(frames)
    merged, _ = backtest(df, formation_days=12, quantiles=2, min_days=6, max_range_pct=0.15)
    old = _legacy_quantile_summary(merged, 2)
    new = quantile_summary(merged, 2)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
