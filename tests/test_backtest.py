"""Backtest logic: forward returns and score/return validation, no DB needed."""

from __future__ import annotations

import math

import pandas as pd

from kr_quant.strategies.backtest import backtest, forward_returns, spearman


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


def test_spearman_perfectly_monotonic():
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([10, 20, 30, 40])
    assert spearman(a, b) == 1.0
    assert spearman(a, -b) == -1.0


def test_forward_returns_uses_abs_close():
    df = pd.DataFrame(
        {
            "code": ["A", "A"],
            "date": ["20260101", "20260110"],
            "close": [-100, 110],  # signed close; magnitude is the price level
        }
    )
    fwd = forward_returns(df, "20260101", "20260110")
    assert math.isclose(fwd["A"], 0.10)


def test_backtest_splits_formation_and_holdout():
    # 14 days: formation = first 12, holdout = last (sideways accumulation pattern).
    closes = [100, 101, 99, 100, 102, 100, 101, 99, 100, 101, 100, 102, 100, 120]
    days = len(closes)
    df = _stock_frame(
        "000001", closes,
        foreign=[5000] * days, inst=[3000] * days, indiv=[-8000] * days,
    )
    merged, summary = backtest(df, formation_days=12, quantiles=5, min_days=10)
    assert summary["base_date"] == "20260512"   # 12th day
    assert summary["eval_date"] == "20260514"   # last day
    assert summary["n"] == 1
    # forward return from close 102 (day 12) to 120 (day 14).
    assert math.isclose(merged.iloc[0]["fwd_ret"], 120 / 102 - 1)


def test_backtest_raises_without_enough_days():
    df = _stock_frame("000001", [100, 101, 102], foreign=[1] * 3, inst=[1] * 3, indiv=[-1] * 3)
    try:
        backtest(df, formation_days=12)
    except ValueError as e:
        assert "거래일이 부족" in str(e)
    else:
        raise AssertionError("expected ValueError for insufficient days")
