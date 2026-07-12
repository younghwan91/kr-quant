"""Shared EWMA/first-difference primitive (features/_common.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features._common import add_ewma_and_change


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def test_ewma_smooths_rising_series_monotonically():
    df = pd.DataFrame({
        "code": ["A"] * 6,
        "date": _dates(6),
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    out = add_ewma_and_change(df, "val", halflife=2.0, ewma_col="val_ewma", change_col="val_chg")
    ewma = out["val_ewma"].to_numpy()
    assert np.all(np.diff(ewma) > 0)          # smoothed level rises monotonically
    assert (out["val_chg"].dropna() > 0).all()  # first difference of a rising EWMA is positive


def test_ewma_independent_per_code_no_cross_leakage():
    # Code B's huge jump must not bleed into code A's smoothed level.
    df = pd.DataFrame({
        "code": ["A", "A", "B", "B"],
        "date": _dates(2) + _dates(2),
        "val": [1.0, 1.0, 100.0, 200.0],
    })
    out = add_ewma_and_change(df, "val", halflife=1.0, ewma_col="val_ewma", change_col="val_chg")
    a = out[out["code"] == "A"]["val_ewma"].to_numpy()
    assert np.allclose(a, 1.0)  # constant input for A -> constant EWMA regardless of B


def test_change_col_first_row_per_code_is_nan():
    df = pd.DataFrame({
        "code": ["A", "A", "B"],
        "date": _dates(2) + _dates(1),
        "val": [5.0, 6.0, 9.0],
    })
    out = add_ewma_and_change(df, "val", halflife=1.0, ewma_col="val_ewma", change_col="val_chg")
    # first observation of each code has no prior EWMA to diff against.
    first_per_code = out.groupby("code", sort=False).head(1)
    assert first_per_code["val_chg"].isna().all()
