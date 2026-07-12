"""Short-covering signal (day-over-day drop in outstanding short balance)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.short_flow import add_short_covering_signal


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def test_shrinking_balance_reads_as_positive_covering():
    df = pd.DataFrame({
        "code": ["A"] * 4,
        "date": _dates(4),
        "short_balance": [1000, 800, 800, 400],
    })
    out = add_short_covering_signal(df)
    cov = out.set_index("date")["short_covering"]
    assert cov["20260102"] > 0          # 1000->800: balance shrank, covering
    assert cov["20260103"] == 0          # 800->800: no change
    assert cov["20260104"] > 0           # 800->400: covering again


def test_growing_balance_reads_as_negative_covering():
    df = pd.DataFrame({
        "code": ["A"] * 3,
        "date": _dates(3),
        "short_balance": [500, 600, 900],
    })
    out = add_short_covering_signal(df)
    cov = out.set_index("date")["short_covering"].dropna()
    assert (cov < 0).all()  # balance grew each day -> shorts added, not covered


def test_normalization_by_prior_balance():
    # 800->400 is a 50% covering of the prior balance, regardless of scale.
    df = pd.DataFrame({
        "code": ["A", "A", "B", "B"],
        "date": _dates(2) + _dates(2),
        "short_balance": [800, 400, 8_000_000, 4_000_000],
    })
    out = add_short_covering_signal(df)
    cov = out.dropna(subset=["short_covering"])
    assert np.allclose(cov["short_covering"].to_numpy(), 0.5)


def test_first_observation_per_code_is_nan_no_crash_on_zero_prior():
    df = pd.DataFrame({
        "code": ["A", "A", "B"],
        "date": _dates(2) + _dates(1),
        "short_balance": [0, 500, 100],  # A starts at 0 -> first ratio undefined either way
    })
    out = add_short_covering_signal(df)
    first_per_code = out.groupby("code", sort=False).head(1)
    assert first_per_code["short_covering"].isna().all()
    # A's second row: prior balance is 0 -> normalization guarded (NaN, not inf/ZeroDivisionError)
    a_second = out[out["code"] == "A"].iloc[1]
    assert pd.isna(a_second["short_covering"])
