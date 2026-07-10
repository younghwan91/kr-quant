"""Supply-flow feature signals. Pure DataFrame in -> DataFrame out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.supply_flow import (
    add_cross_sectional_rank,
    add_ewma_signal,
    add_normalized_ratios,
)


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def test_ewma_rises_after_step_increase():
    n = 30
    step = 15
    # foreign net-buy: flat small value, then a clear step-increase partway through.
    foreign = [1000] * step + [50000] * (n - step)
    df = pd.DataFrame(
        {
            "code": ["000001"] * n,
            "date": _dates(n),
            "foreign_": foreign,
            "market_cap": [1_000_000_000] * n,
        }
    )
    df["foreign_ratio"] = df["foreign_"] / df["market_cap"]

    result = add_ewma_signal(df, "foreign_ratio", halflife=7)

    pre = result.iloc[:step]["foreign_ratio_ewma"]
    post = result.iloc[step:]["foreign_ratio_ewma"]
    assert post.mean() > pre.mean()
    # Acceleration should be net positive right after the step.
    assert result.iloc[step : step + 5]["foreign_ratio_ewma_diff"].sum() > 0


def test_ewma_handles_gaps_without_raising_or_inf():
    # One code with several missing trading dates.
    dates = _dates(20)
    keep = [d for i, d in enumerate(dates) if i not in (3, 4, 9, 15)]
    n = len(keep)
    df = pd.DataFrame(
        {
            "code": ["000002"] * n,
            "date": keep,
            "foreign_": np.linspace(1000, 5000, n),
            "market_cap": [2_000_000_000] * n,
        }
    )
    df["foreign_ratio"] = df["foreign_"] / df["market_cap"]

    result = add_ewma_signal(df, "foreign_ratio", halflife=7)

    assert not np.isinf(result["foreign_ratio_ewma"]).any()
    assert not np.isinf(result["foreign_ratio_ewma_diff"]).any()
    # EWMA itself should never be NaN once data exists; only the first diff row is NaN.
    assert result["foreign_ratio_ewma"].notna().all()
    assert result["foreign_ratio_ewma_diff"].isna().sum() == 1


def test_ewma_independent_per_code_group():
    # Two codes interleaved in the frame; a step in one code must not leak into the other.
    n = 20
    step = 10
    code_a = pd.DataFrame(
        {
            "code": ["A"] * n,
            "date": _dates(n),
            "foreign_": [1000] * step + [50000] * (n - step),
            "market_cap": [1_000_000_000] * n,
        }
    )
    code_b = pd.DataFrame(
        {
            "code": ["B"] * n,
            "date": _dates(n),
            "foreign_": [1000] * n,
            "market_cap": [1_000_000_000] * n,
        }
    )
    df = pd.concat([code_a, code_b], ignore_index=True)
    df["foreign_ratio"] = df["foreign_"] / df["market_cap"]

    result = add_ewma_signal(df, "foreign_ratio", halflife=7)

    b_ewma = result[result["code"] == "B"]["foreign_ratio_ewma"]
    # Code B never stepped up, so its EWMA should stay essentially flat.
    assert b_ewma.max() - b_ewma.min() < 1e-9


def test_add_normalized_ratios_produces_expected_columns_and_values():
    df = pd.DataFrame(
        {
            "code": ["000001"],
            "date": ["20260101"],
            "individual": [-1000],
            "foreign_": [2000],
            "institution": [500],
            "fnnc_invt": [100],
            "insrnc": [50],
            "invtrt": [50],
            "bank": [0],
            "penfnd_etc": [200],
            "samo_fund": [50],
            "natn": [0],
            "etc_corp": [50],
            "market_cap": [1_000_000],
        }
    )
    result = add_normalized_ratios(df)
    assert result.loc[0, "individual_ratio"] == -1000 / 1_000_000
    # "foreign_" (trailing underscore, from INVESTOR_COLUMNS) -> "foreign__ratio".
    assert result.loc[0, "foreign__ratio"] == 2000 / 1_000_000
    assert result.loc[0, "institution_ratio"] == 500 / 1_000_000


def test_cross_sectional_rank_orders_codes_by_value_on_same_date():
    df = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D"],
            "date": ["20260101"] * 4,
            "foreign_ratio": [0.01, 0.05, -0.02, 0.03],
        }
    )
    result = add_cross_sectional_rank(df, "foreign_ratio")

    ranks = result.set_index("code")["foreign_ratio_rank"]
    # Highest ratio -> highest (percentile) rank.
    assert ranks["B"] > ranks["D"] > ranks["A"] > ranks["C"]
    assert ranks["B"] == ranks.max()
    assert ranks["C"] == ranks.min()


def test_cross_sectional_rank_is_per_date_not_rolling():
    df = pd.DataFrame(
        {
            "code": ["A", "B", "A", "B"],
            "date": ["20260101", "20260101", "20260102", "20260102"],
            "foreign_ratio": [0.01, 0.05, 0.05, 0.01],
        }
    )
    result = add_cross_sectional_rank(df, "foreign_ratio")
    ranks = result.set_index(["date", "code"])["foreign_ratio_rank"]
    # On day 1, B > A; on day 2 the ordering flips -- purely a same-day comparison.
    assert ranks[("20260101", "B")] > ranks[("20260101", "A")]
    assert ranks[("20260102", "A")] > ranks[("20260102", "B")]
