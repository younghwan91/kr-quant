"""supply_wave.py: signal definition, lookahead guard, walk-forward evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from kr_quant.features.supply_flow import INVESTOR_TYPES
from kr_quant.strategies.supply_wave import (
    LookaheadError,
    assert_no_lookahead,
    build_supply_wave_signal,
    walk_forward_supply_wave_eval,
)


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def _synthetic_frame(n_days=25, flows=(100, 200, 300, 400, 500), growth=(0.0, 0.005, 0.01, 0.015, 0.02)):
    """Multi-code, multi-day frame with a KNOWN deterministic relationship:
    codes with a higher (constant) foreign net-buy flow also have a higher
    daily price growth rate, so the supply-wave signal (cross-sectional rank
    of EWMA'd foreign flow) should positively predict forward returns in
    essentially every walk-forward split.
    """
    codes = [chr(ord("A") + i) for i in range(len(flows))]
    dates = _dates(n_days)
    rows = []
    for code, flow, g in zip(codes, flows, growth):
        close = 100.0
        for d in dates:
            rows.append(
                {
                    "code": code,
                    "date": d,
                    "close": close,
                    **{col: 0 for col in INVESTOR_TYPES},
                    "foreign_": flow,
                    "market_cap": 1_000_000_000,
                }
            )
            close *= 1 + g
    return pd.DataFrame(rows)


def test_build_supply_wave_signal_ranks_codes_by_flow():
    df = _synthetic_frame(n_days=10)
    result = build_supply_wave_signal(df, investor_col="foreign_", halflife=3)

    assert "supply_wave_signal" in result.columns
    # On the last date, ranks should follow the (constant, so trivially
    # EWMA-converged) flow ordering: A(100) < B(200) < ... < E(500).
    last_date = sorted(result["date"].unique())[-1]
    snap = result[result["date"] == last_date].set_index("code")["supply_wave_signal"]
    assert snap["A"] < snap["B"] < snap["C"] < snap["D"] < snap["E"]
    assert snap["E"] == snap.max()
    assert snap["A"] == snap.min()


def test_build_supply_wave_signal_rejects_unknown_investor_col():
    df = _synthetic_frame(n_days=5)
    with pytest.raises(ValueError):
        build_supply_wave_signal(df, investor_col="not_a_real_column")


def test_assert_no_lookahead_passes_for_legitimately_ordered_timestamps():
    # Signal source data dated strictly before its trade-use date -> fine.
    df = pd.DataFrame(
        {
            "date": ["20260101", "20260102", "20260103"],
            "trade_date": ["20260104", "20260104", "20260104"],
        }
    )
    assert_no_lookahead(df, source_date_col="date", use_date_col="trade_date")  # no raise


def test_assert_no_lookahead_raises_on_future_shifted_source_data():
    # Deliberately shift the "source" data forward so it is on/after the date
    # the signal would be used to trade -- this must be caught, not silently
    # accepted.
    df = pd.DataFrame(
        {
            "date": ["20260101", "20260105", "20260103"],  # row 2 (index 1) is future-shifted
            "trade_date": ["20260104", "20260104", "20260104"],
        }
    )
    with pytest.raises(LookaheadError):
        assert_no_lookahead(df, source_date_col="date", use_date_col="trade_date")


def test_assert_no_lookahead_rejects_same_day_source_and_use():
    # Same-day source/use is treated as a violation too: a signal computed
    # from data dated T may only be used starting T+1, never on T itself.
    df = pd.DataFrame({"date": ["20260101"], "trade_date": ["20260101"]})
    with pytest.raises(LookaheadError):
        assert_no_lookahead(df, source_date_col="date", use_date_col="trade_date")


def test_assert_no_lookahead_noop_on_empty_frame():
    df = pd.DataFrame({"date": [], "trade_date": []})
    assert_no_lookahead(df, source_date_col="date", use_date_col="trade_date")  # no raise


def test_walk_forward_eval_detects_known_positive_relationship():
    # Deterministic synthetic data: higher flow -> higher forward return.
    # This proves the evaluation *logic* is correct, independent of whether
    # real market data has any signal at all.
    df = _synthetic_frame(n_days=25)

    splits, summary = walk_forward_supply_wave_eval(
        df,
        investor_col="foreign_",
        halflife=3,
        horizons=(3,),
        min_formation=10,
    )

    assert summary["n_splits"] > 0
    assert len(splits) == summary["n_splits"]
    assert set(splits["sign"].unique()) <= {"+", "-", "0"}
    # The synthetic relationship is strong and monotonic, so almost every
    # split should show a positive sign.
    assert summary["frac_positive"] >= 0.9


def test_walk_forward_eval_reports_no_hardcoded_threshold_failure():
    # Even a tiny/uninformative dataset should not raise -- the function
    # reports whatever fraction it computes (including NaN/0 splits), it
    # never enforces a pass/fail gate itself.
    df = _synthetic_frame(n_days=6, flows=(100,) * 5, growth=(0.0,) * 5)
    splits, summary = walk_forward_supply_wave_eval(
        df, investor_col="foreign_", halflife=3, horizons=(2,), min_formation=4
    )
    assert isinstance(summary["n_splits"], int)
    assert "frac_positive" in summary
