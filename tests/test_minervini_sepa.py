"""Pivot buy-stop fill — scalar rules + panel next-session application."""

from __future__ import annotations

import math

import pandas as pd

from kr_quant.strategies.minervini_sepa import pivot_fill, pivot_fills


def test_fill_at_pivot_when_high_reaches_it():
    # open below pivot, high pierces it → fill exactly at the pivot.
    assert pivot_fill(90.0, 105.0, 100.0) == 100.0


def test_gap_up_fills_at_open():
    # open already above pivot (gap through) → fill at the open, not the pivot.
    assert pivot_fill(110.0, 115.0, 100.0) == 110.0


def test_no_fill_when_high_below_pivot():
    assert pivot_fill(90.0, 95.0, 100.0) is None


def test_nan_inputs_no_fill():
    assert pivot_fill(float("nan"), 105.0, 100.0) is None


def test_panel_fills_on_next_session_only():
    prices = pd.DataFrame([
        # A: signal on d1, next day d2 high 105 ≥ pivot 100, open 90 → fill 100.
        {"code": "A", "date": "d1", "open": 80, "high": 85},
        {"code": "A", "date": "d2", "open": 90, "high": 105},
        # B: next day d2 high 95 < pivot 100 → no fill.
        {"code": "B", "date": "d1", "open": 80, "high": 88},
        {"code": "B", "date": "d2", "open": 90, "high": 95},
    ])
    entries = pd.DataFrame([
        {"code": "A", "date": "d1", "pivot": 100.0},
        {"code": "B", "date": "d1", "pivot": 100.0},
    ])
    out = pivot_fills(entries, prices).set_index("code")
    assert out.at["A", "filled"] and out.at["A", "fill_price"] == 100.0
    assert out.at["A", "fill_date"] == "d2"          # filled on t+1, not t
    assert not out.at["B", "filled"] and math.isnan(out.at["B", "fill_price"])


def test_panel_no_next_session_is_unfilled():
    prices = pd.DataFrame([{"code": "A", "date": "d1", "open": 90, "high": 105}])
    entries = pd.DataFrame([{"code": "A", "date": "d1", "pivot": 100.0}])
    out = pivot_fills(entries, prices).set_index("code")
    assert not out.at["A", "filled"]                 # no d2 → cannot fill
