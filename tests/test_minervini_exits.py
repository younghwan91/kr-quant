"""Minervini sell rules — pure exit functions on synthetic paths."""

from __future__ import annotations

import numpy as np

from kr_quant.strategies.minervini_exits import (
    breakeven_plus_stop,
    climax_run,
    hard_stop,
    sell_half_level,
    staggered_stops,
    violations,
)


def test_hard_stop_and_max_guard():
    assert hard_stop(100.0) == 95.0                 # default 5%
    assert hard_stop(100.0, pct=0.15) == 90.0       # capped at 10%


def test_staggered_stops_thirds():
    assert staggered_stops(100.0) == [96.0, 94.0, 92.0]


def test_climax_run_detects_vertical_spike():
    spike = np.linspace(100, 130, 10)               # +30% over 10 bars ≥ 25%
    assert climax_run(spike) is True
    flat = np.full(20, 100.0)
    assert climax_run(flat) is False


def test_climax_run_detects_relentless_updays():
    # 15 consecutive up-days over the 15-day window (needs >window closes) but
    # small per-day step → total gain < 25%, so only the up-days branch fires.
    c = [100.0]
    for _ in range(15):
        c.append(c[-1] * 1.005)
    assert climax_run(np.array(c), run_pct=0.99) is True   # spike branch off, updays on


def test_violations_ma_break_on_volume():
    closes = np.concatenate([np.full(19, 100.0), [90.0]])   # last close well below MA20
    volumes = np.concatenate([np.full(19, 1000.0), [3000.0]])  # on a volume surge
    assert violations(closes, volumes) is True


def test_violations_three_lower_lows():
    closes = np.concatenate([np.full(19, 100.0), [99.0, 98.0, 97.0]])  # 3 declining
    volumes = np.full(len(closes), 1000.0)
    assert violations(closes, volumes) is True


def test_violations_normal_hold():
    closes = np.concatenate([np.full(19, 100.0), [101.0]])
    volumes = np.full(20, 1000.0)
    assert violations(closes, volumes) is False


def test_sell_half_is_two_r():
    assert sell_half_level(100.0, 95.0) == 110.0    # 2 × (100−95) above entry


def test_breakeven_plus_only_after_two_r():
    assert breakeven_plus_stop(100.0, 98.0, r_reached=True) == 100.0   # max(entry, ma50)
    assert breakeven_plus_stop(100.0, 103.0, r_reached=True) == 103.0  # ma50 above entry
    assert breakeven_plus_stop(100.0, 98.0, r_reached=False) is None
