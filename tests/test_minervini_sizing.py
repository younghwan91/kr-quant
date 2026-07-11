"""Minervini sizing — concentration, equity risk, pilot, pyramiding."""

from __future__ import annotations

from kr_quant.strategies.minervini_sizing import (
    concentration_weights,
    equity_risk_size,
    fixed_risk_stop,
    pilot_then_full,
    pyramid_adds,
)


def test_concentration_weights_sum_to_one():
    signals = {c: v for c, v in zip("ABCDEFGH", [8, 7, 6, 5, 4, 3, 2, 1])}
    w = concentration_weights(signals)               # frozen 6 / 0.25 / 0.15
    assert len(w) == 6
    assert w["A"] == 0.25                             # top signal gets top weight
    assert all(w[c] == 0.15 for c in "BCDEF")
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert "G" not in w and "H" not in w             # only top 6 held


def test_equity_risk_size_hits_cap_at_frozen_defaults():
    assert equity_risk_size() == 0.25                # 1.25% / 5% = 25% = cap
    assert equity_risk_size(stop_pct=0.10) == 0.125  # wider stop → smaller position
    assert equity_risk_size(equity_risk=0.05, stop_pct=0.05) == 0.25  # capped


def test_pilot_then_full_stages():
    assert pilot_then_full(0.5) == 0.5               # not yet +1R → pilot half
    assert pilot_then_full(1.5) == 1.0               # proven → full


def test_pyramid_add_count_and_cap():
    assert pyramid_adds([2.5, 3.0, 1.0]) == 2        # two qualify (≥2R), within cap
    assert pyramid_adds([2.1, 2.2, 2.3]) == 2        # three qualify but capped at 2
    assert pyramid_adds([1.0, 1.9]) == 0             # none reach +2R


def test_fixed_risk_stop_holds_total_risk_constant():
    units = [(100.0, 10.0), (110.0, 10.0)]           # avg entry 105, 20 shares
    stop = fixed_risk_stop(units, initial_dollar_risk=50.0)
    assert stop == 102.5                             # 105 − 50/20
    total_size = sum(s for _, s in units)
    avg = sum(p * s for p, s in units) / total_size
    assert abs(total_size * (avg - stop) - 50.0) < 1e-9   # risk stayed 50
