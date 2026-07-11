"""Base counter — synthetic multi-base price paths in → stage/is_early out."""

from __future__ import annotations

import numpy as np

from kr_quant.features.base_count import base_count


def _ramp(anchors: list[float], steps: int = 5) -> np.ndarray:
    out = [anchors[0]]
    for a, b in zip(anchors[:-1], anchors[1:]):
        for k in range(1, steps + 1):
            out.append(a + (b - a) * k / steps)
    return np.asarray(out, float)


# Four base cycles: rise → ~15% dip → new high, repeated. Each dip = one base.
_ANCHORS = [100, 130, 110, 140, 118, 150, 128, 160, 136]


def test_first_breakout_is_early_stage_one():
    c = _ramp(_ANCHORS)
    # index 15 = first return to 140 (breakout of base 1): exactly one base so far.
    r = base_count(c, asof_idx=15)
    assert r["base_stage"] == 1
    assert r["is_early"] is True


def test_late_bases_not_early():
    c = _ramp(_ANCHORS)
    r = base_count(c, asof_idx=len(c) - 1)   # after all four dips
    assert r["base_stage"] == 4
    assert r["is_early"] is False


def test_asof_ignores_future_bars():
    c = _ramp(_ANCHORS)
    base = base_count(c, asof_idx=15)
    c2 = np.concatenate([c, [200, 60, 210]])   # future high+crash after asof
    assert base_count(c2, asof_idx=15) == base


def test_deep_decline_resets_the_base_count_clock():
    # Reproduces the real-data bug: 4 bases (stage=4, late), then a bear-market-
    # scale (≥25%) decline and recovery, then ONE new base. Without the reset the
    # unbounded-accumulation bug would report stage=5 (still "late"); the fix
    # should treat this as a fresh cycle — stage=1, early again.
    four_bases = _ramp(_ANCHORS)                        # ends at stage 4 (peak ~160)
    crash = _ramp([160, 110], steps=10)[1:]              # −31% bear decline
    recover_and_one_base = _ramp([110, 145, 122, 150], steps=10)[1:]  # one fresh base
    c = np.concatenate([four_bases, crash, recover_and_one_base])

    r = base_count(c, asof_idx=len(c) - 1)
    assert r["base_stage"] == 1          # clock reset by the deep decline, not 5
    assert r["is_early"] is True


def test_shallow_decline_does_not_trigger_reset():
    # A pullback well under the 25% deep-reset threshold must not reset the clock
    # (regression guard: only bear-market-scale declines should reset).
    c = _ramp(_ANCHORS)                                  # troughs are ~15-18% pullbacks
    r = base_count(c, asof_idx=len(c) - 1)
    assert r["base_stage"] == 4          # unaffected by the new deep-reset logic
