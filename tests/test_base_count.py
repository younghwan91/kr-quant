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
