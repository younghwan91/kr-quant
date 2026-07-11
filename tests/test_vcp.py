"""VCP multi-contraction detector — synthetic bases in → diagnostics out."""

from __future__ import annotations

import numpy as np

from kr_quant.features.vcp import detect_vcp


def _ramp(anchors: list[float], steps: int = 5) -> np.ndarray:
    """Piecewise-linear series through ``anchors`` (steps points per leg)."""
    out = [anchors[0]]
    for a, b in zip(anchors[:-1], anchors[1:]):
        for k in range(1, steps + 1):
            out.append(a + (b - a) * k / steps)
    return np.asarray(out, float)


# −25% → −15% → −8% → −3% contractions off a ~100 pivot (ratios 0.6, 0.53, 0.375).
_VCP_ANCHORS = [100, 75, 100, 85, 100, 92, 100, 97]


def test_clean_vcp_detected():
    c = _ramp(_VCP_ANCHORS)
    vol = np.linspace(1000, 200, len(c))          # right side dries up
    r = detect_vcp(c, c, c, vol, asof_idx=len(c) - 1)
    assert r["is_vcp"] is True
    assert r["n_contractions"] >= 3
    assert abs(r["pivot"] - 100.0) < 1.0          # pivot = final contraction high
    assert r["tightness"] <= 0.10                 # last contraction tight
    assert r["volume_dryup"] < 0.5


def test_expanding_contractions_rejected():
    # Depths grow (−3 → −8 → −20) = not a VCP.
    c = _ramp([100, 97, 100, 92, 100, 80])
    vol = np.linspace(1000, 200, len(c))
    assert detect_vcp(c, c, c, vol, asof_idx=len(c) - 1)["is_vcp"] is False


def test_volume_not_drying_rejected():
    c = _ramp(_VCP_ANCHORS)
    vol = np.full(len(c), 1000.0)                  # no dry-up
    r = detect_vcp(c, c, c, vol, asof_idx=len(c) - 1)
    assert r["volume_dryup"] >= 0.5
    assert r["is_vcp"] is False


def test_final_contraction_too_deep_rejected():
    # Contractions shrink (40% → 22%, ratio 0.55 ≤ 0.6) but the final base is still
    # 22% deep (> 10%) → not tight enough → rejected by final_max_depth alone.
    c = _ramp([100, 60, 100, 78])
    vol = np.linspace(1000, 200, len(c))
    r = detect_vcp(c, c, c, vol, asof_idx=len(c) - 1)
    assert r["tightness"] > 0.10
    assert r["is_vcp"] is False


def test_asof_ignores_future_bars():
    c = _ramp(_VCP_ANCHORS)
    vol = np.linspace(1000, 200, len(c))
    asof = len(c) - 1
    base = detect_vcp(c, c, c, vol, asof_idx=asof)
    # Append a post-asof crash; detection as-of the same bar must be unchanged.
    c2 = np.concatenate([c, [50, 40, 30]])
    v2 = np.concatenate([vol, [9999, 9999, 9999]])
    after = detect_vcp(c2, c2, c2, v2, asof_idx=asof)
    assert after == base
