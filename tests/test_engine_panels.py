"""Engine panels — pivot helpers, signal resolution, and session cache (Step 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.panels import (
    PanelCache,
    adv_panel,
    lookup_panel,
    panel_pivot,
    resolve_signal,
    yoy_panels,
)

# Source functions the engine copied from (equivalence pins). The original
# ``_lookup`` (Step 4) and ``_adv_panel`` (Step 5) bodies were migrated onto the
# engine and deleted; their equivalence is now pinned inline against the
# canonical transforms reproduced below.
from kr_quant.strategies.pead import _panel as _src_panel


def _src_adv_panel(prices: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """The pre-migration ``_adv_panel`` body (equivalence pin)."""
    tv = prices[["code", "date", "trade_value"]].copy()
    tv["trade_value"] = tv["trade_value"].abs()
    tv = tv.sort_values(["code", "date"])
    tv["adv"] = tv.groupby("code")["trade_value"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    return tv.dropna(subset=["adv"])[["code", "date", "adv"]].reset_index(drop=True)


def _prices() -> pd.DataFrame:
    dates = [f"2020-01-{d:02d}" for d in range(1, 8)]
    rows = []
    for code, base in (("A", 100.0), ("B", 200.0)):
        for k, d in enumerate(dates):
            # signed close (Kiwoom convention) — panel_pivot must abs it.
            rows.append({"code": code, "date": d, "close": -(base + k),
                         "trade_value": (k + 1) * 1000.0})
    return pd.DataFrame(rows)


def test_panel_pivot_abs_shape_and_matches_source():
    prices = _prices()
    panel = panel_pivot(prices, "close")
    assert panel.shape == (2, 7)
    assert (panel.to_numpy() >= 0).all()  # signed close abs'd
    assert panel.at["A", "2020-01-01"] == 100.0
    pd.testing.assert_frame_equal(panel, _src_panel(prices, "close"))


def test_lookup_panel_reindexes_and_matches_source():
    prices = _prices()
    piv = prices.assign(close=prices["close"].abs())
    codes, dates = ["A", "B", "Z"], ["2020-01-01", "2020-01-03", "2020-01-99"]
    arr = lookup_panel(piv, "close", codes, dates)
    assert arr.shape == (3, 3)
    assert arr[0, 0] == 100.0
    assert np.isnan(arr[2, 0])   # unknown code Z
    assert np.isnan(arr[0, 2])   # unknown date
    expected = (
        piv.pivot_table(index="code", columns="date", values="close", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )  # the canonical pivot+reindex the original _lookup performed
    np.testing.assert_array_equal(arr, expected)


def test_adv_panel_trailing_mean_and_matches_source():
    prices = _prices()
    out = adv_panel(prices, window=3)
    assert list(out.columns) == ["code", "date", "adv"]
    # first 2 rows/code dropped (min_periods=window); trade_value 1000,2000,3000 -> mean 2000
    a = out[(out["code"] == "A") & (out["date"] == "2020-01-03")]["adv"].iloc[0]
    assert abs(a - 2000.0) < 1e-9
    pd.testing.assert_frame_equal(out, _src_adv_panel(prices, window=3))


def test_yoy_panels_shape_and_age_fallback():
    codes, dates = ["A", "B"], ["2020-01-01", "2020-01-02"]
    ep = pd.DataFrame([
        {"code": "A", "date": "2020-01-01", "yoy": 0.5},
        {"code": "B", "date": "2020-01-02", "yoy": -0.2},
    ])
    yoy, age = yoy_panels(ep, codes, dates)
    assert yoy.shape == (2, 2)
    assert yoy[0, 0] == 0.5
    assert np.isnan(age).all()  # no age_days column -> all NaN


def test_resolve_signal_prefers_precomputed_panel():
    codes, dates = ["A", "B"], ["2020-01-01", "2020-01-02"]
    ep = pd.DataFrame([{"code": "A", "date": "2020-01-01", "yoy": 0.5}])
    sig_panel = pd.DataFrame([
        {"code": "A", "date": "2020-01-01", "signal": 0.9},
        {"code": "B", "date": "2020-01-02", "signal": 0.1},
    ])
    sig, age = resolve_signal(ep, sig_panel, codes, dates)
    assert sig[0, 0] == 0.9
    assert np.isnan(age).all()  # precomputed path has no freshness
    # None signal_panel -> falls back to the yoy path
    sig2, _ = resolve_signal(ep, None, codes, dates)
    assert sig2[0, 0] == 0.5


def test_panel_cache_hit_and_miss():
    cache = PanelCache(maxsize=8)
    prices = _prices()
    p1 = cache.panel_pivot(prices, "close")
    assert (cache.hits, cache.misses) == (0, 1)
    # identical content (distinct object) -> cache hit
    p2 = cache.panel_pivot(prices.copy(), "close")
    assert (cache.hits, cache.misses) == (1, 1)
    pd.testing.assert_frame_equal(p1, p2)
    # different value column -> cache miss
    cache.panel_pivot(prices, "trade_value")
    assert (cache.hits, cache.misses) == (1, 2)
    cache.clear()
    assert (cache.hits, cache.misses) == (0, 0)


def test_panel_cache_evicts_least_recently_used():
    cache = PanelCache(maxsize=1)
    prices = _prices()
    cache.panel_pivot(prices, "close")          # miss, stored
    cache.panel_pivot(prices, "trade_value")    # miss, evicts close
    cache.panel_pivot(prices, "close")          # miss again (was evicted)
    assert (cache.hits, cache.misses) == (0, 3)
