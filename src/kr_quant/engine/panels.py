"""Panel construction helpers + session-scoped caching for the backtest engine.

Long (``code``/``date``/value) frames pivot to ``code × date`` numpy panels the
same way everywhere so no experiment re-derives the abs/reindex conventions.
A content-keyed session cache lets a parameter sweep pivot the same DB-loaded
frame once instead of on every run.

Provenance (Step 0 of the backtest-engine migration — copied, signatures
preserved):
    panel_pivot    <- pead._panel / minervini_sepa._panel
    lookup_panel   <- minervini_sepa._lookup
    adv_panel      <- sepa_experiment._adv_panel
    yoy_panels     <- pead._yoy_panels
    resolve_signal <- pead._resolve_signal
    forward_returns <- backtest.forward_returns
    build_panels   <- sepa_experiment.build_panels (Step 5, + session cache)
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import pandas as pd


def panel_pivot(prices: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot a long price frame to a code × date panel (abs — close is signed)."""
    return prices.pivot_table(index="code", columns="date", values=value, aggfunc="first").abs()


def forward_returns(df: pd.DataFrame, base_date: str, eval_date: str) -> pd.Series:
    """Per-code return from ``base_date``'s close to ``eval_date``'s close.

    Kiwoom stores a signed close (the sign marks the day's direction), so we
    take the absolute value to recover the price level.
    """
    piv = df.pivot_table(index="code", columns="date", values="close", aggfunc="first").abs()
    return (piv[eval_date] / piv[base_date] - 1.0).rename("fwd_ret")


def lookup_panel(panel: pd.DataFrame, value: str, codes, dates) -> np.ndarray:
    """Reindex a long code/date/value panel to a codes×dates numpy array."""
    return (
        panel.pivot_table(index="code", columns="date", values=value, aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )


def adv_panel(prices: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Trailing ``window``-day average trade value → long code/date/adv (as-of)."""
    tv = prices[["code", "date", "trade_value"]].copy()
    tv["trade_value"] = tv["trade_value"].abs()
    tv = tv.sort_values(["code", "date"])
    tv["adv"] = tv.groupby("code")["trade_value"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    return tv.dropna(subset=["adv"])[["code", "date", "adv"]].reset_index(drop=True)


def yoy_panels(earnings_panel, codes, dates):
    """Pivot the long earnings panel to code×date ``yoy`` and ``age_days`` arrays."""
    yoy = (
        earnings_panel.pivot_table(index="code", columns="date", values="yoy", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )
    if "age_days" in earnings_panel.columns:
        age = (
            earnings_panel.pivot_table(index="code", columns="date", values="age_days", aggfunc="first")
            .reindex(index=codes, columns=dates).to_numpy(float)
        )
    else:
        age = np.full_like(yoy, np.nan)
    return yoy, age


def resolve_signal(earnings_panel, signal_panel, codes, dates):
    """Return ``(sig, age)`` code×date arrays from the YoY panel or a precomputed
    ``signal_panel`` (long ``code``/``date``/``signal`` — e.g. a PEAD+value blend
    from :func:`kr_quant.features.fundamentals.blend_rank`). Freshness (``age``)
    only applies to the raw YoY path; it is ``NaN`` for a precomputed signal.
    """
    if signal_panel is not None:
        sig = (
            signal_panel.pivot_table(index="code", columns="date", values="signal", aggfunc="first")
            .reindex(index=codes, columns=dates).to_numpy(float)
        )
        return sig, np.full_like(sig, np.nan)
    return yoy_panels(earnings_panel, codes, dates)


# --- Session-scoped LRU cache for panel construction ---------------------------
#
# DataFrames are unhashable, so functools.lru_cache can't wrap panel_pivot
# directly. This is a small content-keyed LRU: a parameter sweep that repeatedly
# pivots the same DB-loaded frame (same content) hits the cache instead of
# re-pivoting. Keyed on the value column + shape + a stable content digest so two
# equal-content-but-distinct frame objects share an entry.


def _panel_cache_key(prices: pd.DataFrame, value: str) -> tuple:
    sub = prices[["code", "date", value]]
    digest = hashlib.sha1(
        pd.util.hash_pandas_object(sub, index=False).to_numpy().tobytes()
    ).hexdigest()
    return (value, sub.shape, digest)


class PanelCache:
    """A tiny content-keyed LRU cache for :func:`panel_pivot` results."""

    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def panel_pivot(self, prices: pd.DataFrame, value: str) -> pd.DataFrame:
        key = _panel_cache_key(prices, value)
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        result = panel_pivot(prices, value)
        self._store[key] = result
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)  # evict least-recently-used
        return result

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


# Module-level session cache. Import and call ``PANEL_CACHE.panel_pivot(...)`` to
# share pivots across a sweep; call ``PANEL_CACHE.clear()`` between sessions.
PANEL_CACHE = PanelCache()


def cached_panel_pivot(prices: pd.DataFrame, value: str) -> pd.DataFrame:
    """Session-scoped cached wrapper over :func:`panel_pivot` (see :data:`PANEL_CACHE`)."""
    return PANEL_CACHE.panel_pivot(prices, value)


# --- Panel assembly (split-adjusted, point-in-time) + session cache ------------
#
# ``build_panels`` (migrated from ``sepa_experiment`` in Step 5) assembles every
# panel a Minervini-SEPA experiment needs. It depends on feature/strategy modules
# that themselves import this engine package, so those imports are made *lazily*
# inside the function to keep ``engine`` importable as a leaf. A content-keyed
# session cache lets a parameter sweep that reuses the same DB-loaded frames build
# the panels once instead of on every recipe run.


def _frame_digest(df: pd.DataFrame) -> str:
    return hashlib.sha1(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    ).hexdigest()


def _build_panels_key(prices, earnings, shares, adjust, large_cap_rank) -> tuple:
    return (
        _frame_digest(prices), prices.shape,
        _frame_digest(earnings), earnings.shape,
        _frame_digest(shares), shares.shape,
        bool(adjust), tuple(large_cap_rank),
    )


_BUILD_PANELS_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_BUILD_PANELS_MAXSIZE = 8


def build_panels(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    adjust: bool = True,
    large_cap_rank: tuple[int, int] = (0, 100),
    use_cache: bool = True,
) -> dict:
    """Assemble every panel the SEPA arms need (split-adjusted, point-in-time).

    Args:
        prices: Long ``code``/``date``/``open``/``high``/``low``/``close``/
            ``trade_value``/``volume``.
        earnings: DART rows for :func:`code33_panel` (code, avail_date, netinc,
            netinc_prior, revenue, revenue_prior, op_income, op_income_prior).
        shares: ``code``/``date``/shares-outstanding for market cap.
        adjust: Apply corporate-action back-adjustment (both strategy and bench).
        large_cap_rank: Cap-rank tier for the B-shell (large-cap) universe.
        use_cache: Reuse a cached result for identical-content inputs (sweep win).

    Returns a dict of panels: ``prices``, ``cap``, ``smallmid``, ``largecap``,
    ``rs``, ``code33``, ``pe``.
    """
    if use_cache:
        key = _build_panels_key(prices, earnings, shares, adjust, large_cap_rank)
        cached = _BUILD_PANELS_CACHE.get(key)
        if cached is not None:
            _BUILD_PANELS_CACHE.move_to_end(key)
            return cached

    # Lazy imports (keep ``engine`` a leaf — these modules import engine).
    from ..features.fundamentals import code33_panel, earnings_yield_panel
    from ..features.rs_rating import rs_rating_panel
    from ..features.universe import CAP_BAND, smallmid_universe
    from ..price_adjust import adjust_prices
    from ..strategies.pead import market_cap_panel

    # Normalize date to string so every downstream to_datetime yields the same
    # resolution — DB columns arrive at mixed datetime64 units and break merge_asof.
    prices = prices.copy()
    prices["date"] = prices["date"].astype(str)
    shares = shares.copy()
    shares["date"] = shares["date"].astype(str)
    if adjust:
        prices = adjust_prices(prices)
    dates = sorted(prices["date"].astype(str).unique())
    cap = market_cap_panel(prices, shares)
    adv = adv_panel(prices)
    annual = earnings[earnings["period"].astype(str).str.endswith("Q4")]
    ep = earnings_yield_panel(annual, cap)                             # code/date/ep (E/P)
    ep["pe"] = (1.0 / ep["ep"]).where(ep["ep"] > 0)                    # P/E for the sell rule
    result = {
        "prices": prices,
        "cap": cap,
        # Absolute cap band, not rank — rank-within-a-liquidity-filtered-universe
        # silently lands on large/mega caps, not small-mid (see universe.py note).
        "smallmid": smallmid_universe(cap, adv, cap_band=CAP_BAND),
        "largecap": smallmid_universe(cap, adv, cap_rank=large_cap_rank),  # B-shell
        "rs": rs_rating_panel(prices),
        "code33": code33_panel(earnings, dates),
        "pe": ep[["code", "date", "pe"]],
    }
    if use_cache:
        _BUILD_PANELS_CACHE[key] = result
        if len(_BUILD_PANELS_CACHE) > _BUILD_PANELS_MAXSIZE:
            _BUILD_PANELS_CACHE.popitem(last=False)
    return result
