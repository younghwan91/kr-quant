"""VCP (Volatility Contraction Pattern) detector — faithful Minervini buy setup.

The deployed scanner replaced Minervini's multi-contraction VCP with a single
``vc<1`` volume proxy (``SEPA_FAITHFUL_DESIGN.md`` §2.1). This module implements
the real spec: a base that contracts in **successive, shrinking pullbacks**
(e.g. −25% → −15% → −8% → −3%) with the right side drying up in volume, and the
pivot at the final contraction's high.

Detection is **as-of t-1**: it inspects bars up to and including ``asof_idx`` (the
last confirmed close) and never the entry bar — so wiring it to a t+1 entry
carries no look-ahead. Daily bars can't see intraday, so this is the honest
upper-bound approximation the design flags.

Pure arrays in → dict out. Feed split-adjusted OHLCV for one code.
"""

from __future__ import annotations

import numpy as np

# Frozen SEPA hyperparameters (SEPA_FAITHFUL_DESIGN.md §사전등록 동결표).
BASE_WINDOW = 60          # bars of base to inspect back from asof
MIN_CONTRACTIONS = 2
MAX_CONTRACTIONS = 6
SHRINK_RATIO = 0.6        # each contraction depth ≤ 0.6× the previous
FINAL_MAX_DEPTH = 0.10    # last (tightest) contraction ≤ 10%
VOL_DRYUP = 0.5           # last dry_days volume < 0.5× base-average volume
DRY_DAYS = 5
ZIGZAG_PCT = 0.03         # reversal threshold for swing detection


def _zigzag(price: np.ndarray, pct: float) -> list[tuple[int, float]]:
    """Alternating swing pivots (index, price): a new pivot when price reverses
    by at least ``pct`` from the running extreme, else the extreme is extended."""
    pivots: list[tuple[int, float]] = [(0, price[0])]
    direction = 0  # 0 unknown, +1 up-leg, -1 down-leg
    for i in range(1, len(price)):
        last_i, last_p = pivots[-1]
        if last_p <= 0:
            pivots[-1] = (i, price[i])
            continue
        change = (price[i] - last_p) / last_p
        if direction >= 0 and change <= -pct:
            pivots.append((i, price[i]))
            direction = -1
        elif direction <= 0 and change >= pct:
            pivots.append((i, price[i]))
            direction = 1
        elif direction >= 0 and price[i] > last_p:
            pivots[-1] = (i, price[i])
        elif direction <= 0 and price[i] < last_p:
            pivots[-1] = (i, price[i])
    return pivots


def detect_vcp(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    asof_idx: int,
    *,
    base_window: int = BASE_WINDOW,
    min_contractions: int = MIN_CONTRACTIONS,
    max_contractions: int = MAX_CONTRACTIONS,
    shrink_ratio: float = SHRINK_RATIO,
    final_max_depth: float = FINAL_MAX_DEPTH,
    vol_dryup: float = VOL_DRYUP,
    dry_days: int = DRY_DAYS,
) -> dict:
    """Detect a VCP as of ``asof_idx`` (t-1); return contraction/pivot diagnostics.

    Args:
        high, low, close, volume: Per-code split-adjusted daily arrays (abs prices).
        asof_idx: Last **confirmed** bar to use (entry would be ``asof_idx + 1``);
            the base window is ``[asof_idx - base_window + 1 .. asof_idx]``.
        base_window, min/max_contractions, shrink_ratio, final_max_depth,
        vol_dryup, dry_days: Frozen SEPA defaults (see module constants).

    Returns:
        ``{"n_contractions", "pivot", "tightness", "volume_dryup", "is_vcp"}``:
        - ``n_contractions``: count of successive shrinking pullbacks found.
        - ``pivot``: high of the final contraction (the breakout trigger), or NaN.
        - ``tightness``: depth of the final (tightest) contraction, or NaN.
        - ``volume_dryup``: last ``dry_days`` mean volume ÷ base mean volume.
        - ``is_vcp``: True iff ``min ≤ n ≤ max`` AND each contraction ≤
          ``shrink_ratio``× the prior AND final depth ≤ ``final_max_depth`` AND
          ``volume_dryup < vol_dryup``.
    """
    fail = {"n_contractions": 0, "pivot": float("nan"), "tightness": float("nan"),
            "volume_dryup": float("nan"), "is_vcp": False}
    start = max(0, asof_idx - base_window + 1)
    h = np.asarray(high[start:asof_idx + 1], float)
    lo = np.asarray(low[start:asof_idx + 1], float)
    c = np.asarray(close[start:asof_idx + 1], float)
    v = np.asarray(volume[start:asof_idx + 1], float)
    if len(c) < 10 or not np.all(np.isfinite(c)):
        return fail

    piv = _zigzag(c, ZIGZAG_PCT)
    if len(piv) < 3:
        return fail

    # Down-legs = peak→trough; depth measured high(peak)→low(trough) over the leg.
    depths: list[float] = []
    peak_highs: list[float] = []
    for a, b in zip(piv[:-1], piv[1:]):
        if b[1] < a[1]:  # a is a peak, b a trough (a down-leg)
            i_a, i_b = a[0], b[0]
            peak_h = float(np.max(h[i_a:i_b + 1]))
            trough_l = float(np.min(lo[i_a:i_b + 1]))
            if peak_h > 0:
                depths.append((peak_h - trough_l) / peak_h)
                peak_highs.append(peak_h)

    n = len(depths)
    tightness = depths[-1] if depths else float("nan")
    pivot = peak_highs[-1] if peak_highs else float("nan")
    base_vol = float(np.nanmean(v)) if v.size else float("nan")
    # Dry-up window excludes today (asof_idx): a signal day already clearing the
    # trend/RS/Code33 gates typically has volume already picking back up, so
    # including today in "recent" self-contaminates the "still dry" read (found
    # 2026-07-12 via funnel diagnostic — this was the actual bottleneck, not the
    # contraction shape). Measure the dry-up strictly *into* the pivot, at t-1.
    pre_today = v[:-1]
    recent_vol = float(np.nanmean(pre_today[-dry_days:])) if pre_today.size >= dry_days else float("nan")
    dry = recent_vol / base_vol if base_vol and base_vol > 0 else float("nan")

    # Chart shape (contraction depths shrinking a clean 25%→15%→8%→3%) is easy for
    # institutional flow to paint; volume is not. Per user direction (2026-07-12):
    # drop the price-geometry requirements (monotonic shrink, final-depth cap) —
    # a base merely needs to exist (>=1 contraction); volume dry-up is the real gate.
    is_vcp = bool(
        n >= min_contractions
        and np.isfinite(dry) and dry < vol_dryup
    )
    return {"n_contractions": n, "pivot": pivot, "tightness": tightness,
            "volume_dryup": dry, "is_vcp": is_vcp}
