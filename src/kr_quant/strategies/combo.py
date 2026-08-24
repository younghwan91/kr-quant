"""Combining validated edges into one book via inverse-volatility weighting.

Given the monthly net-return series of two (or more) standalone strategies —
here PEAD and low-vol — allocate by naive risk parity (``w ∝ 1/σ``) and report
whether the blend beats each standalone on Sharpe and cuts the drawdown.

Provenance: ported from scalp-it combo note #32. The headline scalp-it result
(unified daily-close-marking engine, 50억, net) was **Sharpe 1.39 (PEAD) / 1.08
(low-vol) → 1.57 combined**, with **MDD −15.7% → −12.3%**. A key honest finding
travels with it: the dramatic inverse correlation that motivated the combo
(note #31's quarterly −0.73) did **not** reproduce under a single marking engine
(monthly net corr came out ≈ +0.17). So the real driver of the improvement is
inverse-vol weighting damping low-vol's large ~34% vol, not a −0.73 offset. See
``docs/combo-book.md``.

Pure Series in -> Series out (no DB), unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.metrics import ann_sharpe, max_drawdown

PPY = 12  # monthly series


def _common(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Align the strategy series on their common index (inner join)."""
    df = pd.DataFrame({k: v.astype(float) for k, v in series_map.items()})
    return df.dropna(how="any")


def inverse_vol_weights(series_map: dict[str, pd.Series]) -> dict[str, float]:
    """Risk-parity weights ``w_i ∝ 1/σ_i`` over the common index, summing to 1."""
    df = _common(series_map)
    inv = {k: (1.0 / df[k].std(ddof=1)) if df[k].std(ddof=1) > 0 else 0.0 for k in df.columns}
    tot = sum(inv.values())
    if tot <= 0:
        n = len(inv)
        return {k: 1.0 / n for k in inv}
    return {k: v / tot for k, v in inv.items()}


def combine_inverse_vol(
    series_map: dict[str, pd.Series],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    """Combine strategy net series by inverse-vol (or supplied) weights.

    Args:
        series_map: ``{name: monthly_net_series}``.
        weights: Explicit weights ``{name: w}``; ``None`` = inverse-vol.

    Returns:
        ``(combined_series, weights)`` — ``combined_series`` on the common index.
    """
    df = _common(series_map)
    w = weights or inverse_vol_weights(series_map)
    combined = sum(w[k] * df[k] for k in df.columns)
    return combined.rename("combined"), w


def expanding_inverse_vol(
    series_map: dict[str, pd.Series],
    *,
    min_periods: int = 24,
) -> pd.Series:
    """Lookahead-safe combo: each month's weights use only prior months' vols.

    The robustness form of :func:`combine_inverse_vol` — at month ``t`` the
    inverse-vol weights are estimated from data strictly before ``t`` (expanding,
    ``min_periods`` warm-up), so no future volatility leaks into the allocation.

    Returns:
        The expanding-weight combined Series (warm-up months dropped).
    """
    df = _common(series_map)
    cols = list(df.columns)
    out = pd.Series(index=df.index, dtype=float)
    vals = {k: df[k].to_numpy() for k in cols}
    for i in range(len(df)):
        if i < min_periods:
            continue
        stds = {k: vals[k][:i].std(ddof=1) for k in cols}
        if any(s <= 0 or not np.isfinite(s) for s in stds.values()):
            continue
        inv = {k: 1.0 / stds[k] for k in cols}
        tot = sum(inv.values())
        out.iloc[i] = sum((inv[k] / tot) * vals[k][i] for k in cols)
    return out.dropna()


def series_metrics(net: pd.Series, *, ppy: int = PPY) -> dict:
    """Annualized return / vol / Sharpe / MDD of a monthly net series.

    ``ann_ret``/``ann_vol`` follow note #32 (mean×ppy, std×√ppy); ``sharpe`` and
    ``mdd`` reuse the engine primitives so every strategy scores identically.
    """
    x = net.dropna().to_numpy(float)
    if len(x) < 6:
        return {"n": len(x), "ann_ret": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan"), "mdd": float("nan")}
    return {
        "n": len(x),
        "ann_ret": float(x.mean() * ppy),
        "ann_vol": float(x.std(ddof=1) * np.sqrt(ppy)),
        "sharpe": ann_sharpe(x, ppy),
        "mdd": max_drawdown(x),
    }
