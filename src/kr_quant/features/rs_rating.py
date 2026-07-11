"""IBD-style Relative Strength (RS) rating — the 8th Minervini trend-template gate.

The deployed Minervini scanner implements 7 of the trend template's 8 criteria but
omits **RS rating ≥ 70** (see ``SEPA_FAITHFUL_DESIGN.md`` §1.2h). This module adds
it, lookahead-safe, for the faithful SEPA reproduction (arm A).

RS raw = ``2·r63 + r126 + r189 + r252`` (IBD's front-weighted blend of 3/6/9/12-month
returns), then percentile-ranked across the eligible universe **on each date** and
scaled to 0–100. A name at RS 70 outperformed 70% of the universe on that blend.

Pure DataFrame in → DataFrame out (no DB, no network), consistent with the rest of
:mod:`kr_quant.features`. Feed split-adjusted prices; ``close`` is abs'd (Kiwoom
sign convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# IBD front-weighted quarters: the most recent 63-day return counts double.
RS_WEIGHTS = (2.0, 1.0, 1.0, 1.0)
RS_LOOKBACKS = (63, 126, 189, 252)


def rs_rating_panel(
    prices: pd.DataFrame,
    *,
    weights: tuple[float, ...] = RS_WEIGHTS,
    lookbacks: tuple[int, ...] = RS_LOOKBACKS,
) -> pd.DataFrame:
    """Lookahead-safe (code × date) IBD RS rating (0–100 cross-sectional percentile).

    Args:
        prices: Long rows with ``code``, ``date``, ``close`` (split-adjusted;
            abs'd internally for Kiwoom's signed close).
        weights: Blend weights per lookback (default IBD front-weighting).
        lookbacks: Trailing return windows in trading days (default 63/126/189/252).

    Returns:
        Long ``code``/``date``/``rs_rating`` DataFrame. ``rs_rating`` is the
        per-date percentile of the weighted trailing-return blend, ×100 (so
        ``≥ 70`` = top-30% momentum that day). A code with fewer than
        ``max(lookbacks)`` prior bars on a date is dropped (NaN → filtered),
        never look-ahead: each date uses only closes on/before it.
    """
    close = prices.pivot_table(index="code", columns="date", values="close", aggfunc="first").abs()
    codes = close.index
    dates = close.columns
    C = close.to_numpy(float)
    n_codes, n_dates = C.shape
    longest = max(lookbacks)

    rs = np.full((n_codes, n_dates), np.nan)
    for j in range(longest, n_dates):
        acc = np.zeros(n_codes)
        ok = np.ones(n_codes, dtype=bool)
        for w, lb in zip(weights, lookbacks):
            r = C[:, j] / C[:, j - lb] - 1.0  # trailing return, past closes only
            acc += w * r
            ok &= np.isfinite(r)
        raw = np.where(ok, acc, np.nan)
        m = np.isfinite(raw)
        if m.sum() >= 2:  # need a cross-section to rank against
            rs[m, j] = pd.Series(raw[m]).rank(pct=True).to_numpy() * 100.0

    out = (
        pd.DataFrame(rs, index=codes, columns=dates)
        .reset_index()
        .melt(id_vars="code", var_name="date", value_name="rs_rating")
        .dropna(subset=["rs_rating"])
        .reset_index(drop=True)
    )
    return out
