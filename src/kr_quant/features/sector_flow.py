"""Phase 1.5: cheap pandas-only pre-check of the cross-stock flow-propagation
hypothesis ("sector rotation") — no GNN, no heavy infra.

The project's real thesis is that investor-flow doesn't just persist within a
single stock, it *propagates* across stocks in the same sector (institutional
sector rotation). Before investing in a graph/GNN model to capture that, this
module answers a much cheaper question: does a sector's average flow today
predict an individual member stock's flow 1-3 trading days later, and is that
lagged relationship meaningfully stronger than the same lagged relationship
against a random, unrelated sector (the control/baseline)?

Every function here is pure DataFrame in -> DataFrame/dict out: no DB
connection, no network call, consistent with
:mod:`kr_quant.features.supply_flow`. Callers should feed in a frame that
already carries a market-cap-normalized ratio column (e.g. produced by
:func:`kr_quant.features.supply_flow.add_normalized_ratios`) joined with
``stocks.sector`` (see ``storage.py``'s ``stocks`` table).

Lag convention: "day t+lag" means *lag rows ahead in that stock's own sorted
date sequence*, not lag calendar days ahead. This matches the gap-tolerant,
positional convention already used by
:func:`kr_quant.features.supply_flow.add_ewma_signal` (a stock's per-code rows
are processed in date order; a missing trading day simply isn't a row).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3)


def add_sector_average(
    df: pd.DataFrame,
    ratio_col: str,
    *,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Add each row's sector-average ratio for that (sector, date).

    Args:
        df: Rows with ``code``, ``date``, ``sector_col``, and the
            market-cap-normalized ratio column ``ratio_col`` (e.g.
            ``"foreign__ratio"`` from
            :func:`kr_quant.features.supply_flow.add_normalized_ratios`).
        ratio_col: Name of the per-code, per-date ratio column to aggregate.
        sector_col: Name of the sector-grouping column (default ``"sector"``,
            matching ``stocks.sector``).

    Returns:
        A copy of ``df`` with two new columns:

        - ``sector_avg_ratio``: the plain mean of ``ratio_col`` across all
          stocks in that (sector, date), including the row's own stock.
        - ``sector_avg_ratio_loo``: the "leave-one-out" mean — the sector
          average *excluding the row's own stock*. This is what correlation
          analyses should use: correlating a stock against a sector average
          that includes itself is a trivial self-correlation, not evidence of
          propagation from *other* stocks. ``NaN`` when the stock is the only
          member of its sector on that date (no "other stocks" to average).
    """
    out = df.copy()
    group = out.groupby([sector_col, "date"])[ratio_col]
    sector_sum = group.transform("sum")
    sector_count = group.transform("count")
    out["sector_avg_ratio"] = sector_sum / sector_count

    denom = sector_count - 1
    loo = (sector_sum - out[ratio_col]) / denom
    out["sector_avg_ratio_loo"] = loo.where(denom > 0)
    return out


def _corr_within_sector(
    df: pd.DataFrame,
    ratio_col: str,
    lag: int,
    sector_col: str,
) -> pd.Series:
    """Per-stock correlation of (own sector's LOO avg at t) vs (own ratio at t+lag)."""
    d = add_sector_average(df, ratio_col, sector_col=sector_col)
    d = d.sort_values(["code", "date"])
    d["_own_future"] = d.groupby("code")[ratio_col].shift(-lag)

    sub = d.dropna(subset=["sector_avg_ratio_loo", "_own_future"])
    sub = sub[["code", "sector_avg_ratio_loo", "_own_future"]]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 3:
            return np.nan
        return g["sector_avg_ratio_loo"].corr(g["_own_future"])

    per_stock = sub.groupby("code")[["sector_avg_ratio_loo", "_own_future"]].apply(_corr)
    return per_stock.dropna()


def _corr_cross_sector(
    df: pd.DataFrame,
    ratio_col: str,
    lag: int,
    sector_col: str,
    rng: np.random.Generator,
) -> pd.Series:
    """Per-stock correlation of (a DIFFERENT random sector's avg at t) vs (own ratio at t+lag).

    Control/baseline: same lagged-correlation machinery, but each stock is
    paired against a sector it does NOT belong to, so any correlation found
    here is attributable to chance/shared market-wide noise rather than
    genuine sector-level propagation.
    """
    sectors = df[sector_col].dropna().unique()
    code_sector = df[["code", sector_col]].drop_duplicates().set_index("code")[sector_col]

    other_sector_map: dict[str, object] = {}
    for code, sec in code_sector.items():
        choices = [s for s in sectors if s != sec]
        if not choices:
            continue
        other_sector_map[code] = rng.choice(choices)

    sector_avg_full = (
        df.groupby([sector_col, "date"])[ratio_col].mean().rename("_other_sector_avg").reset_index()
    )

    d = df.sort_values(["code", "date"]).copy()
    d["_own_future"] = d.groupby("code")[ratio_col].shift(-lag)
    d["_other_sector"] = d["code"].map(other_sector_map)
    d = d.dropna(subset=["_other_sector"])

    merged = d.merge(
        sector_avg_full,
        left_on=["_other_sector", "date"],
        right_on=[sector_col, "date"],
        suffixes=("", "_other"),
    )
    sub = merged.dropna(subset=["_other_sector_avg", "_own_future"])
    sub = sub[["code", "_other_sector_avg", "_own_future"]]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 3:
            return np.nan
        return g["_other_sector_avg"].corr(g["_own_future"])

    per_stock = sub.groupby("code")[["_other_sector_avg", "_own_future"]].apply(_corr)
    return per_stock.dropna()


def lagged_sector_correlation(
    df: pd.DataFrame,
    ratio_col: str,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    sector_col: str = "sector",
    seed: int | None = 0,
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Per-stock, per-lag correlations: within-sector vs. cross-sector baseline.

    Args:
        df: Rows with ``code``, ``date``, ``sector_col``, and ``ratio_col``.
        ratio_col: Market-cap-normalized net-buy ratio column to correlate.
        lags: Lags (in trading-day rows, see module docstring) to evaluate.
            Defaults to 1-3 days per the Phase 1.5 acceptance wording.
        sector_col: Sector-grouping column name.
        seed: Seed for the RNG that picks each stock's random "other sector"
            for the cross-sector control. Fixed default for reproducibility.

    Returns:
        ``(within, cross)`` — two ``dict[lag -> pandas.Series]``, each Series
        indexed by ``code`` with that stock's Pearson correlation for that
        lag. ``within[lag][code]`` is corr(sector LOO avg at t, code's own
        ratio at t+lag); ``cross[lag][code]`` is the same but against a
        randomly chosen *different* sector's average (the control). Stocks
        with too few valid (non-NaN, non-degenerate) observations are simply
        absent from the Series for that lag.
    """
    rng = np.random.default_rng(seed)
    within: dict[int, pd.Series] = {}
    cross: dict[int, pd.Series] = {}
    for lag in lags:
        within[lag] = _corr_within_sector(df, ratio_col, lag, sector_col)
        cross[lag] = _corr_cross_sector(df, ratio_col, lag, sector_col, rng)
    return within, cross


def bootstrap_sector_effect(
    within: Mapping[int, pd.Series],
    cross: Mapping[int, pd.Series],
    *,
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int | None = 0,
) -> dict[str, float | int]:
    """Bootstrap the within-sector vs. cross-sector correlation effect.

    Pools every (lag, code) pair present in both ``within`` and ``cross``
    into paired (within_corr, cross_corr) observations, then resamples the
    paired differences (within - cross) with replacement to build a
    confidence interval for the difference.

    This function does NOT assert or enforce a pass/fail threshold — the
    plan's "2x + CI lower bound > 0" bar is a research decision for whoever
    reads the returned numbers, not something to hardcode here.

    Args:
        within: ``dict[lag -> Series(code -> corr)]`` from
            :func:`lagged_sector_correlation`.
        cross: Same shape, the cross-sector control.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level for the interval (default 0.90 per the plan).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with:

        - ``within_mean`` / ``cross_mean``: mean correlation across all
          pooled (lag, code) pairs.
        - ``ratio``: ``within_mean / cross_mean`` (NaN if ``cross_mean`` is 0).
          Note: when the cross-sector baseline is itself near zero (the
          expected case when there is truly no propagation), this ratio's
          *sign* can be unstable even though its magnitude is informative —
          prefer ``diff_mean`` / ``ci_lower`` for a robust pass/fail read.
        - ``diff_mean``: mean of (within - cross) paired differences.
        - ``ci_lower`` / ``ci_upper``: bootstrap confidence interval on
          ``diff_mean`` at the requested ``ci`` level.
        - ``ci``: the confidence level used.
        - ``n_pairs``: number of paired observations the bootstrap ran on.
    """
    rng = np.random.default_rng(seed)

    w_vals: list[float] = []
    c_vals: list[float] = []
    for lag, w_series in within.items():
        c_series = cross.get(lag)
        if c_series is None:
            continue
        common = w_series.index.intersection(c_series.index)
        w_vals.extend(w_series.loc[common].tolist())
        c_vals.extend(c_series.loc[common].tolist())

    if not w_vals:
        raise ValueError(
            "No overlapping (lag, code) pairs between within-sector and "
            "cross-sector correlation results; cannot bootstrap."
        )

    w_arr = np.asarray(w_vals, dtype=float)
    c_arr = np.asarray(c_vals, dtype=float)
    diff = w_arr - c_arr
    n = len(diff)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diff[idx].mean()

    alpha = 1 - ci
    ci_lower = float(np.quantile(boot_means, alpha / 2))
    ci_upper = float(np.quantile(boot_means, 1 - alpha / 2))

    within_mean = float(w_arr.mean())
    cross_mean = float(c_arr.mean())
    ratio = within_mean / cross_mean if cross_mean != 0 else float("nan")

    return {
        "within_mean": within_mean,
        "cross_mean": cross_mean,
        "ratio": ratio,
        "diff_mean": float(diff.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci": ci,
        "n_pairs": n,
    }


def analyze_sector_propagation(
    df: pd.DataFrame,
    ratio_col: str,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    sector_col: str = "sector",
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int | None = 0,
) -> dict:
    """End-to-end Phase 1.5 pre-check: lagged sector correlation + bootstrap.

    Convenience wrapper combining :func:`lagged_sector_correlation` and
    :func:`bootstrap_sector_effect` into one call, plus per-lag mean
    correlations for inspection.

    Returns:
        The dict from :func:`bootstrap_sector_effect`, extended with
        ``within_by_lag`` and ``cross_by_lag`` (``dict[lag -> mean corr]``).
    """
    within, cross = lagged_sector_correlation(
        df, ratio_col, lags=lags, sector_col=sector_col, seed=seed
    )
    result = bootstrap_sector_effect(within, cross, n_boot=n_boot, ci=ci, seed=seed)
    result["within_by_lag"] = {lag: float(s.mean()) if len(s) else float("nan") for lag, s in within.items()}
    result["cross_by_lag"] = {lag: float(s.mean()) if len(s) else float("nan") for lag, s in cross.items()}
    return result
