"""SEPA arm comparison harness — the Phase 2 evaluation frame.

Turns each arm's trades into an aligned monthly return series and runs the
pre-registered honest comparison (``SEPA_FAITHFUL_DESIGN.md`` §정직검증 프레임 /
판정 기준): annualized Sharpe / CAGR / MaxDD, regime-bucket sign persistence, and
— the piece the architect review required — **paired block-bootstrap** ΔSharpe /
ΔCAGR CIs so an arm only "wins" when the difference CI excludes 0 (not a
high-variance point estimate).

Operates on return series, so it is arm-agnostic: feed A / A₋집중 / A₋VCP (from
:func:`kr_quant.strategies.minervini_sepa.sepa_trades`), the deployed shell B, and
the benchmark C. Pure functions; synthetic-testable without the earnings backfill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.metrics import (
    ann_sharpe,
    cagr,
    max_drawdown,
    paired_bootstrap as _engine_bootstrap,
    regime_buckets as _engine_regimes,
)

from .minervini_sizing import PILOT_FRAC, REST_WEIGHT, STOP_PCT, TOP_WEIGHT

PPY = 12  # periods per year (monthly return series)

# Metric primitives now live in ``kr_quant.engine.metrics`` (single source of
# truth). Re-exported under their historical private names for backward compat —
# ``sepa_experiment.py`` still does ``from .sepa_compare import _ann_sharpe, _cagr``.
_ann_sharpe = ann_sharpe
_cagr = cagr
_max_drawdown = max_drawdown


def monthly_book_returns(trades: pd.DataFrame, months: list[str]) -> pd.Series:
    """Aggregate per-trade returns into an equal-weight monthly book return series.

    Each trade's realized return is booked in its **exit month**; a month's return
    is the mean over trades exiting that month (0 when none — the book is flat).
    Returns a Series indexed by ``months`` (``YYYY-MM``) so arms align for the
    paired bootstrap. (Skeleton portfolio layer — realized-at-exit; a concurrent
    mark-to-market book can refine this later without changing the comparison API.)
    """
    idx = pd.Index(months, name="month")
    if trades.empty:
        return pd.Series(0.0, index=idx, name="ret")
    ex = trades.copy()
    ex["month"] = ex["exit_date"].astype(str).str.slice(0, 7)
    by_month = ex.groupby("month")["ret"].mean()
    return by_month.reindex(idx).fillna(0.0).rename("ret")


def _monthly_last_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-code last close in each month (code × ``YYYY-MM``)."""
    p = prices[["code", "date", "close"]].copy()
    p["close"] = p["close"].abs()
    p["month"] = p["date"].astype(str).str.slice(0, 7)
    return p.sort_values("date").groupby(["code", "month"])["close"].last().unstack("month")


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-code monthly returns (code × ``YYYY-MM``) from the last close each month."""
    return _monthly_last_close(prices).pct_change(axis=1)


def book_returns(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    n_slots: int = 6,
    top_w: float = TOP_WEIGHT,
    rest_w: float = REST_WEIGHT,
    pilot_frac: float = PILOT_FRAC,
    r_threshold: float = STOP_PCT,
    sized: bool = True,
    pyramid: bool = True,
    pyramid_max_adds: int = 2,
    pyramid_step: float = 1.0,
) -> pd.Series:
    """Monthly return series of a concurrent ≤ ``n_slots`` book, marked to market.

    ``n_slots`` is the concentration lever (frozen 6 for arm A vs large for the
    diversified / shell arms). When ``sized`` and ``trades`` carry a ``score``
    column, the frozen Minervini exposure curve is applied per position along one
    proof axis: ``pilot_frac`` before +1R (pilot), full at +1R, then **pyramid** —
    ``+pyramid_step`` exposure at each +2R milestone up to ``pyramid_max_adds`` (add
    to winners). Positions are ranked by score and weighted ``top_w`` (best) /
    ``rest_w`` (others). Without a ``score`` column it stays equal-weight
    (backward-compat). Weights are renormalized each month; more-than-``n_slots``
    opens keep the earliest-entered.

    ⚠️ **Exit precision (fixed 2026-07-12):** the entry/exit month's contribution
    uses the position's actual fill/exit price (``trades["entry_price"]``/
    ``["exit_price"]`` when present, else an exact-date close lookup in ``prices``)
    rather than that month's aggregate market return — otherwise a stop/sell-half/
    tennis-cull that fires mid-month was invisible whenever it didn't cross a month
    boundary, silently making ``sepa_trades`` exit-rule parameters (e.g. ``stop_pct``
    in :func:`kr_quant.strategies.sepa_experiment.robustness_sweep`) have **no
    effect** on the aggregated book Sharpe. Interior (fully-held) months still mark
    to the market close, an accepted simplification for multi-month holds.
    """
    mret = _monthly_returns(prices)
    last_close = _monthly_last_close(prices)
    months = list(mret.columns)
    month_idx = {m: i for i, m in enumerate(months)}
    idx = pd.Index(months, name="month")
    if trades.empty or not months:
        return pd.Series(0.0, index=idx, name="ret")
    close_lookup = prices.pivot_table(index="code", columns="date", values="close", aggfunc="first").abs()

    def _price_at(code: str, date: str) -> float:
        try:
            return float(close_lookup.at[code, date])
        except KeyError:
            return float("nan")

    tr = trades.sort_values("entry_date").reset_index(drop=True)
    apply_sizing = sized and "score" in tr.columns
    has_prices = "entry_price" in tr.columns and "exit_price" in tr.columns
    pos = []
    for _, t in tr.iterrows():
        entry_date, exit_date = str(t["entry_date"]), str(t["exit_date"])
        ep = float(t["entry_price"]) if has_prices and pd.notna(t.get("entry_price")) else _price_at(t["code"], entry_date)
        xp = float(t["exit_price"]) if has_prices and pd.notna(t.get("exit_price")) else _price_at(t["code"], exit_date)
        pos.append({
            "code": t["code"], "f": entry_date[:7], "x": exit_date[:7],
            "entry_price": ep, "exit_price": xp,
            "score": float(t["score"]) if apply_sizing and pd.notna(t.get("score")) else np.nan,
        })
    cum = [0.0] * len(pos)  # cumulative return since entry, per position (for pilot)

    def _month_return(k: int, m: str) -> float:
        """Position ``k``'s return contribution for month ``m`` — exact fill/exit
        price at the entry/exit boundary, market close-to-close for interior months."""
        p = pos[k]
        code, ep, xp = p["code"], p["entry_price"], p["exit_price"]
        if m == p["f"] == p["x"]:                       # entered and exited within this month
            return xp / ep - 1.0 if (np.isfinite(ep) and ep > 0) else float("nan")
        if m == p["f"]:                                  # entry month, held past month-end
            lc = last_close.at[code, m] if code in last_close.index else np.nan
            return float(lc) / ep - 1.0 if (np.isfinite(lc) and np.isfinite(ep) and ep > 0) else float("nan")
        if m == p["x"]:                                  # exit month, held since before it
            mi = month_idx[m]
            prior = last_close.at[code, months[mi - 1]] if mi > 0 and code in last_close.index else np.nan
            return xp / float(prior) - 1.0 if (np.isfinite(prior) and prior > 0) else float("nan")
        return mret.at[code, m] if code in mret.index else float("nan")  # interior: mark to market

    out = {}
    for m in months:
        active = [k for k, p in enumerate(pos) if p["f"] <= m <= p["x"]][:n_slots]
        if not active:
            out[m] = 0.0
            continue
        if apply_sizing:
            order = sorted(active, key=lambda k: pos[k]["score"] if np.isfinite(pos[k]["score"]) else -1e18,
                           reverse=True)
            base_w = {order[0]: top_w, **{k: rest_w for k in order[1:]}}
        else:
            base_w = {k: 1.0 for k in active}
        num = den = 0.0
        for k in active:
            r = _month_return(k, m)
            if not np.isfinite(r):
                continue
            if not apply_sizing or cum[k] >= r_threshold:
                # full at +1R, then +pyramid_step exposure per +2R milestone (winners scale up).
                adds = min(int((cum[k] + 1e-9) // (2.0 * r_threshold)), pyramid_max_adds) \
                    if (apply_sizing and pyramid) else 0
                scale = 1.0 + adds * pyramid_step
            else:
                scale = pilot_frac                            # pilot: half until proven (+1R)
            w = base_w[k] * scale
            num += w * r
            den += w
            cum[k] = (1.0 + cum[k]) * (1.0 + r) - 1.0  # update after using prior cum
        out[m] = num / den if den > 0 else 0.0
    return pd.Series(out, name="ret").rename_axis("month")


def benchmark_returns(prices: pd.DataFrame, cap_panel: pd.DataFrame) -> pd.Series:
    """Cap-weighted monthly index-proxy return (arm C) — the honest benchmark.

    Each month weights every name's monthly return by its market cap at the prior
    month-end (as-of, no look-ahead), mirroring a cap-weighted KOSPI proxy.
    """
    mret = _monthly_returns(prices)
    months = list(mret.columns)
    cap = cap_panel[["code", "date", "market_cap"]].copy()
    cap["month"] = cap["date"].astype(str).str.slice(0, 7)
    mcap = cap.sort_values("date").groupby(["code", "month"])["market_cap"].last().unstack("month")
    out = {}
    for k, m in enumerate(months):
        if k == 0:
            out[m] = 0.0
            continue
        prev = months[k - 1]
        w = mcap[prev] if prev in mcap.columns else None
        r = mret[m] if m in mret.columns else None
        if w is None or r is None:
            out[m] = 0.0
            continue
        common = w.dropna().index.intersection(r.dropna().index)
        if len(common) == 0 or w.reindex(common).sum() <= 0:
            out[m] = 0.0
            continue
        wt = w.reindex(common) / w.reindex(common).sum()
        out[m] = float((wt * r.reindex(common)).sum())
    return pd.Series(out, name="ret").rename_axis("month")


def regime_buckets(returns: pd.Series, *, n: int = 4) -> list[dict]:
    """Split the return series into ``n`` equal chronological buckets; report each
    bucket's mean and sign — the design's regime-persistence check (want 3+/4 +).

    Delegates to :func:`kr_quant.engine.metrics.regime_buckets` (single source of
    truth); kept here so ``compare_arms`` and callers keep the same import path.
    """
    return _engine_regimes(returns, n=n)


def paired_bootstrap(
    ret_a: pd.Series,
    ret_b: pd.Series,
    *,
    block: int = 6,
    n_boot: int = 2000,
    seed: int = 0,
    ppy: int = PPY,
) -> dict:
    """Block bootstrap of the **paired** difference A−B in Sharpe and CAGR.

    Resamples aligned blocks of the two series jointly (same indices for A and B,
    preserving their pairing and autocorrelation), recomputes ΔSharpe and ΔCAGR on
    each resample, and returns the 95% CIs plus P(A>B). An arm only clears the
    pre-registered bar when the CI **excludes 0**.

    Returns:
        ``{"d_sharpe_ci", "d_cagr_ci", "prob_a_better_sharpe", "n"}`` where the CIs
        are ``(lo, hi)`` at the 2.5/97.5 percentiles.

    Delegates to :func:`kr_quant.engine.metrics.paired_bootstrap` (single source of
    truth); kept here so ``compare_arms`` and callers keep the same import path.
    """
    return _engine_bootstrap(ret_a, ret_b, block=block, n_boot=n_boot, seed=seed, ppy=ppy)


def compare_arms(
    arm_returns: dict[str, pd.Series],
    *,
    deployed: str = "B",
    benchmark: str = "C",
    n_regimes: int = 4,
    **boot_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Full arm comparison table + pre-registered paired verdicts.

    Args:
        arm_returns: ``{arm_name: monthly_return_series}`` (aligned or reindexable).
        deployed, benchmark: keys of the B (shell) and C (index) reference arms.
        n_regimes: chronological buckets for the persistence check.
        boot_kwargs: forwarded to :func:`paired_bootstrap` (block, n_boot, seed).

    Returns:
        ``(table, verdicts)``. ``table`` has per-arm Sharpe/CAGR/MaxDD and the
        positive-regime count. ``verdicts`` maps each non-reference arm to its
        paired bootstrap vs ``deployed`` and vs ``benchmark`` and a ``beats_b_ci``
        / ``beats_c_ci`` flag (ΔSharpe CI excludes 0 on the positive side).
    """
    rows = []
    for name, r in arm_returns.items():
        regs = regime_buckets(r, n=n_regimes)
        rows.append({
            "arm": name,
            "sharpe": _ann_sharpe(r.to_numpy(float)),
            "cagr": _cagr(r.to_numpy(float)),
            "max_dd": _max_drawdown(r.to_numpy(float)),
            "pos_regimes": f"{sum(x['positive'] for x in regs)}/{len(regs)}" if regs else "n/a",
        })
    table = pd.DataFrame(rows).set_index("arm")

    verdicts: dict[str, dict] = {}
    for name, r in arm_returns.items():
        if name in (deployed, benchmark):
            continue
        vs_b = paired_bootstrap(r, arm_returns[deployed], **boot_kwargs)
        vs_c = paired_bootstrap(r, arm_returns[benchmark], **boot_kwargs)
        verdicts[name] = {
            "vs_deployed": vs_b,
            "vs_benchmark": vs_c,
            "beats_b_ci": vs_b["d_sharpe_ci"][0] > 0,
            "beats_c_ci": vs_c["d_sharpe_ci"][0] > 0,
        }
    return table, verdicts
