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

PPY = 12  # periods per year (monthly return series)


def _ann_sharpe(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std() < 1e-12:  # degenerate/no-dispersion → Sharpe undefined
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ppy))


def _cagr(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float((1.0 + r).prod() ** (ppy / r.size) - 1.0)


def _max_drawdown(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


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


def regime_buckets(returns: pd.Series, *, n: int = 4) -> list[dict]:
    """Split the return series into ``n`` equal chronological buckets; report each
    bucket's mean and sign — the design's regime-persistence check (want 3+/4 +)."""
    r = returns.to_numpy(float)
    out: list[dict] = []
    if len(r) < n:
        return out
    b = len(r) // n
    for k in range(n):
        seg = r[k * b:(k + 1) * b if k < n - 1 else len(r)]
        m = float(np.nanmean(seg))
        out.append({"start": returns.index[k * b], "mean": m, "positive": m > 0})
    return out


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
    """
    a = ret_a.reindex(ret_b.index).to_numpy(float)
    b = ret_b.to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < block + 1:
        return {"d_sharpe_ci": (float("nan"),) * 2, "d_cagr_ci": (float("nan"),) * 2,
                "prob_a_better_sharpe": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    d_sharpe = np.empty(n_boot)
    d_cagr = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        d_sharpe[i] = _ann_sharpe(a[idx], ppy) - _ann_sharpe(b[idx], ppy)
        d_cagr[i] = _cagr(a[idx], ppy) - _cagr(b[idx], ppy)
    ds = d_sharpe[np.isfinite(d_sharpe)]
    dc = d_cagr[np.isfinite(d_cagr)]
    return {
        "d_sharpe_ci": (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))),
        "d_cagr_ci": (float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))),
        "prob_a_better_sharpe": float((ds > 0).mean()),
        "n": n,
    }


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
