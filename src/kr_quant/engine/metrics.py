"""Shared backtest performance metrics — the engine's single source of truth.

Pure, stateless numeric primitives extracted (byte-identical) from the strategy
files so every experiment computes Sharpe / CAGR / drawdown / significance the
same way. Imports numpy/pandas only; this is a leaf module imported *by*
strategies, never the reverse.

Provenance (Step 0 of the backtest-engine migration — copied, signatures
preserved):
    newey_west_t      <- pead._newey_west_t
    summarize_periods <- pead._summarize
    spearman          <- backtest.spearman
    quantile_summary  <- backtest._quantile_summary

``ann_sharpe`` / ``cagr`` / ``max_drawdown`` / ``paired_bootstrap`` /
``regime_buckets`` were extracted from a since-removed comparison module; the
engine copy is now their only definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PPY = 12  # periods per year (monthly return series)


def ann_sharpe(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std() < 1e-12:  # degenerate/no-dispersion → Sharpe undefined
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ppy))


def cagr(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float((1.0 + r).prod() ** (ppy / r.size) - 1.0)


def max_drawdown(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    # 초기자본 1.0 을 첫 peak 로 세운다. 이게 없으면 **첫 구간의 손실이 안 잡힌다** —
    # 자기 자신이 peak 가 되어 drawdown 0 이 되기 때문이다.
    # 실측 반례: max_drawdown([-0.20, +0.05, +0.05]) 가 0.0 을 돌려줬다(정답 -0.20).
    equity = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def newey_west_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean and Newey-West (HAC) t-stat of ``x``, robust to serial correlation.

    Overlapping horizon returns are autocorrelated; a plain t overstates
    significance. The Bartlett-kernel HAC variance with ``lag`` corrects it.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 2:
        return float("nan"), float("nan")
    mu = x.mean()
    d = x - mu
    var = (d @ d) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * ((d[k:] @ d[:-k]) / n)
    se = np.sqrt(var / n)
    return float(mu), float(mu / se) if se > 0 else float("nan")


def summarize_periods(periods: pd.DataFrame, horizon: int) -> dict:
    """Annualized net Sharpe, full-sample t-stat and cumulative return."""
    if periods.empty:
        return {"n": 0, "sharpe": float("nan"), "t_stat": float("nan"),
                "mean_net": float("nan"), "hit_rate": float("nan"),
                "cum_net": float("nan"), "avg_turnover": float("nan")}
    net = periods["net"].to_numpy()
    per_year = 252 / horizon
    std = net.std()
    ann = (1 + net.mean()) ** per_year - 1
    wins = net[net > 0]
    losses = net[net < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0
    return {
        "n": len(net),
        "sharpe": float(ann / (std * np.sqrt(per_year))) if std > 0 else float("nan"),
        "t_stat": float(net.mean() / (std / np.sqrt(len(net)))) if std > 0 else float("nan"),
        "mean_net": float(net.mean()),
        "hit_rate": float((net > 0).mean()),
        "cum_net": float((1 + net).prod() - 1),
        "avg_turnover": float(periods["turnover"].mean()),
        # payoff profile (a quant cares about this more than win rate): a low
        # hit_rate with payoff_ratio > 1 is an asymmetric, convex strategy.
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(avg_win / avg_loss) if avg_loss > 0 else float("nan"),
        "best": float(net.max()),
        "worst": float(net.min()),
    }


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation (Pearson on ranks — no scipy dependency)."""
    if len(a) < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def quantile_summary(merged: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """Mean forward return + hit rate per score quantile (Q1 = highest score)."""
    cols = ["quantile", "n", "mean_fwd", "hit_rate"]
    if len(merged) < quantiles:
        return pd.DataFrame(columns=cols)
    # rank=False so Q1 is the top score bucket; labels 1..quantiles.
    q = pd.qcut(merged["score"].rank(method="first", ascending=False), quantiles, labels=False) + 1
    out = (
        merged.assign(_q=q)
        .groupby("_q")
        .agg(n=("fwd_ret", "size"), mean_fwd=("fwd_ret", "mean"), hit_rate=("fwd_ret", lambda s: (s > 0).mean()))
        .reset_index()
        .rename(columns={"_q": "quantile"})
    )
    return out[cols]


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
        d_sharpe[i] = ann_sharpe(a[idx], ppy) - ann_sharpe(b[idx], ppy)
        d_cagr[i] = cagr(a[idx], ppy) - cagr(b[idx], ppy)
    ds = d_sharpe[np.isfinite(d_sharpe)]
    dc = d_cagr[np.isfinite(d_cagr)]
    nan2 = (float("nan"), float("nan"))
    return {
        "d_sharpe_ci": (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))) if ds.size else nan2,
        "d_cagr_ci": (float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))) if dc.size else nan2,
        "prob_a_better_sharpe": float((ds > 0).mean()) if ds.size else float("nan"),
        "n": n,
    }
