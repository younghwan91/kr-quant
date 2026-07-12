"""Lightweight validation of the accumulation score against forward returns.

Splits the collected window into a *formation* period (used to screen and score
candidates) and a *holdout* period (used to measure what happened next), then
checks whether a higher accumulation score lines up with a higher subsequent
return.

This is an **in-sample illustration**, not a rigorous backtest: with a single
short window there is no rolling out-of-sample evaluation, no transaction costs,
and no survivorship control. It is meant to show the screener carries signal,
with the rank correlation and per-quintile spread quantifying how much.

The core :func:`backtest` takes a DataFrame so it is unit-testable without a DB.
:func:`main` wires it to SQLite for the CLI (``kq-backtest``).
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..storage import connect, db_default
from .accumulation import load_frame, screen


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation (Pearson on ranks — no scipy dependency)."""
    if len(a) < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def forward_returns(df: pd.DataFrame, base_date: str, eval_date: str) -> pd.Series:
    """Per-code return from ``base_date``'s close to ``eval_date``'s close.

    Kiwoom stores a signed close (the sign marks the day's direction), so we
    take the absolute value to recover the price level.
    """
    piv = df.pivot_table(index="code", columns="date", values="close", aggfunc="first").abs()
    return (piv[eval_date] / piv[base_date] - 1.0).rename("fwd_ret")


def backtest(
    df: pd.DataFrame,
    *,
    formation_days: int = 12,
    quantiles: int = 5,
    **screen_kwargs: object,
) -> tuple[pd.DataFrame, dict]:
    """Score candidates on the formation window, score forward returns on the rest.

    Args:
        df: Supply/demand rows joined with the stock master (see
            :func:`kr_quant.strategies.accumulation.load_frame`).
        formation_days: Number of leading trading days used to screen/score.
            The remaining days are the holdout used to measure forward return.
        quantiles: Number of score buckets for the per-quintile summary.
        **screen_kwargs: Forwarded to :func:`screen` (e.g. ``max_range_pct``).

    Returns:
        ``(merged, summary)`` where ``merged`` is the candidate table plus a
        ``fwd_ret`` column (sorted by score, descending), and ``summary`` holds
        ``n``, ``spearman``, ``universe_mean`` and a ``buckets`` DataFrame of
        mean forward return per score quantile.
    """
    dates = sorted(df["date"].unique())
    if len(dates) < formation_days + 2:
        raise ValueError(
            f"백테스트에 거래일이 부족합니다: {len(dates)}일 (형성 {formation_days}일 + 보유 ≥2일 필요)"
        )

    form_dates = dates[:formation_days]
    base_date, eval_date = form_dates[-1], dates[-1]

    candidates = screen(df[df["date"].isin(form_dates)], **screen_kwargs)  # type: ignore[arg-type]
    fwd = forward_returns(df, base_date, eval_date)
    merged = (
        candidates.merge(fwd, left_on="code", right_index=True, how="inner")
        .dropna(subset=["fwd_ret"])
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    buckets = _quantile_summary(merged, quantiles)
    summary = {
        "formation_days": formation_days,
        "base_date": base_date,
        "eval_date": eval_date,
        "n": len(merged),
        "spearman": spearman(merged["score"], merged["fwd_ret"]) if not merged.empty else float("nan"),
        "universe_mean": float(fwd.dropna().mean()),
        "buckets": buckets,
    }
    return merged, summary


def _quantile_summary(merged: pd.DataFrame, quantiles: int) -> pd.DataFrame:
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


def rolling_backtest(
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (3, 5),
    min_formation: int = 8,
    quantiles: int = 5,
    **screen_kwargs: object,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward validation: slide the formation cutoff across the window.

    The single-split :func:`backtest` can be lucky. Here we expand the formation
    window day by day (cutoff ``t`` from ``min_formation`` to the end minus the
    holding horizon) and, for each forward ``horizon`` in days, re-screen and
    measure the score↔return rank correlation. Pooling every candidate across
    splits also yields an outlier-robust **median** return per score quantile.

    Caveat: with a short single window the splits **overlap** (expanding window,
    overlapping holding periods), so they are correlated — this measures
    stability across as-of dates, not independent out-of-sample replication.
    True OOS needs a longer history (collect more days).

    Returns:
        ``(splits, summary)``: ``splits`` has one row per (cutoff, horizon) with
        ``base_date, eval_date, horizon, n, spearman``; ``summary`` holds
        ``n_splits``, Spearman mean/min/max and ``frac_positive``, plus a pooled
        ``buckets`` DataFrame (mean_fwd, **median_fwd**, hit_rate per quantile).
    """
    dates = sorted(df["date"].unique())
    splits: list[dict] = []
    pooled: list[pd.DataFrame] = []

    for h in horizons:
        for t in range(min_formation - 1, len(dates) - h):
            form_dates = dates[: t + 1]
            base_date, eval_date = dates[t], dates[t + h]
            cands = screen(df[df["date"].isin(form_dates)], **screen_kwargs)  # type: ignore[arg-type]
            if cands.empty:
                continue
            fwd = forward_returns(df, base_date, eval_date)
            m = (
                cands.merge(fwd, left_on="code", right_index=True, how="inner")
                .dropna(subset=["fwd_ret"])
            )
            if len(m) < quantiles:
                continue
            splits.append({
                "base_date": base_date, "eval_date": eval_date, "horizon": h,
                "n": len(m), "spearman": spearman(m["score"], m["fwd_ret"]),
            })
            q = pd.qcut(m["score"].rank(method="first", ascending=False), quantiles, labels=False) + 1
            pooled.append(m.assign(quantile=q)[["quantile", "fwd_ret"]])

    splits_df = pd.DataFrame(splits)
    pooled_df = (
        pd.concat(pooled, ignore_index=True)
        if pooled else pd.DataFrame(columns=["quantile", "fwd_ret"])
    )
    buckets = (
        pooled_df.groupby("quantile")
        .agg(
            n=("fwd_ret", "size"),
            mean_fwd=("fwd_ret", "mean"),
            median_fwd=("fwd_ret", "median"),
            hit_rate=("fwd_ret", lambda s: (s > 0).mean()),
        )
        .reset_index()
    ) if not pooled_df.empty else pd.DataFrame(columns=["quantile", "n", "mean_fwd", "median_fwd", "hit_rate"])

    sp = splits_df["spearman"] if not splits_df.empty else pd.Series(dtype=float)
    summary = {
        "n_splits": len(splits_df),
        "spearman_mean": float(sp.mean()) if not sp.empty else float("nan"),
        "spearman_min": float(sp.min()) if not sp.empty else float("nan"),
        "spearman_max": float(sp.max()) if not sp.empty else float("nan"),
        "frac_positive": float((sp > 0).mean()) if not sp.empty else float("nan"),
        "buckets": buckets,
    }
    return splits_df, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="매집 점수 vs 후속 수익률 검증 (형성구간 스크리닝 → 보유구간 수익률)"
    )
    parser.add_argument("--db", default=db_default())
    parser.add_argument("--formation-days", type=int, default=12,
                        help="형성구간(스크리닝) 거래일 수 — 나머지는 보유구간")
    parser.add_argument("--max-range", type=float, default=0.15)
    parser.add_argument("--min-days", type=int, default=8)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--rolling", action="store_true",
                        help="워크포워드 검증 (형성구간 확장 × 보유기간, 분할별 Spearman 집계)")
    args = parser.parse_args()

    con = connect(args.db)
    df = load_frame(con)
    con.close()

    if args.rolling:
        return _run_rolling(df, args)

    merged, summary = backtest(
        df,
        formation_days=args.formation_days,
        quantiles=args.quantiles,
        min_days=args.min_days,
        max_range_pct=args.max_range,
    )
    print(f"형성구간: {df['date'].min()}..{summary['base_date']} ({args.formation_days}일)  "
          f"보유구간: {summary['base_date']}..{summary['eval_date']}")
    print(f"후보 {summary['n']}개 | Spearman(점수~수익률) = {summary['spearman']:.3f} | "
          f"전체 평균 수익률 {summary['universe_mean']:+.2%}")
    if not summary["buckets"].empty:
        b = summary["buckets"].copy()
        b["mean_fwd"] = b["mean_fwd"].map("{:+.2%}".format)
        b["hit_rate"] = b["hit_rate"].map("{:.0%}".format)
        print("\n점수 분위별 (Q1=최고점):")
        print(b.to_string(index=False))
    return 0


def _run_rolling(df: pd.DataFrame, args: argparse.Namespace) -> int:
    splits, roll = rolling_backtest(
        df, min_formation=args.min_days, quantiles=args.quantiles,
        min_days=args.min_days, max_range_pct=args.max_range,
    )
    print(f"워크포워드 분할 {roll['n_splits']}개 | "
          f"Spearman 평균 {roll['spearman_mean']:+.3f} "
          f"(min {roll['spearman_min']:+.3f}, max {roll['spearman_max']:+.3f}) | "
          f"양(+) 비율 {roll['frac_positive']:.0%}")
    if not roll["buckets"].empty:
        b = roll["buckets"].copy()
        for c in ("mean_fwd", "median_fwd"):
            b[c] = b[c].map("{:+.2%}".format)
        b["hit_rate"] = b["hit_rate"].map("{:.0%}".format)
        print("\n분위별 (전 분할 풀링, Q1=최고점):")
        print(b.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
