""""파도타기" supply-wave signal: combine EWMA + cross-sectional rank supply-flow
features into a single per-code, per-date signal, guard against lookahead bias,
and walk-forward evaluate the signal against forward returns.

This is Phase 1 of the supply-flow prediction plan
(``.omc/plans/20260708-supply-flow-prediction.md``). The Phase 1 gate is
**not** a hard-fail Sharpe threshold — it is "the fraction of walk-forward
splits where the supply-wave signal's Spearman correlation with the forward
return is positive." This module computes and reports that fraction; whether
80% (or some other threshold) is "enough" is a research decision made after
looking at real data, not something enforced here as a pass/fail gate.

Like :mod:`kr_quant.strategies.accumulation` and
:mod:`kr_quant.strategies.backtest`, the core functions are pure DataFrame
in -> DataFrame/dict out: no DB connection, no network call. :func:`main`
wires them to SQLite (or Postgres/TimescaleDB via ``storage.connect``) for the
CLI (``kq-supply-wave``).
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..features.supply_flow import (
    INVESTOR_TYPES,
    add_cross_sectional_rank,
    add_ewma_signal,
    add_normalized_ratios,
)
from ..storage import connect, db_default, market_cap_asof_bulk
from .backtest import forward_returns, spearman


class LookaheadError(Exception):
    """Raised when a signal appears to use data timestamped at/after its use point."""


def assert_no_lookahead(
    df: pd.DataFrame,
    *,
    source_date_col: str = "date",
    use_date_col: str = "trade_date",
) -> None:
    """Assert every row's source-data date is strictly before its trade-use date.

    This is the lookahead guard required by the plan: the raw data a signal is
    built from (``source_date_col``, e.g. the ``supply_demand``/EWMA date the
    signal value is computed as-of) must be strictly earlier than the date the
    signal is actually used to enter a trade (``use_date_col``). Same-day is
    **not** treated as legitimate here — a signal computed from data dated
    ``T`` may only be acted on starting ``T+1`` or later, matching how
    :func:`kr_quant.strategies.backtest.rolling_backtest` treats its formation
    cutoff (``base_date``) as strictly before the date being predicted
    (``eval_date``).

    Dates are compared as strings (``YYYYMMDD``, this repo's convention
    throughout ``storage.py``/``backtest.py``), which sorts correctly
    lexicographically.

    Raises:
        LookaheadError: if any row has ``source_date_col >= use_date_col``
            (i.e. the signal would be using data not yet available at the
            time it is supposedly acted on). This is a genuinely enforced
            check, not a no-op: feeding it deliberately future-shifted data
            raises.
    """
    if df.empty:
        return
    bad = df[df[source_date_col].astype(str) >= df[use_date_col].astype(str)]
    if not bad.empty:
        example = bad.iloc[0][[source_date_col, use_date_col]].to_dict()
        raise LookaheadError(
            f"Lookahead violation: {len(bad)} row(s) have "
            f"{source_date_col} >= {use_date_col} (signal source data is not "
            f"strictly before its trade-use date). Example row: {example}"
        )


def build_supply_wave_signal(
    df: pd.DataFrame,
    *,
    investor_col: str = "foreign_",
    halflife: float = 7.0,
    rank_method: str = "pct",
) -> pd.DataFrame:
    """Combine normalized ratio + EWMA + cross-sectional rank into one signal.

    Pipeline (each step delegates to :mod:`kr_quant.features.supply_flow`):
    1. :func:`add_normalized_ratios` — market-cap-normalized net-buy ratio per
       investor type.
    2. :func:`add_ewma_signal` on the chosen investor's ratio column — smooths
       day-to-day noise into a "supply velocity" signal (no burn-in needed,
       tolerates gaps).
    3. :func:`add_cross_sectional_rank` on the EWMA column — turns the
       absolute smoothed ratio into a same-day percentile rank across the
       universe, which is what actually gets compared across codes/dates.

    Args:
        df: Rows with ``code``, ``date``, the investor-type net-buy columns
            (see ``INVESTOR_TYPES``), and ``market_cap`` (join in via
            :func:`kr_quant.storage.market_cap_asof` before calling this).
        investor_col: Which investor type's flow to build the signal from
            (must be a key of ``INVESTOR_TYPES``, e.g. ``"foreign_"``).
        halflife: EWMA halflife in trading days, forwarded to
            :func:`add_ewma_signal`.
        rank_method: Forwarded to :func:`add_cross_sectional_rank`.

    Returns:
        A copy of ``df`` with the intermediate ratio/EWMA/rank columns plus a
        final ``supply_wave_signal`` column (the cross-sectional percentile
        rank of the EWMA-smoothed, market-cap-normalized net-buy ratio).
    """
    if investor_col not in INVESTOR_TYPES:
        raise ValueError(f"investor_col must be one of {INVESTOR_TYPES}, got {investor_col!r}")

    out = add_normalized_ratios(df)
    ratio_col = f"{investor_col}_ratio"
    out = add_ewma_signal(out, ratio_col, halflife=halflife)
    ewma_col = f"{ratio_col}_ewma"
    out = add_cross_sectional_rank(out, ewma_col, method=rank_method)
    out["supply_wave_signal"] = out[f"{ewma_col}_rank"]
    return out


def walk_forward_supply_wave_eval(
    df: pd.DataFrame,
    *,
    investor_col: str = "foreign_",
    halflife: float = 7.0,
    horizons: tuple[int, ...] = (3, 5),
    min_formation: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: per split, Spearman-correlate the supply-wave signal with
    the forward return; report the fraction of splits with a positive sign.

    Uses the same expanding-window split structure as
    :func:`kr_quant.strategies.backtest.rolling_backtest` (cutoff ``t`` slides
    from ``min_formation`` to ``len(dates) - horizon``, for each ``horizon``
    in ``horizons``), but the per-split score is the supply-wave signal from
    :func:`build_supply_wave_signal` rather than the accumulation
    :func:`kr_quant.strategies.accumulation.screen` score.

    For each split, the lookahead guard (:func:`assert_no_lookahead`) is
    applied: the signal's source date (``base_date``, i.e. the last date of
    data it was computed from) must be strictly before the date the position
    would actually be entered (the next trading day after ``base_date``).

    Args:
        df: Rows with ``code``, ``date``, ``close``, the investor-type
            columns, and ``market_cap`` — same shape :func:`build_supply_wave_signal`
            expects.
        investor_col, halflife: Forwarded to :func:`build_supply_wave_signal`.
        horizons: Forward-return holding periods (trading days) to evaluate.
        min_formation: Minimum number of leading trading days before the
            first split's cutoff (mirrors ``rolling_backtest``'s
            ``min_formation``).

    Returns:
        ``(splits, summary)``:
        ``splits`` has one row per (cutoff, horizon) with ``base_date,
        eval_date, horizon, n, spearman, sign`` (``sign`` is ``"+"``, ``"-"``,
        or ``"0"``).
        ``summary`` holds ``n_splits``, ``signs`` (list of per-split signs)
        and ``frac_positive`` (fraction of splits with ``spearman > 0`` — the
        Phase 1 gate metric, reported but **not** enforced as pass/fail here).
    """
    signal_df = build_supply_wave_signal(df, investor_col=investor_col, halflife=halflife)
    dates = sorted(signal_df["date"].astype(str).unique())
    splits: list[dict] = []

    for h in horizons:
        for t in range(min_formation - 1, len(dates) - h):
            base_date, eval_date = dates[t], dates[t + h]
            snap = signal_df[signal_df["date"].astype(str) == base_date].dropna(
                subset=["supply_wave_signal"]
            )
            if len(snap) < 2:
                continue

            # Guard: the signal computed as-of base_date may only be used to
            # trade starting the *next* trading day, never on/before base_date.
            use_date = dates[t + 1] if t + 1 < len(dates) else eval_date
            guard_df = pd.DataFrame({"date": snap["date"], "trade_date": use_date})
            assert_no_lookahead(guard_df, source_date_col="date", use_date_col="trade_date")

            fwd = forward_returns(signal_df, base_date, eval_date)
            merged = snap.merge(fwd, left_on="code", right_index=True, how="inner").dropna(
                subset=["fwd_ret"]
            )
            if len(merged) < 2:
                continue

            corr = spearman(merged["supply_wave_signal"], merged["fwd_ret"])
            if pd.isna(corr):
                continue
            sign = "+" if corr > 0 else ("-" if corr < 0 else "0")
            splits.append(
                {
                    "base_date": base_date,
                    "eval_date": eval_date,
                    "horizon": h,
                    "n": len(merged),
                    "spearman": corr,
                    "sign": sign,
                }
            )

    splits_df = pd.DataFrame(
        splits, columns=["base_date", "eval_date", "horizon", "n", "spearman", "sign"]
    )
    if splits_df.empty:
        summary = {"n_splits": 0, "signs": [], "frac_positive": float("nan")}
    else:
        summary = {
            "n_splits": len(splits_df),
            "signs": splits_df["sign"].tolist(),
            "frac_positive": float((splits_df["spearman"] > 0).mean()),
        }
    return splits_df, summary


def load_frame(con) -> pd.DataFrame:
    """Load ``supply_demand`` joined with the stock master and market cap.

    Same base query as :func:`kr_quant.strategies.accumulation.load_frame`,
    with a ``market_cap`` column added via
    :func:`kr_quant.storage.market_cap_asof` (lookahead-safe: only uses
    ``shares_outstanding_history`` rows dated on/before each ``date``).
    """
    df = pd.read_sql_query(
        """
        SELECT sd.*, s.name, s.market, s.sector
        FROM supply_demand sd
        JOIN stocks s ON s.code = sd.code
        """,
        con,
    )
    # Same backend-normalization as accumulation.load_frame — Postgres
    # returns datetime.date objects for a DATE column, sqlite returns TEXT;
    # normalize before market_cap_asof() (string comparisons) or any
    # downstream string-keyed lookup.
    df["date"] = df["date"].astype(str)
    df["market_cap"] = market_cap_asof_bulk(con, df)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="수급 파도타기 신호(EWMA+랭크) vs 후속 수익률 워크포워드 검증"
    )
    parser.add_argument("--db", default=db_default())
    parser.add_argument("--investor-col", default="foreign_", choices=list(INVESTOR_TYPES))
    parser.add_argument("--halflife", type=float, default=7.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--min-formation", type=int, default=8)
    args = parser.parse_args()

    con = connect(args.db)
    df = load_frame(con)
    con.close()

    splits, summary = walk_forward_supply_wave_eval(
        df,
        investor_col=args.investor_col,
        halflife=args.halflife,
        horizons=tuple(args.horizons),
        min_formation=args.min_formation,
    )
    print(
        f"워크포워드 분할 {summary['n_splits']}개 | "
        f"양(+) 부호 비율 {summary['frac_positive']:.0%}"
        if summary["n_splits"]
        else "워크포워드 분할 0개 — 데이터가 부족합니다."
    )
    if not splits.empty:
        print(splits.to_string(index=False))
    print(
        "\n참고: Phase 1 게이트(양(+) 비율 80% 이상)는 이 스크립트가 강제하지 않습니다 — "
        "실제 데이터로 얻은 위 수치를 보고 판단하는 리서치 결정입니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
