"""Multi-channel supply-wave signal: combines several distinct "who's buying
and how" channels into one composite score, instead of Phase 1's single
foreign-investor EWMA-rank signal.

Rationale (from a user discussion on 2026-07-09, after Phase 1's single-
channel foreign_ signal showed only 55% directional consistency): a single
investor type's raw net-buy flow conflates several different behaviors that
plausibly have different predictive value:

- **Long-term institutional accumulation** (연기금/사모펀드, ``penfnd_etc``/
  ``samo_fund``) turns over far less than 금융투자 (program/prop-desk flow)
  — treating "기관계" as one undifferentiated block, as Phase 1 effectively
  did by only trying ``foreign_``, throws this distinction away.
- **Short covering** (:mod:`kr_quant.features.short_flow`) is forced/
  incentivized buying, mechanically different from discretionary
  accumulation, and comes from a different source table entirely
  (``short_selling``, not ``supply_demand``).
- **Average-cost gap** (:func:`kr_quant.features.supply_flow.add_avg_cost_gap`)
  asks a different question than raw flow: not "is this investor type
  buying today" but "are they sitting on gains or still underwater" — a
  proxy for whether continued buying is likely to persist (averaging down)
  or reverse (profit-taking).

Each channel is reduced to the same same-day cross-sectional percentile rank
(so channels are on a comparable [0, 1] scale before combining) and the
composite score is their unweighted mean. This is deliberately the simplest
possible combination rule — a first pass to see whether adding channels
helps *at all* before investing in a learned weighting.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..features.credit_flow import add_credit_signal
from ..features.short_flow import add_short_covering_signal
from ..features.supply_flow import (
    INVESTOR_TYPES,
    add_avg_cost_gap,
    add_cross_sectional_rank,
    add_ewma_signal,
    add_normalized_ratios,
)
from ..storage import connect, default_db_path, market_cap_asof_bulk
from .backtest import forward_returns, spearman
from .supply_wave import assert_no_lookahead

# Default channel set: two long-horizon institutional sub-types (연기금,
# 사모펀드) plus foreign flow (kept from Phase 1 for continuity/comparison).
DEFAULT_FLOW_CHANNELS: tuple[str, ...] = ("foreign_", "penfnd_etc", "samo_fund")


def build_multi_channel_signal(
    supply_df: pd.DataFrame,
    short_df: pd.DataFrame,
    *,
    flow_channels: tuple[str, ...] = DEFAULT_FLOW_CHANNELS,
    cost_gap_channel: str = "penfnd_etc",
    halflife: float = 7.0,
    rank_method: str = "pct",
    flow_feature: str = "level",
) -> pd.DataFrame:
    """Combine EWMA-flow, avg-cost-gap, and short-covering channels into one score.

    Args:
        supply_df: Rows with ``code``, ``date``, ``close``, ``market_cap``,
            and the investor-type net-buy columns (see
            :data:`kr_quant.features.supply_flow.INVESTOR_TYPES`) — same shape
            :func:`kr_quant.features.supply_flow.add_normalized_ratios` expects.
        short_df: Rows with ``code``, ``date``, and ``short_balance`` (from
            the ``short_selling`` table) — same shape
            :func:`kr_quant.features.short_flow.add_short_covering_signal`
            expects.
        flow_channels: Which investor types' EWMA-smoothed, market-cap-
            normalized net-buy ratio to include as channels (each becomes its
            own same-day cross-sectional rank).
        cost_gap_channel: Which investor type's average-cost gap
            (:func:`kr_quant.features.supply_flow.add_avg_cost_gap`) to
            include as an additional channel.
        halflife: EWMA halflife forwarded to
            :func:`kr_quant.features.supply_flow.add_ewma_signal`.
        rank_method: Forwarded to
            :func:`kr_quant.features.supply_flow.add_cross_sectional_rank`.

    Returns:
        A copy of ``supply_df`` (short-covering merged in on ``code``/
        ``date``, inner join — rows with no short-selling data for that
        code/date are dropped) with one ``{channel}_rank`` column per flow
        channel, ``{cost_gap_channel}_cost_gap_rank``,
        ``short_covering_rank``, and a final ``multi_signal`` column: the
        unweighted row-wise mean of all channel rank columns (``NaN``
        channels are skipped per row via ``mean(skipna=True)``, so a stock
        missing one channel on a given day still gets a score from the rest).
    """
    out, channel_rank_cols = build_channel_features(
        supply_df,
        short_df,
        flow_channels=flow_channels,
        cost_gap_channel=cost_gap_channel,
        halflife=halflife,
        rank_method=rank_method,
        flow_feature=flow_feature,
    )
    out["multi_signal"] = out[channel_rank_cols].mean(axis=1, skipna=True)
    return out


def build_channel_features(
    supply_df: pd.DataFrame,
    short_df: pd.DataFrame,
    *,
    flow_channels: tuple[str, ...] = DEFAULT_FLOW_CHANNELS,
    cost_gap_channel: str = "penfnd_etc",
    halflife: float = 7.0,
    rank_method: str = "pct",
    credit_df: pd.DataFrame | None = None,
    flow_feature: str = "level",
) -> tuple[pd.DataFrame, list[str]]:
    """Build the per-channel rank columns without collapsing them into a score.

    Shared by :func:`build_multi_channel_signal` (equal-weight mean) and
    :mod:`kr_quant.models.ensemble_signal` (learned weights) so both combiners
    operate on identically-defined channels. See
    :func:`build_multi_channel_signal` for argument docs.

    Args:
        credit_df: Optional rows with ``code``, ``date``, ``balance_rt``
            (from the ``credit_balance`` table). When given, adds a credit-
            crowding-trend channel via
            :func:`kr_quant.features.credit_flow.add_credit_signal` (inner
            join, same as the ``short_df`` merge — stocks with no credit
            data on a date are dropped only from this additional merge step,
            not from earlier channels). When ``None`` (default), no credit
            channel is added — existing callers are unaffected.
        flow_feature: Which representation of each ``flow_channels`` EWMA
            signal to use as a channel — this was previously hardcoded to
            "level" everywhere, silently discarding the "acceleration"
            (``_ewma_diff``) that :func:`kr_quant.features.supply_flow.add_ewma_signal`
            already computes. One of:

            - ``"level"`` (default, prior behavior): the EWMA-smoothed ratio
              itself — "how much is this investor type net-buying right now".
            - ``"accel"``: the EWMA's day-over-day change — "is the buying
              speeding up or slowing down", independent of the absolute
              level.
            - ``"vector"``: both level *and* accel as separate channels per
              flow type (doubles the number of flow-channel columns) — lets
              the ridge model weigh them independently rather than forcing a
              single representation.

    Returns:
        ``(df, channel_rank_cols)`` — ``df`` is ``supply_df`` with short (and
        optionally credit) data merged in and one ``{channel}_rank`` column
        added per channel; ``channel_rank_cols`` lists those column names in
        a stable order.
    """
    if not all(c in INVESTOR_TYPES for c in flow_channels):
        raise ValueError(f"flow_channels must all be in {INVESTOR_TYPES}, got {flow_channels!r}")
    if cost_gap_channel not in INVESTOR_TYPES:
        raise ValueError(
            f"cost_gap_channel must be one of {INVESTOR_TYPES}, got {cost_gap_channel!r}"
        )
    if flow_feature not in ("level", "accel", "vector"):
        raise ValueError(f"flow_feature must be 'level'/'accel'/'vector', got {flow_feature!r}")

    out = add_normalized_ratios(supply_df)
    channel_rank_cols: list[str] = []

    for investor_col in flow_channels:
        ratio_col = f"{investor_col}_ratio"
        out = add_ewma_signal(out, ratio_col, halflife=halflife)
        ewma_col = f"{ratio_col}_ewma"
        diff_col = f"{ewma_col}_diff"
        if flow_feature in ("level", "vector"):
            out = add_cross_sectional_rank(out, ewma_col, method=rank_method)
            channel_rank_cols.append(f"{ewma_col}_rank")
        if flow_feature in ("accel", "vector"):
            out = add_cross_sectional_rank(out, diff_col, method=rank_method)
            channel_rank_cols.append(f"{diff_col}_rank")

    out = add_avg_cost_gap(out, cost_gap_channel)
    gap_col = f"{cost_gap_channel}_cost_gap"
    out = add_cross_sectional_rank(out, gap_col, method=rank_method)
    channel_rank_cols.append(f"{gap_col}_rank")

    short = add_short_covering_signal(short_df)[["code", "date", "short_covering"]]
    out = out.merge(short, on=["code", "date"], how="inner")
    out = add_cross_sectional_rank(out, "short_covering", method=rank_method)
    channel_rank_cols.append("short_covering_rank")

    if credit_df is not None:
        credit = add_credit_signal(credit_df, halflife=halflife)[
            ["code", "date", "credit_balance_rt_ewma", "credit_balance_rt_chg"]
        ]
        out = out.merge(credit, on=["code", "date"], how="inner")
        out = add_cross_sectional_rank(out, "credit_balance_rt_chg", method=rank_method)
        channel_rank_cols.append("credit_balance_rt_chg_rank")

    return out, channel_rank_cols


def walk_forward_multi_signal_eval(
    supply_df: pd.DataFrame,
    short_df: pd.DataFrame,
    *,
    flow_channels: tuple[str, ...] = DEFAULT_FLOW_CHANNELS,
    cost_gap_channel: str = "penfnd_etc",
    halflife: float = 7.0,
    horizons: tuple[int, ...] = (3, 5),
    min_formation: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: multi-channel composite signal vs. forward return.

    Same expanding-window structure, lookahead guard, and reported metrics
    (per-split sign, ``frac_positive``) as
    :func:`kr_quant.strategies.supply_wave.walk_forward_supply_wave_eval`,
    except the per-split score is :func:`build_multi_channel_signal`'s
    ``multi_signal`` column instead of the single-channel
    ``supply_wave_signal``. Directly comparable to the Phase 1 result on the
    same metric definition.

    Returns:
        ``(splits, summary)`` — see
        :func:`kr_quant.strategies.supply_wave.walk_forward_supply_wave_eval`
        for the exact shape; column/key names are identical here.
    """
    signal_df = build_multi_channel_signal(
        supply_df,
        short_df,
        flow_channels=flow_channels,
        cost_gap_channel=cost_gap_channel,
        halflife=halflife,
    )
    dates = sorted(signal_df["date"].astype(str).unique())
    splits: list[dict] = []

    for h in horizons:
        for t in range(min_formation - 1, len(dates) - h):
            base_date, eval_date = dates[t], dates[t + h]
            snap = signal_df[signal_df["date"].astype(str) == base_date].dropna(
                subset=["multi_signal"]
            )
            if len(snap) < 2:
                continue

            use_date = dates[t + 1] if t + 1 < len(dates) else eval_date
            guard_df = pd.DataFrame({"date": snap["date"], "trade_date": use_date})
            assert_no_lookahead(guard_df, source_date_col="date", use_date_col="trade_date")

            fwd = forward_returns(signal_df, base_date, eval_date)
            merged = snap.merge(fwd, left_on="code", right_index=True, how="inner").dropna(
                subset=["fwd_ret"]
            )
            if len(merged) < 2:
                continue

            corr = spearman(merged["multi_signal"], merged["fwd_ret"])
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


def load_multi_frame(con) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ``supply_demand`` (+market_cap), ``short_selling``, and
    ``credit_balance`` frames.

    Returns ``(supply_df, short_df, credit_df)`` shaped for
    :func:`build_multi_channel_signal` / :func:`walk_forward_multi_signal_eval`
    / :mod:`kr_quant.models.ensemble_signal`. Callers that don't use the
    credit channel can simply ignore the third element.
    """
    supply_df = pd.read_sql_query(
        """
        SELECT sd.*, s.name, s.market, s.sector
        FROM supply_demand sd
        JOIN stocks s ON s.code = sd.code
        """,
        con,
    )
    supply_df["date"] = supply_df["date"].astype(str)
    supply_df["market_cap"] = market_cap_asof_bulk(con, supply_df)

    short_df = pd.read_sql_query(
        "SELECT code, date, short_balance FROM short_selling", con
    )
    short_df["date"] = short_df["date"].astype(str)

    credit_df = pd.read_sql_query(
        "SELECT code, date, balance_rt FROM credit_balance", con
    )
    credit_df["date"] = credit_df["date"].astype(str)

    return supply_df, short_df, credit_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="다채널 수급 신호(장기기관 EWMA+평단갭+숏커버링) vs 후속 수익률 워크포워드 검증"
    )
    parser.add_argument("--db", default=str(default_db_path()))
    parser.add_argument(
        "--flow-channels", nargs="+", default=list(DEFAULT_FLOW_CHANNELS), choices=INVESTOR_TYPES
    )
    parser.add_argument("--cost-gap-channel", default="penfnd_etc", choices=INVESTOR_TYPES)
    parser.add_argument("--halflife", type=float, default=7.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--min-formation", type=int, default=8)
    args = parser.parse_args()

    con = connect(args.db)
    supply_df, short_df, _credit_df = load_multi_frame(con)
    con.close()

    splits, summary = walk_forward_multi_signal_eval(
        supply_df,
        short_df,
        flow_channels=tuple(args.flow_channels),
        cost_gap_channel=args.cost_gap_channel,
        halflife=args.halflife,
        horizons=tuple(args.horizons),
        min_formation=args.min_formation,
    )
    if summary["n_splits"]:
        print(
            f"워크포워드 분할 {summary['n_splits']}개 | "
            f"다채널 신호 양(+) 비율 {summary['frac_positive']:.0%}"
        )
    else:
        print("워크포워드 분할 0개 — 데이터가 부족합니다.")
    if not splits.empty:
        print(splits.to_string(index=False))
    print(
        "\n참고: Phase 1 단일채널(foreign_) 결과와 나란히 비교하기 위한 것으로, "
        "이 스크립트는 임계값을 강제하지 않습니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
