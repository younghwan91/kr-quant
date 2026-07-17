"""Phase 2: graph-based propagation of the supply-wave signal across stocks.

This is deliberately **not** a deep GNN. The consensus plan (Phase 2) picked a
parameter-free graph-diffusion model over GCN/GraphSAGE because the real
data history is only ~95 trading days per stock — a GNN with thousands of
learned parameters trained on that little data would overfit long before it
captured anything real. A graph is still the right structure for the
project's actual thesis (flow *propagates* between related stocks); what
changes is how a signal moves across that graph. Here it moves by one or two
rounds of plain matrix multiplication against a fixed (not learned)
adjacency matrix — the same operation a single untrained GCN layer performs,
without the parameters that would need more data than exists yet to fit
safely. If/when 6+ months of gap-free data accumulate, a learned GNN becomes
a reasonable upgrade path over this baseline (see the plan's Open Questions).

Graph construction: nodes are stocks, edges connect same-sector stocks
(mirroring the propagation channel already validated cheaply in
:mod:`kr_quant.features.sector_flow`). Edge weight is uniform within a
sector (``1 / (sector_size - 1)``), so the adjacency matrix is row-stochastic
and propagation is literally "this stock's neighbors' average signal."

Every function here is pure array/DataFrame in -> array/DataFrame out: no DB
connection, no network call, no training loop, consistent with the rest of
this package.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features.supply_flow import INVESTOR_TYPES
from ..storage import connect, db_default
from ..strategies.backtest import forward_returns, spearman
from ..strategies.supply_wave import assert_no_lookahead, build_supply_wave_signal, load_frame


def build_sector_adjacency(
    stocks: pd.DataFrame,
    *,
    code_col: str = "code",
    sector_col: str = "sector",
) -> tuple[list[str], np.ndarray]:
    """Row-stochastic same-sector adjacency matrix.

    Args:
        stocks: Rows with at least ``code_col`` and ``sector_col`` (e.g. one
            row per stock, as in the ``stocks`` master table). Duplicate
            codes are dropped, keeping the first occurrence.
        code_col, sector_col: Column names.

    Returns:
        ``(codes, adjacency)`` where ``codes`` is the row/column order (a
        sorted list of unique codes with a non-null sector) and ``adjacency``
        is an ``(n, n)`` float matrix. ``adjacency[i, j]`` is
        ``1 / (sector_size - 1)`` if stocks ``i`` and ``j`` share a sector and
        ``i != j``, else 0. Each row of a non-singleton sector sums to 1
        (row-stochastic); rows for singleton-sector stocks (no sector peers)
        are all-zero — propagation leaves such stocks' own signal untouched,
        which :func:`propagate_signal` handles via its residual blend.
    """
    df = stocks[[code_col, sector_col]].dropna(subset=[sector_col]).drop_duplicates(subset=[code_col])
    df = df.sort_values(code_col).reset_index(drop=True)
    codes = df[code_col].tolist()
    n = len(codes)
    adjacency = np.zeros((n, n), dtype=float)

    sector_groups = df.groupby(sector_col).indices  # sector -> array of row positions
    for _sector, idx in sector_groups.items():
        idx = np.asarray(idx)
        size = len(idx)
        if size < 2:
            continue
        weight = 1.0 / (size - 1)
        for i in idx:
            for j in idx:
                if i != j:
                    adjacency[i, j] = weight
    return codes, adjacency


def propagate_signal(
    signal: pd.Series,
    codes: list[str],
    adjacency: np.ndarray,
    *,
    steps: int = 1,
    alpha: float = 0.5,
) -> pd.Series:
    """Diffuse ``signal`` across the graph for ``steps`` rounds.

    Each round computes ``alpha * own_signal + (1 - alpha) * (adjacency @
    signal)`` — a residual blend (like a single GCN layer with a skip
    connection) so a stock's own signal is never fully overwritten by its
    neighbors', just nudged toward them. ``steps > 1`` repeats this, letting
    influence reach two-hop neighbors (e.g. same-sector-of-a-same-sector, via
    chained rounds) at the cost of over-smoothing risk — the plan defaults to
    1-2 steps, not deeper, precisely to avoid smoothing every stock in a
    sector toward the same value.

    Args:
        signal: Per-code signal values, indexed by code (missing codes are
            treated as 0 for the matrix multiply, then restored to NaN in the
            output so silently-zero-filled values don't masquerade as real
            signal).
        codes: Row/column order matching ``adjacency`` (from
            :func:`build_sector_adjacency`).
        adjacency: Row-stochastic adjacency matrix from
            :func:`build_sector_adjacency`.
        steps: Number of propagation rounds (default 1).
        alpha: Own-signal retention weight in ``[0, 1]``; ``alpha=1`` means no
            propagation at all (returns ``signal`` unchanged), ``alpha=0``
            means full replacement by the neighbor average each round.

    Returns:
        Propagated signal, indexed by ``codes`` (same order/index as input
        would need reindexing to; codes not present in ``signal`` come back
        as ``NaN``).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    vec = signal.reindex(codes).fillna(0.0).to_numpy()
    known_mask = signal.reindex(codes).notna().to_numpy()

    for _ in range(max(steps, 0)):
        neighbor_avg = adjacency @ vec
        vec = alpha * vec + (1 - alpha) * neighbor_avg

    out = pd.Series(vec, index=codes)
    out[~known_mask] = np.nan
    return out


def walk_forward_graph_eval(
    df: pd.DataFrame,
    stocks: pd.DataFrame,
    *,
    investor_col: str = "foreign_",
    halflife: float = 7.0,
    horizons: tuple[int, ...] = (3, 5),
    min_formation: int = 8,
    propagation_steps: int = 1,
    alpha: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: propagated supply-wave signal vs. forward return.

    Same expanding-window structure as
    :func:`kr_quant.strategies.supply_wave.walk_forward_supply_wave_eval`
    (and the same lookahead guard on every split), except the per-split score
    is the *graph-propagated* signal from :func:`propagate_signal` rather
    than the raw per-stock signal. Also reports the raw (Phase 1, no
    propagation) split-level correlations side by side, so a caller can see
    directly whether propagation helped, hurt, or made no difference versus
    the Phase 1 baseline on the same data/splits — that comparison is the
    actual Phase 2 acceptance signal, not an absolute threshold.

    Args:
        df: Same shape as
            :func:`kr_quant.strategies.supply_wave.build_supply_wave_signal`
            expects (``code``, ``date``, investor columns, ``market_cap``).
        stocks: Stock master rows with ``code``/``sector``, for
            :func:`build_sector_adjacency`.
        investor_col, halflife: Forwarded to ``build_supply_wave_signal``.
        horizons, min_formation: Forwarded to the walk-forward split logic.
        propagation_steps, alpha: Forwarded to :func:`propagate_signal`.

    Returns:
        ``(splits, summary)``. ``splits`` has one row per (cutoff, horizon)
        with ``base_date, eval_date, horizon, n, spearman_raw, spearman_graph,
        sign_raw, sign_graph``. ``summary`` has ``n_splits``,
        ``frac_positive_raw``, ``frac_positive_graph`` (the Phase 1 vs.
        Phase 2 gate metrics on identical splits, for direct comparison) and
        ``mean_improvement`` (``spearman_graph - spearman_raw``, averaged
        across splits — positive means propagation helped on average).
    """
    codes, adjacency = build_sector_adjacency(stocks)
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

            use_date = dates[t + 1] if t + 1 < len(dates) else eval_date
            guard_df = pd.DataFrame({"date": snap["date"], "trade_date": use_date})
            assert_no_lookahead(guard_df, source_date_col="date", use_date_col="trade_date")

            raw_signal = snap.set_index("code")["supply_wave_signal"]
            graph_signal = propagate_signal(
                raw_signal, codes, adjacency, steps=propagation_steps, alpha=alpha
            ).dropna()

            fwd = forward_returns(signal_df, base_date, eval_date)

            raw_merged = raw_signal.to_frame("score").merge(
                fwd, left_index=True, right_index=True, how="inner"
            ).dropna()
            graph_merged = graph_signal.to_frame("score").merge(
                fwd, left_index=True, right_index=True, how="inner"
            ).dropna()
            if len(raw_merged) < 2 or len(graph_merged) < 2:
                continue

            corr_raw = spearman(raw_merged["score"], raw_merged["fwd_ret"])
            corr_graph = spearman(graph_merged["score"], graph_merged["fwd_ret"])
            if pd.isna(corr_raw) or pd.isna(corr_graph):
                continue

            splits.append(
                {
                    "base_date": base_date,
                    "eval_date": eval_date,
                    "horizon": h,
                    "n": len(graph_merged),
                    "spearman_raw": corr_raw,
                    "spearman_graph": corr_graph,
                    "sign_raw": "+" if corr_raw > 0 else ("-" if corr_raw < 0 else "0"),
                    "sign_graph": "+" if corr_graph > 0 else ("-" if corr_graph < 0 else "0"),
                }
            )

    splits_df = pd.DataFrame(
        splits,
        columns=[
            "base_date", "eval_date", "horizon", "n",
            "spearman_raw", "spearman_graph", "sign_raw", "sign_graph",
        ],
    )
    if splits_df.empty:
        summary = {
            "n_splits": 0,
            "frac_positive_raw": float("nan"),
            "frac_positive_graph": float("nan"),
            "mean_improvement": float("nan"),
        }
    else:
        summary = {
            "n_splits": len(splits_df),
            "frac_positive_raw": float((splits_df["spearman_raw"] > 0).mean()),
            "frac_positive_graph": float((splits_df["spearman_graph"] > 0).mean()),
            "mean_improvement": float(
                (splits_df["spearman_graph"] - splits_df["spearman_raw"]).mean()
            ),
        }
    return splits_df, summary


def _load_stocks(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT code, sector FROM stocks", con)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: 섹터 그래프 확산 신호 vs 후속 수익률 워크포워드 검증 "
        "(Phase 1 원신호와 나란히 비교)"
    )
    parser.add_argument("--db", default=db_default())
    parser.add_argument("--investor-col", default="foreign_", choices=list(INVESTOR_TYPES))
    parser.add_argument("--halflife", type=float, default=7.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--min-formation", type=int, default=8)
    parser.add_argument("--propagation-steps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    con = connect(args.db)
    df = load_frame(con)
    stocks = _load_stocks(con)
    con.close()

    splits, summary = walk_forward_graph_eval(
        df,
        stocks,
        investor_col=args.investor_col,
        halflife=args.halflife,
        horizons=tuple(args.horizons),
        min_formation=args.min_formation,
        propagation_steps=args.propagation_steps,
        alpha=args.alpha,
    )
    if summary["n_splits"]:
        print(
            f"워크포워드 분할 {summary['n_splits']}개 | "
            f"Phase 1(원신호) 양(+) 비율 {summary['frac_positive_raw']:.0%} | "
            f"Phase 2(그래프) 양(+) 비율 {summary['frac_positive_graph']:.0%} | "
            f"평균 개선폭 {summary['mean_improvement']:+.4f}"
        )
    else:
        print("워크포워드 분할 0개 — 데이터가 부족합니다.")
    if not splits.empty:
        print(splits.to_string(index=False))
    print(
        "\n참고: 이 수치는 그래프 확산이 Phase 1 원신호 대비 개선/악화/무변화 중 "
        "무엇을 보이는지 보여주는 것이지, 특정 임계값 통과를 강제하지 않습니다 — "
        "실제 데이터로 얻은 위 수치를 보고 판단하는 리서치 결정입니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
