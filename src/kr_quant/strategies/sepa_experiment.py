"""SEPA faithful-reproduction experiment — the one-command orchestrator.

Wires every Phase 0/1/2 component into the pre-registered multi-arm comparison
(``SEPA_FAITHFUL_DESIGN.md``): build the panels from real prices + DART earnings +
shares (split-adjusted, point-in-time), run the five arms, and hand aligned monthly
return series to :func:`kr_quant.strategies.sepa_compare.compare_arms` for the
paired-bootstrap verdict.

Arms:
    A            faithful SEPA (small-mid, trend template + RS + Code33 + VCP, 6 names)
    A-diversified  A's signal, deployed-style diversified sizing (isolates concentration)
    A-noVCP      A without the VCP/pivot gate (isolates VCP's contribution)
    B-shell      deployed large-cap breakout shell (no earnings, diversified)
    C-bench      cap-weighted index proxy

``run_experiment`` is pure (DataFrames in → verdict out) so it is synthetic-testable
before the real backfill; ``main`` wires the DB + earnings CSV for the live run.
"""

from __future__ import annotations

import pandas as pd

from ..features.fundamentals import code33_panel
from ..features.rs_rating import rs_rating_panel
from ..features.universe import smallmid_universe
from ..price_adjust import adjust_prices
from .minervini_sepa import sepa_entries, sepa_trades
from .pead import market_cap_panel
from .sepa_compare import benchmark_returns, book_returns, compare_arms

N_CONCENTRATED = 6      # arm A frozen concentration
N_DIVERSIFIED = 50      # A-diversified / B-shell (deployed over-diversification)
LARGE_CAP_RANK = (0, 100)   # B-shell universe: mega/large (the deployed inversion)


def _adv_panel(prices: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Trailing ``window``-day average trade value → long code/date/adv (as-of)."""
    tv = prices[["code", "date", "trade_value"]].copy()
    tv["trade_value"] = tv["trade_value"].abs()
    tv = tv.sort_values(["code", "date"])
    tv["adv"] = tv.groupby("code")["trade_value"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    return tv.dropna(subset=["adv"])[["code", "date", "adv"]].reset_index(drop=True)


def build_panels(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    adjust: bool = True,
) -> dict:
    """Assemble every panel the arms need (split-adjusted, point-in-time).

    Args:
        prices: Long ``code``/``date``/``open``/``high``/``low``/``close``/
            ``trade_value``/``volume``.
        earnings: DART rows for :func:`code33_panel` (code, avail_date, netinc,
            netinc_prior, revenue, revenue_prior, op_income, op_income_prior).
        shares: ``code``/``date``/shares-outstanding for market cap.
        adjust: Apply corporate-action back-adjustment (both strategy and bench).
    """
    # Normalize date to string so every downstream to_datetime yields the same
    # resolution — DB columns arrive at mixed datetime64 units and break merge_asof.
    prices = prices.copy()
    prices["date"] = prices["date"].astype(str)
    shares = shares.copy()
    shares["date"] = shares["date"].astype(str)
    if adjust:
        prices = adjust_prices(prices)
    dates = sorted(prices["date"].astype(str).unique())
    cap = market_cap_panel(prices, shares)
    adv = _adv_panel(prices)
    return {
        "prices": prices,
        "cap": cap,
        "smallmid": smallmid_universe(cap, adv),                       # rank 100-400
        "largecap": smallmid_universe(cap, adv, cap_rank=LARGE_CAP_RANK),  # B-shell
        "rs": rs_rating_panel(prices),
        "code33": code33_panel(earnings, dates),
    }


def run_experiment(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    adjust: bool = True,
    **boot_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Run all five arms and return ``(comparison_table, verdicts)``.

    Everything downstream of the frozen hyperparameters — no tuning knobs — so a
    live run is a single call once the earnings backfill lands.
    """
    p = build_panels(prices, earnings, shares, adjust=adjust)
    px, rs, c33 = p["prices"], p["rs"], p["code33"]

    ent_a = sepa_entries(px, p["smallmid"], rs, c33, use_vcp=True, use_code33=True)
    trades_a = sepa_trades(px, ent_a)
    ent_avcp = sepa_entries(px, p["smallmid"], rs, c33, use_vcp=False, use_code33=True)
    ent_b = sepa_entries(px, p["largecap"], rs, c33, use_vcp=False, use_code33=False, rs_min=0.0)

    arms = {
        "A": book_returns(px, trades_a, n_slots=N_CONCENTRATED),
        "A-diversified": book_returns(px, trades_a, n_slots=N_DIVERSIFIED),
        "A-noVCP": book_returns(px, sepa_trades(px, ent_avcp), n_slots=N_CONCENTRATED),
        "B-shell": book_returns(px, sepa_trades(px, ent_b), n_slots=N_DIVERSIFIED),
        "C-bench": benchmark_returns(px, p["cap"]),
    }
    # Align every arm on the union of months so the paired bootstrap is well-defined.
    months = sorted(set().union(*[r.index for r in arms.values()]))
    idx = pd.Index(months, name="month")
    arms = {k: r.reindex(idx).fillna(0.0) for k, r in arms.items()}
    return compare_arms(arms, deployed="B-shell", benchmark="C-bench", **boot_kwargs)


def main() -> int:
    """CLI (``kq-sepa``): run the faithful-SEPA arm comparison on real data."""
    import argparse

    import pandas as _pd

    from ..storage import connect, default_db_path

    ap = argparse.ArgumentParser(description="미너비니 SEPA 충실 재현 — 다-arm 비교 판정")
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--earnings-csv", required=True,
                    help="DART 실적 CSV: code,period,avail_date,netinc,netinc_prior,"
                         "revenue,revenue_prior,op_income,op_income_prior")
    ap.add_argument("--no-adjust", action="store_true", help="분할조정 생략(디버그용)")
    args = ap.parse_args()

    # dart_earnings.main() CSV 스키마 (10칸, 헤더 없음): yoy가 6번째 — code33_panel엔
    # 불필요하나 위치 정렬을 위해 이름을 준다. (실적 DB 테이블로 이관되면 여기만 교체.)
    cols = ["code", "period", "avail_date", "netinc", "netinc_prior", "yoy",
            "revenue", "revenue_prior", "op_income", "op_income_prior"]
    ea = _pd.read_csv(args.earnings_csv, names=cols, dtype={"code": str, "avail_date": str, "period": str})
    con = connect(args.db)
    codes = sorted(ea["code"].unique())
    prices = _pd.read_sql_query(
        "SELECT code,date,open,high,low,close,volume,trade_value FROM daily_bars "
        "WHERE code = ANY(%(c)s)", con, params={"c": codes})
    shares = _pd.read_sql_query(
        "SELECT code,date,shares_outstanding FROM shares_outstanding_history WHERE code = ANY(%(c)s)",
        con, params={"c": codes})
    con.close()
    prices["date"] = prices["date"].astype(str)

    table, verdicts = run_experiment(prices, ea, shares, adjust=not args.no_adjust)
    print("\n=== SEPA 다-arm 비교 (분할조정·PIT·페어드 부트스트랩) ===")
    print(table.to_string())
    print("\n=== 판정 (ΔSharpe CI가 0 배제해야 승) ===")
    for arm, v in verdicts.items():
        b, c = v["vs_deployed"]["d_sharpe_ci"], v["vs_benchmark"]["d_sharpe_ci"]
        print(f"{arm}: vs B-shell ΔSharpe CI [{b[0]:+.2f},{b[1]:+.2f}] {'승' if v['beats_b_ci'] else '무'} | "
              f"vs C-bench CI [{c[0]:+.2f},{c[1]:+.2f}] {'승' if v['beats_c_ci'] else '무'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
