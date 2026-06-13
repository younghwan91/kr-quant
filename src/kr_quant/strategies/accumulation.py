"""Accumulation screener: sideways price + foreign/institution net buying.

Wyckoff-style idea: when price moves sideways while "smart money" (foreign and
institutional investors) quietly accumulates — absorbing supply from retail —
it can precede a markup. This screens the collected ``supply_demand`` DB for
that footprint and ranks candidates.

The core :func:`screen` takes a DataFrame so it is unit-testable without a DB
or network. :func:`main` wires it to SQLite for the CLI (``kq-screen``).
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from ..storage import connect, default_db_path

# Floor for the sideways range so tiny ranges don't make the score explode.
_RANGE_FLOOR = 0.02


def load_frame(con: sqlite3.Connection) -> pd.DataFrame:
    """Load supply_demand joined with the stock master into a DataFrame."""
    return pd.read_sql_query(
        """
        SELECT sd.*, s.name, s.market, s.sector
        FROM supply_demand sd
        JOIN stocks s ON s.code = sd.code
        """,
        con,
    )


def screen(
    df: pd.DataFrame,
    *,
    min_days: int = 10,
    max_range_pct: float = 0.15,
    require_retail_selling: bool = True,
    exclude_spac: bool = True,
    min_avg_vol: float = 0.0,
    require_longterm_inst: bool = False,
) -> pd.DataFrame:
    """Rank accumulation candidates.

    Args:
        df: Rows with at least code, name, date, close, acc_trde_qty,
            individual, foreign_, institution.
        min_days: Minimum trading days required per stock.
        max_range_pct: Max (high-low)/mean close to count as "sideways".
        require_retail_selling: Require net individual selling over the window.
        exclude_spac: Drop SPACs (name contains "스팩"). They trade flat near
            par value, so their near-zero range dominates a naive sideways
            screen without representing genuine accumulation.
        min_avg_vol: Minimum average daily volume (shares) — drops illiquid
            micro-caps where flows are noise.
        require_longterm_inst: Require net buying by long-term institutions
            (연기금 + 투신), not just the aggregate "institution" which is
            dominated by 금융투자 (program/arbitrage) flows.

    Returns:
        One row per qualifying stock, sorted by ``score`` (descending). Columns:
        name, market, sector, days, range_pct, foreign_cum, inst_cum,
        indiv_cum, smart_turnover, score.
    """
    if df.empty:
        return _empty_result()

    if exclude_spac:
        df = df[~df["name"].str.contains("스팩", na=False)]
        if df.empty:
            return _empty_result()

    rows = []
    for code, g in df.sort_values("date").groupby("code"):
        if len(g) < min_days:
            continue
        close = g["close"].astype(float)
        mean_close = close.mean()
        if mean_close <= 0:
            continue
        range_pct = (close.max() - close.min()) / mean_close

        foreign_cum = int(g["foreign_"].sum())
        inst_cum = int(g["institution"].sum())
        indiv_cum = int(g["individual"].sum())
        # 연기금 + 투신 = 장기 성격 자금 (금융투자=프로그램/차익 노이즈 제외).
        longterm_cum = int(g["penfnd_etc"].sum() + g["invtrt"].sum())
        avg_vol = float(g["acc_trde_qty"].mean()) or 1.0

        smart_net = foreign_cum + inst_cum
        # Accumulation relative to liquidity: how many average-volume days were
        # net-absorbed by smart money, rewarded for a tighter (more sideways) range.
        smart_turnover = smart_net / avg_vol
        score = smart_turnover / max(range_pct, _RANGE_FLOOR)

        rows.append(
            {
                "code": code,
                "name": g["name"].iloc[0],
                "market": g["market"].iloc[0],
                "sector": g["sector"].iloc[0],
                "days": len(g),
                "range_pct": round(range_pct, 4),
                "foreign_cum": foreign_cum,
                "inst_cum": inst_cum,
                "longterm_cum": longterm_cum,
                "indiv_cum": indiv_cum,
                "avg_vol": int(avg_vol),
                "smart_turnover": round(smart_turnover, 3),
                "score": round(score, 3),
            }
        )

    result = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    if result.empty:
        return result

    mask = (
        (result["range_pct"] <= max_range_pct)
        & (result["foreign_cum"] > 0)
        & (result["inst_cum"] > 0)
        & (result["avg_vol"] >= min_avg_vol)
    )
    if require_retail_selling:
        mask &= result["indiv_cum"] < 0
    if require_longterm_inst:
        mask &= result["longterm_cum"] > 0
    result = result[mask].sort_values("score", ascending=False).reset_index(drop=True)
    return result


_RESULT_COLUMNS = [
    "code", "name", "market", "sector", "days", "range_pct",
    "foreign_cum", "inst_cum", "longterm_cum", "indiv_cum", "avg_vol",
    "smart_turnover", "score",
]


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=_RESULT_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="매집 후보 스크리너 (횡보 + 외인·기관 순매수)")
    parser.add_argument("--db", default=str(default_db_path()))
    parser.add_argument("--top", type=int, default=30, help="상위 N개 출력")
    parser.add_argument("--min-days", type=int, default=10)
    parser.add_argument("--max-range", type=float, default=0.15,
                        help="횡보 판정 최대 변동범위 비율 (기본 0.15=15%%)")
    parser.add_argument("--allow-retail-buying", action="store_true",
                        help="개인 순매수 종목도 포함 (기본: 개인 순매도만)")
    parser.add_argument("--include-spac", action="store_true",
                        help="스팩 포함 (기본: 제외 — 액면가 횡보로 점수 왜곡)")
    parser.add_argument("--min-avg-vol", type=float, default=0.0,
                        help="최소 평균 거래량(주) — 저유동성 종목 제외")
    parser.add_argument("--require-longterm", action="store_true",
                        help="연기금+투신(장기자금) 순매수 종목만")
    parser.add_argument("--csv", default="", help="결과를 CSV 로 저장할 경로")
    args = parser.parse_args()

    con = connect(args.db)
    df = load_frame(con)
    con.close()

    result = screen(
        df,
        min_days=args.min_days,
        max_range_pct=args.max_range,
        require_retail_selling=not args.allow_retail_buying,
        exclude_spac=not args.include_spac,
        min_avg_vol=args.min_avg_vol,
        require_longterm_inst=args.require_longterm,
    )
    if result.empty:
        print("후보 없음 — 데이터가 없거나(먼저 kq-collect) 조건이 너무 엄격합니다.")
        return 0

    top = result.head(args.top)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(top.to_string(index=False))
    print(f"\n총 후보 {len(result)}개 중 상위 {len(top)}개")
    if args.csv:
        result.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"💾 CSV 저장: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
