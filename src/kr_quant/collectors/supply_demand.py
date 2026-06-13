"""Collect per-stock investor supply/demand (개인·외국인·기관 순매수) into SQLite.

Source: ``ka10059`` (투자자기관별종목별요청) — one request per stock returns up to
100 daily rows, so a 1-month window fits in a single call. Because this is a
single TR, the kiwoom-client per-TR rate limiter throttles to ~1 req/s; a full
KOSPI+KOSDAQ common-stock sweep (~2,600 stocks) takes roughly 45 minutes.

CLI:
    kq-collect --market all --days 30            # mock server
    kq-collect --market all --days 30 --prod     # real data
    kq-collect --resume                          # skip already-collected
"""

from __future__ import annotations

import argparse
import sqlite3
import time

from kiwoom_rest_api import KiwoomAPI
from kiwoom_rest_api.base import KiwoomAPIError

from ..config import make_api
from ..storage import (
    INVESTOR_COLUMNS,
    connect,
    default_db_path,
    to_float,
    to_int,
    upsert_stocks,
    upsert_supply_demand,
)

# ka10099 (종목정보 리스트) market-type codes.
MARKETS: dict[str, str] = {"kospi": "0", "kosdaq": "10"}


def is_common_stock(row: dict) -> bool:
    """True for common shares only.

    Excludes ETF/ETN/REITs (``market`` is not 거래소/코스닥) and preferred
    shares (KRX common-stock codes end in ``0``; preferred end in 5/7/K/...).
    """
    return row["market"] in ("거래소", "코스닥") and row["code"].endswith("0")


def fetch_stock_list(api: KiwoomAPI, markets: list[str]) -> list[dict]:
    """Fetch and normalize the stock master for the given markets."""
    out: list[dict] = []
    for market in markets:
        resp = api.stock_info.stock_info_list(mrkt_tp=MARKETS[market])
        for row in resp.get("list", []):
            out.append(
                {
                    "code": row.get("code", "").strip(),
                    "name": row.get("name", "").strip(),
                    "market": row.get("marketName", "").strip(),
                    "sector": row.get("upName", "").strip(),
                    "kind": row.get("kind", "").strip(),
                }
            )
    return out


def _has_recent_rows(con: sqlite3.Connection, code: str, cutoff: str) -> bool:
    cur = con.execute(
        "SELECT 1 FROM supply_demand WHERE code=? AND date>=? LIMIT 1", (code, cutoff)
    )
    return cur.fetchone() is not None


def collect(
    api: KiwoomAPI,
    con: sqlite3.Connection,
    stocks: list[dict],
    *,
    days: int = 30,
    resume: bool = False,
    progress_every: int = 50,
) -> dict[str, int]:
    """Collect supply/demand for ``stocks`` into the DB. Returns a summary dict."""
    cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - days * 86400))
    today = time.strftime("%Y%m%d")
    stats = {"done": 0, "skipped": 0, "failed": 0, "rows": 0}
    started = time.monotonic()

    for i, stock in enumerate(stocks, 1):
        code = stock["code"]
        if resume and _has_recent_rows(con, code, cutoff):
            stats["skipped"] += 1
            continue
        try:
            resp = api.stock_info.investor_institution_by_stock(
                dt=today, stk_cd=code, amt_qty_tp="2", trde_tp="0", unit_tp="1"
            )
            records = []
            for row in resp.get("stk_invsr_orgn", []) or []:
                date = row.get("dt", "")
                if date < cutoff:
                    continue
                records.append(
                    (
                        code,
                        date,
                        # cur_prc 의 부호는 전일대비 등락 방향이므로 절댓값(가격)으로 저장.
                        abs(to_int(row.get("cur_prc"))),
                        to_float(row.get("flu_rt")),
                        to_int(row.get("acc_trde_qty")),
                        *[to_int(row.get(src)) for src in INVESTOR_COLUMNS.values()],
                    )
                )
            stats["rows"] += upsert_supply_demand(con, records)
            stats["done"] += 1
        except KiwoomAPIError as e:
            stats["failed"] += 1
            print(f"  ⚠️ {code} {stock['name']}: rc={e.code} {e.message[:50]}")
        except Exception as e:  # noqa: BLE001 — isolate per-stock failures
            stats["failed"] += 1
            print(f"  💥 {code} {stock['name']}: {type(e).__name__}: {e}")

        if i % progress_every == 0 or i == len(stocks):
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed else 0
            eta = (len(stocks) - i) / rate / 60 if rate else 0
            print(
                f"  [{i}/{len(stocks)}] done={stats['done']} skip={stats['skipped']} "
                f"fail={stats['failed']} | {stats['rows']:,} rows | "
                f"{rate:.1f} stk/s | ETA {eta:.1f}m"
            )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="키움 수급 데이터 SQLite 수집기")
    parser.add_argument("--prod", action="store_true", help="실서버 사용 (기본: 모의)")
    parser.add_argument("--market", choices=["kospi", "kosdaq", "all"], default="all")
    parser.add_argument("--days", type=int, default=30, help="최근 N일")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N종목만 (테스트)")
    parser.add_argument("--db", default=str(default_db_path()))
    parser.add_argument("--resume", action="store_true", help="수집된 종목 건너뜀")
    parser.add_argument(
        "--all-kinds", action="store_true",
        help="ETF/ETN/리츠/우선주 등 모두 포함 (기본: 보통주만)",
    )
    parser.add_argument(
        "--rate", type=float, default=0.9,
        help="TR당 요청 속도(req/s). 긴 전수 수집의 429 방지를 위해 기본 0.9",
    )
    args = parser.parse_args()

    con = connect(args.db)
    # 장시간 단일-TR 반복이라 보수적으로: 약간 느린 속도 + 넉넉한 재시도로 429 흡수.
    api = make_api(is_mock=not args.prod, rate_limit=args.rate, max_retries=5)

    markets = ["kospi", "kosdaq"] if args.market == "all" else [args.market]
    stocks = fetch_stock_list(api, markets)
    if not args.all_kinds:
        stocks = [s for s in stocks if is_common_stock(s)]
    if args.limit:
        stocks = stocks[: args.limit]

    upsert_stocks(con, stocks)
    server = "모의" if not args.prod else "실서버"
    print(f"🔌 {server} | 시장={args.market} | 종목 {len(stocks)}개 | 최근 {args.days}일")
    print(f"💾 {args.db}\n")

    stats = collect(api, con, stocks, days=args.days, resume=args.resume)

    api.close()
    con.close()
    print(
        f"\n✅ 완료: done={stats['done']} skip={stats['skipped']} "
        f"fail={stats['failed']} rows={stats['rows']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
