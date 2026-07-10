"""Collect analyst consensus (목표주가·투자의견) from Naver Finance.

Kiwoom's broker API has no analyst consensus; Naver Finance exposes it (FnGuide-
sourced) via the mobile integration endpoint, no auth required. This is the
forward-looking signal PEAD lacks — target-price implied upside and, once a time
series accumulates, **consensus revisions** (the re-rating that drives mega-caps
where post-earnings drift is arbitraged away). See docs/pead-strategy.md.

The endpoint is a **current snapshot**, so this collector is meant to run daily,
appending a date-stamped row per code to build the revision time series over
time. ``parse_consensus`` is pure (JSON in → numbers out) and unit-tested.
Writes CSV: date,code,target_mean,recomm_mean,base_date.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import date

BASE = "https://m.stock.naver.com/api/stock"
UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
      "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148")


def _to_float(s: object) -> float | None:
    txt = str(s or "").replace(",", "").strip()
    try:
        return float(txt)
    except ValueError:
        return None


def parse_consensus(payload: dict) -> tuple[float | None, float | None, str | None]:
    """Extract (target_price_mean, recomm_mean, base_date) from an integration response.

    Args:
        payload: Parsed Naver ``/stock/{code}/integration`` JSON.

    Returns:
        ``(target_mean, recomm_mean, base_date)`` from ``consensusInfo``, or
        ``(None, None, None)`` when the stock has no analyst coverage.
        ``recomm_mean`` is Naver's 1–5 scale (5 = strong buy). ``target_mean`` is
        the mean 12m target price (원). ``base_date`` is the consensus as-of date.
    """
    ci = payload.get("consensusInfo") or {}
    return (
        _to_float(ci.get("priceTargetMean")),
        _to_float(ci.get("recommMean")),
        ci.get("createDate") or None,
    )


def fetch_consensus(code: str, *, retries: int = 3) -> tuple[float | None, float | None, str | None]:
    """Fetch consensus for one code from Naver (throttle/retry on rate limits)."""
    req = urllib.request.Request(f"{BASE}/{code}/integration", headers={"User-Agent": UA})
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return parse_consensus(json.loads(r.read().decode()))
        except Exception:
            time.sleep(1.0)
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="네이버 애널리스트 컨센서스 수집 (목표주가·투자의견, 일별 스냅샷)")
    ap.add_argument("--out", required=True, help="출력 CSV (일별 append)")
    ap.add_argument("--top-n", type=int, default=800, help="유동성 상위 N종목")
    ap.add_argument("--db", default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    import pandas as pd
    from ..storage import connect, default_db_path
    con = connect(args.db or str(default_db_path()))
    top = pd.read_sql_query(
        "SELECT code FROM daily_bars WHERE date >= (SELECT MAX(date) FROM daily_bars) - INTERVAL '90 days' "
        "GROUP BY code ORDER BY AVG(trade_value) DESC LIMIT %(n)s",
        con, params={"n": args.top_n})
    con.close()
    codes = top["code"].tolist()

    today = date.today().isoformat()
    done: set[str] = set()
    if os.path.exists(args.out):
        for r in csv.reader(open(args.out)):
            if r and r[0] == today:
                done.add(r[1])  # (date, code) already collected today
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    n = 0
    for i, code in enumerate(codes, 1):
        if code in done:
            continue
        tm, rm, bd = fetch_consensus(code)
        time.sleep(args.sleep)
        if tm is None and rm is None:
            continue
        w.writerow([today, code, tm if tm is not None else "",
                    rm if rm is not None else "", bd or ""])
        n += 1
        if i % 50 == 0:
            f.flush()
            print(f"[{i}/{len(codes)}] rows={n}", flush=True)
    f.close()
    print(f"DONE date={today} rows={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
