"""Collect quarterly net income (당기순이익) YoY from DART — the input to the
validated PEAD⊕value alpha (see :mod:`kr_quant.strategies.pead`).

DART's ``fnlttSinglAcnt`` returns ``thstrm_amount`` (current period) and
``frmtrm_amount`` (prior-year same period), so YoY earnings growth — the PEAD
surprise proxy — comes straight from one call. Each figure is stamped with a
lookahead-safe ``avail_date`` = period-end + filing lag (see
:func:`kr_quant.features.fundamentals.available_date`), so downstream use never
peeks at a report before it was public.

``parse_net_income`` is pure (JSON in → numbers out) and unit-tested without the
network. ``main`` (``kq-collect-earnings``) wires fetching + the liquid universe
and writes the CSV that ``kq-pead`` consumes.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

from ..features.fundamentals import available_date

BASE = "https://opendart.fss.or.kr/api"
# reprt_code: Q1, half-year(=Q2 cumulative), Q3, annual.
QUARTER_REPORT = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}
QUARTER_END = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
NET_INCOME_ACCOUNTS = ("당기순이익", "당기순이익(손실)", "분기순이익", "반기순이익")


def _to_float(s: object) -> float | None:
    txt = str(s or "").replace(",", "").strip()
    try:
        return float(txt)
    except ValueError:
        return None


def parse_net_income(payload: dict) -> tuple[float | None, float | None]:
    """Extract (current, prior-year) net income from a ``fnlttSinglAcnt`` response.

    Args:
        payload: Parsed DART JSON. ``status`` must be "000"; ``list`` holds the
            statement rows.

    Returns:
        ``(netinc, netinc_prior)`` — current-period and prior-year-same-period net
        income, or ``(None, None)`` when the report is missing or has no
        net-income line. Picks the first row whose ``account_nm`` is a known
        net-income label (see :data:`NET_INCOME_ACCOUNTS`).
    """
    if payload.get("status") != "000":
        return None, None
    ni = nip = None
    for row in payload.get("list", []):
        if row.get("account_nm", "").strip() in NET_INCOME_ACCOUNTS:
            v = _to_float(row.get("thstrm_amount"))
            vp = _to_float(row.get("frmtrm_amount"))
            if v is not None:
                ni = v
            if vp is not None:
                nip = vp
            if ni is not None:
                break
    return ni, nip


def yoy_growth(netinc: float | None, prior: float | None) -> float | None:
    """YoY net-income growth = (curr - prior) / |prior|, or None if not computable."""
    if netinc is None or prior in (None, 0):
        return None
    return (netinc - prior) / abs(prior)


def _get_json(url: str, params: dict, *, retries: int = 3) -> dict:
    q = urllib.parse.urlencode(params)
    for _ in range(retries):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(1.0)
    return {}


def load_corp_map(api_key: str) -> dict[str, str]:
    """Return ``{stock_code: corp_code}`` from DART's corpCode.xml zip.

    Raises ``RuntimeError`` with the DART status when the response is an error
    XML (e.g. status 020 = daily call-limit exceeded, 010 = bad key) rather than
    the expected zip — otherwise the caller would see an opaque ``BadZipFile``.
    """
    q = urllib.parse.urlencode({"crtfc_key": api_key})
    with urllib.request.urlopen(f"{BASE}/corpCode.xml?{q}", timeout=60) as r:
        raw = r.read()
    if not raw[:2] == b"PK":  # zip magic; DART errors come back as small XML
        status = ""
        try:
            status = ET.fromstring(raw.decode()).findtext("status") or ""
        except Exception:
            pass
        raise RuntimeError(f"DART corpCode 오류 (status={status!r}) — 한도초과(020)/키오류(010) 등 확인")
    z = zipfile.ZipFile(io.BytesIO(raw))
    root = ET.fromstring(z.read(z.namelist()[0]).decode())
    out: dict[str, str] = {}
    for it in root.iter("list"):
        sc = (it.findtext("stock_code") or "").strip()
        cc = (it.findtext("corp_code") or "").strip()
        if sc and cc:
            out[sc] = cc
    return out


def fetch_net_income(api_key: str, corp_code: str, year: int, quarter: int) -> tuple[float | None, float | None]:
    """Fetch (current, prior) net income for one corp/year/quarter from DART."""
    payload = _get_json(f"{BASE}/fnlttSinglAcnt.json", {
        "crtfc_key": api_key, "corp_code": corp_code,
        "bsns_year": str(year), "reprt_code": QUARTER_REPORT[quarter],
    })
    return parse_net_income(payload)


def main() -> int:
    ap = argparse.ArgumentParser(description="DART 분기 순이익 YoY 수집 (PEAD 입력)")
    ap.add_argument("--out", required=True, help="출력 CSV 경로")
    ap.add_argument("--top-n", type=int, default=800, help="유동성 상위 N종목")
    ap.add_argument("--from-year", type=int, default=2018)
    ap.add_argument("--to-year", type=int, default=datetime.now().year)
    ap.add_argument("--db", default=None)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    api_key = os.environ.get("DART_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 DART_API_KEY 필요")

    import pandas as pd
    from ..storage import connect, default_db_path
    con = connect(args.db or str(default_db_path()))
    top = pd.read_sql_query(
        "SELECT code FROM daily_bars WHERE date >= (SELECT MAX(date) FROM daily_bars) - INTERVAL '90 days' "
        "GROUP BY code ORDER BY AVG(trade_value) DESC LIMIT %(n)s",
        con, params={"n": args.top_n})
    con.close()
    codes = top["code"].tolist()

    done: set[str] = set()
    if os.path.exists(args.out):
        for r in csv.reader(open(args.out)):
            if r:
                done.add(r[0])
    corp = load_corp_map(api_key)
    print(f"corp_map {len(corp)} | universe {len(codes)} | already done {len(done)}", flush=True)

    today = datetime.now().strftime("%Y%m%d")
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    n = 0
    for i, code in enumerate(codes, 1):
        if code in done or code not in corp:
            continue
        for year in range(args.from_year, args.to_year + 1):
            for q in (1, 2, 3, 4):
                avail = available_date(f"{year}-{QUARTER_END[q][:2]}-{QUARTER_END[q][2:]}",
                                       is_annual=(q == 4)).strftime("%Y%m%d")
                if avail > today:
                    continue
                ni, nip = fetch_net_income(api_key, corp[code], year, q)
                time.sleep(args.sleep)
                if ni is None:
                    continue
                w.writerow([code, f"{year}Q{q}", avail, ni, nip if nip is not None else "",
                            yoy_growth(ni, nip) if yoy_growth(ni, nip) is not None else ""])
                n += 1
        f.flush()
        if i % 25 == 0:
            print(f"[{i}/{len(codes)}] rows={n}", flush=True)
    f.close()
    print(f"DONE rows={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
