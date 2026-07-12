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
from typing import Any

from ..features.fundamentals import available_date

BASE = "https://opendart.fss.or.kr/api"
# reprt_code: Q1, half-year(=Q2 cumulative), Q3, annual.
QUARTER_REPORT = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}
QUARTER_END = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
NET_INCOME_ACCOUNTS = ("당기순이익", "당기순이익(손실)", "분기순이익", "반기순이익")
# 매출·영업이익 계정명 (krx-fundamentals-api ACCOUNT_MAP 준용) — Code33(EPS·매출·마진
# 3분기 연속 가속) 재현의 매출·마진 소스. 동일 ``fnlttSinglAcnt`` 응답에서 함께 온다.
REVENUE_ACCOUNTS = ("매출액", "수익(매출액)", "영업수익")
OP_INCOME_ACCOUNTS = ("영업이익", "영업이익(손실)")


def _to_float(s: object) -> float | None:
    txt = str(s or "").replace(",", "").strip()
    try:
        return float(txt)
    except ValueError:
        return None


def _pick(rows: list[dict], names: tuple[str, ...]) -> tuple[float | None, float | None]:
    """First (current, prior-year) amounts whose ``account_nm`` is in ``names``.

    Shared by :func:`parse_net_income` and :func:`parse_financials`. Mirrors the
    original net-income scan: fill prior even before current is found, and stop
    at the first row that yields a current amount.
    """
    cur = prior = None
    for row in rows:
        if row.get("account_nm", "").strip() in names:
            v = _to_float(row.get("thstrm_amount"))
            vp = _to_float(row.get("frmtrm_amount"))
            if v is not None:
                cur = v
            if vp is not None:
                prior = vp
            if cur is not None:
                break
    return cur, prior


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
    return _pick(payload.get("list", []), NET_INCOME_ACCOUNTS)


def parse_financials(
    payload: dict,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Extract net income + revenue + operating income (current & prior) at once.

    The Code33 SEPA filter needs EPS **and** revenue **and** margin acceleration;
    revenue and operating income ride along in the same ``fnlttSinglAcnt`` payload
    as net income, so one parse yields all three. Margin = op_income / revenue is
    derived downstream (Phase 1), keeping this a pure extractor.

    Args:
        payload: Parsed DART JSON (same shape as :func:`parse_net_income`).

    Returns:
        ``(netinc, netinc_prior, revenue, revenue_prior, op_income, op_income_prior)``.
        Any leg absent from the report is ``None`` — a net-income-only response
        yields the net-income pair with the revenue/op-income legs ``None`` (no
        crash), and ``status != "000"`` yields all six ``None``.
    """
    if payload.get("status") != "000":
        return None, None, None, None, None, None
    rows = payload.get("list", [])
    ni, nip = _pick(rows, NET_INCOME_ACCOUNTS)
    rev, revp = _pick(rows, REVENUE_ACCOUNTS)
    oi, oip = _pick(rows, OP_INCOME_ACCOUNTS)
    return ni, nip, rev, revp, oi, oip


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


def load_corp_map_with_rotation(keys: list[str]) -> dict[str, str]:
    """``load_corp_map`` with key rotation on daily-limit (020) — same rotation
    ``_fetch_with_rotation`` already does for the per-quarter fetches, but for
    the one-time corp_code map load at startup. Without this, a single
    exhausted key[0] kills the whole run even when key[1..] still have quota
    (real incident: a 14.5h overnight backfill run legitimately stopped for
    the day via 020 mid-run, and the very next retry died instantly on
    ``load_corp_map(keys[0])`` alone despite a second key being available).
    """
    last_err: Exception | None = None
    for key in keys:
        try:
            return load_corp_map(key)
        except RuntimeError as e:
            # 에러 메시지 템플릿 자체에 "한도초과(020)/키오류(010)" 안내문구가 항상
            # 박혀있어 "020" in str(e)는 010 에러에도 항상 참이 된다 — status='020'
            # 형태로 실제 값만 정확히 매칭해야 함(실측 버그).
            if "status='020'" not in str(e):
                raise
            last_err = e
    raise last_err if last_err else RuntimeError("DART 키 없음")


def fetch_net_income(api_key: str, corp_code: str, year: int, quarter: int) -> tuple[float | None, float | None]:
    """Fetch (current, prior) net income for one corp/year/quarter from DART."""
    return parse_net_income(_fetch_payload(api_key, corp_code, year, quarter))


def _fetch_payload(api_key: str, corp_code: str, year: int, quarter: int) -> dict:
    return _get_json(f"{BASE}/fnlttSinglAcnt.json", {
        "crtfc_key": api_key, "corp_code": corp_code,
        "bsns_year": str(year), "reprt_code": QUARTER_REPORT[quarter],
    })


def fetch_financials(
    api_key: str, corp_code: str, year: int, quarter: int,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Fetch net income + revenue + operating income (current & prior) in one call."""
    return parse_financials(_fetch_payload(api_key, corp_code, year, quarter))


def collect_keys() -> list[str]:
    """DART keys from env in priority order: ``DART_API_KEY``, ``DART_API_KEY_2/3/...``.

    Each key has its own 20,000-call/day quota (per-key, not per-IP), so listing
    several lets collection roll over to the next when one hits the daily cap.
    """
    keys: list[str] = []
    for name in ("DART_API_KEY", "DART_API_KEY_2", "DART_API_KEY_3", "DART_API_KEY_4"):
        v = os.environ.get(name)
        if v:
            keys.append(v)
    return keys


def _fetch_with_rotation(
    keys: list[str], ki: list[int], corp_code: str, year: int, quarter: int,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Fetch financials, rotating to the next key on DART daily-limit (status 020).

    ``ki`` is a one-element list holding the current key index, mutated in place so
    the rotation persists across calls (once a key is exhausted it stays skipped).
    """
    payload = _fetch_payload(keys[ki[0]], corp_code, year, quarter)
    while payload.get("status") == "020" and ki[0] + 1 < len(keys):
        ki[0] += 1
        print(f"DART 키 일한도(020) 도달 → 키{ki[0] + 1}로 로테이션", flush=True)
        payload = _fetch_payload(keys[ki[0]], corp_code, year, quarter)
    return parse_financials(payload)


def _write_row_csv(w: "csv.writer", code: str, period: str, avail: str,
                    ni: float | None, nip: float | None, rev: float | None,
                    revp: float | None, oi: float | None, oip: float | None) -> None:
    """첫 6컬럼(code,period,avail,netinc,prior,yoy)은 기존 스키마 불변 —
    매출·영업이익 4컬럼을 뒤에 append (하위호환: 기존 리더는 앞 6개만 읽음)."""
    def _c(x):
        return x if x is not None else ""
    w.writerow([code, period, avail, ni, _c(nip),
                _c(yoy_growth(ni, nip)), _c(rev), _c(revp), _c(oi), _c(oip)])


def _write_row_db(con: Any, code: str, period: str, avail: str,
                   ni: float | None, nip: float | None, rev: float | None,
                   revp: float | None, oi: float | None, oip: float | None) -> None:
    from ..storage import upsert_earnings
    upsert_earnings(con, [(code, period, avail, ni, nip, rev, revp, oi, oip)])


def _recent_quarters(n: int, today: datetime | None = None) -> list[tuple[int, int]]:
    """The N most recent (year, quarter) pairs counting back from the current quarter."""
    today = today or datetime.now()
    year = today.year
    q = (today.month - 1) // 3 + 1
    out: list[tuple[int, int]] = []
    for _ in range(n):
        out.append((year, q))
        q -= 1
        if q == 0:
            q = 4
            year -= 1
    return out


def _universe_query(args: argparse.Namespace) -> tuple[str, dict]:
    """SQL (+ params) selecting the code universe: all ``daily_bars`` codes or top-N liquid."""
    if args.all_codes:
        return "SELECT DISTINCT code FROM daily_bars ORDER BY code", {}
    return (
        "SELECT code FROM daily_bars WHERE date >= (SELECT MAX(date) FROM daily_bars) - INTERVAL '90 days' "
        "GROUP BY code ORDER BY AVG(trade_value) DESC LIMIT %(n)s",
        {"n": args.top_n},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="DART 분기 순이익 YoY 수집 (PEAD 입력)")
    ap.add_argument("--out", required=False, default=None, help="출력 CSV 경로")
    ap.add_argument("--top-n", type=int, default=800, help="유동성 상위 N종목")
    ap.add_argument("--from-year", type=int, default=2018)
    ap.add_argument("--to-year", type=int, default=datetime.now().year)
    ap.add_argument("--db", default=None)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--db-table", action="store_true", help="CSV 대신 earnings 테이블에 직접 upsert")
    ap.add_argument("--all-codes", action="store_true", help="유동성 상위 N 대신 daily_bars 전종목 사용")
    ap.add_argument("--recent-quarters", type=int, default=None,
                    help="전체 이력 대신 최근 N개 분기만 수집 (현재+직전, 일일 증분용)")
    args = ap.parse_args()
    if not args.db_table and not args.out:
        ap.error("--out is required unless --db-table is set")

    keys = collect_keys()
    if not keys:
        raise SystemExit("환경변수 DART_API_KEY 필요")
    ki = [0]  # 현재 키 인덱스 (020 한도 시 로테이션)

    import pandas as pd
    from ..storage import connect, default_db_path
    con = connect(args.db or str(default_db_path()))
    q_sql, q_params = _universe_query(args)
    top = pd.read_sql_query(q_sql, con, params=q_params)
    done_periods: set[tuple[str, str]] = set()
    if args.db_table:
        existing = pd.read_sql_query("SELECT code, period FROM earnings", con)
        done_periods = set(zip(existing["code"], existing["period"]))
    else:
        con.close()
    codes = top["code"].tolist()

    done: set[str] = set()
    if args.out and os.path.exists(args.out):
        for r in csv.reader(open(args.out)):
            if r:
                done.add(r[0])
    corp = load_corp_map_with_rotation(keys)
    print(f"corp_map {len(corp)} | universe {len(codes)} | keys {len(keys)} | already done {len(done)}", flush=True)

    today = datetime.now().strftime("%Y%m%d")
    f = open(args.out, "a", newline="") if args.out else None
    w = csv.writer(f) if f else None
    n = 0
    if args.recent_quarters is not None:
        periods = _recent_quarters(args.recent_quarters)
    else:
        periods = [(year, q) for year in range(args.from_year, args.to_year + 1) for q in (1, 2, 3, 4)]

    for i, code in enumerate(codes, 1):
        if not args.db_table and (code in done or code not in corp):
            continue
        if args.db_table and code not in corp:
            continue
        for year, q in periods:
            avail = available_date(f"{year}-{QUARTER_END[q][:2]}-{QUARTER_END[q][2:]}",
                                   is_annual=(q == 4)).strftime("%Y%m%d")
            if avail > today:
                continue
            period = f"{year}Q{q}"
            if args.db_table and (code, period) in done_periods:
                continue
            ni, nip, rev, revp, oi, oip = _fetch_with_rotation(keys, ki, corp[code], year, q)
            time.sleep(args.sleep)
            if ni is None:
                continue
            if args.db_table:
                _write_row_db(con, code, period, avail, ni, nip, rev, revp, oi, oip)
            else:
                _write_row_csv(w, code, period, avail, ni, nip, rev, revp, oi, oip)
            n += 1
        if f:
            f.flush()
        if i % 25 == 0:
            print(f"[{i}/{len(codes)}] rows={n}", flush=True)
    if f:
        f.close()
    if args.db_table:
        con.close()
    print(f"DONE rows={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
