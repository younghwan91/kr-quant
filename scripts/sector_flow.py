#!/usr/bin/env python
"""섹터 자금흐름 — 어느 시점부터 어느 장에서 어느 섹터로 돈이 흘러갔나.

DB 읽기 전용. 수급 정문(:func:`kr_quant.storage.read_supply_demand`)으로 순매매
**수량**을 읽어 그날 종가를 곱해 금액으로 환산하고, `stocks` 의 섹터·시장으로 묶어
**일별 시계열**을 낸다. 하루치만 보면 블록딜 한 건이 섹터를 통째로 흔들어 보이므로,
임의 구간 누적이 기본 단위다.

세 가지 관점을 같이 낸다:

* **금액 누적** — 구간 동안 그 섹터로 순유입된 금액(억원).
* **비중 변화** — 누적 순매수금액 ÷ **기간말 섹터 시가총액**. 기관 지분율이 몇 %p
  올랐나에 대한 근사다. 금액만 보면 대형 섹터가 늘 이기므로 이쪽이 "어디로 쏠렸나"를
  더 정직하게 준다.
* **시장 분리** — 거래소/코스닥은 주체 구성이 달라 섞으면 신호가 상쇄된다.

⚠️ **순매매대금은 종가 환산 근사다.** DB 는 수량만 주므로 금액은 종가를 곱해 복원한다.
참값은 VWAP 가중이다 — 방향과 상대 크기를 보는 용도이며 원 단위 정확도를 주장하지 않는다.

Run:  python scripts/sector_flow.py --html viewer.html
      python scripts/sector_flow.py --days 120 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from kr_quant.storage import (
    connect, db_default, market_cap_asof_bulk, read_supply_demand)

EOK = 1e8                 # 원 → 억원
DEFAULT_DAYS = 260        # 거래일 ≈ 1년
TOP_NAMES = 200           # 종목 드릴다운에 실어보낼 종목 수(거래대금 상위)
FINALIZED_KST_HOUR = 17   # 16:00 DAG + ~48분 → 그날 수급은 16:50 경 확정

ACTORS = (("individual", "indiv"), ("foreign_", "forgn"),
          ("institution", "inst"), ("etc_corp", "etc"))
#: 기관 세부. ``natn``(국가)은 90일 내내 전부 0 이라 뺀다(quant-airflow 실측).
INST_DETAIL = (("fnnc_invt", "금투"), ("insrnc", "보험"), ("invtrt", "투신"),
               ("bank", "은행"), ("penfnd_etc", "연기금"), ("samo_fund", "사모"))


def _finalized(date: str) -> bool:
    """그 날짜 수급이 확정됐나 — 오늘이면 16:50(KST) 이후여야 한다."""
    now = datetime.now(timezone(timedelta(hours=9)))
    return date != now.strftime("%Y-%m-%d") or now.hour >= FINALIZED_KST_HOUR


def load(con, days: int) -> tuple[pd.DataFrame, list[str]]:
    cur = con.cursor()
    cur.execute("SELECT DISTINCT date FROM supply_demand ORDER BY date DESC LIMIT %s"
                if hasattr(con, "cursor") and con.__class__.__module__.startswith("psycopg2")
                else "SELECT DISTINCT date FROM supply_demand ORDER BY date DESC LIMIT ?",
                (days,))
    dates = sorted(str(r[0]) for r in cur.fetchall())

    sd = read_supply_demand(
        con,
        cols=("code", "date", "close", "flu_rt", "acc_trde_qty",
              "individual", "foreign_", "institution", "etc_corp")
             + tuple(c for c, _ in INST_DETAIL),
        start=dates[0], end=dates[-1],
        # 개인·기관세부는 키움에만 있다. 소스를 안 좁히면 지표마다 유니버스가 달라진다.
        sources=("kiwoom",), allow_individual_survivorship=True, require_delisted=False,
    )
    sd["date"] = sd["date"].astype(str)
    meta = pd.read_sql_query("SELECT code, name, sector, market FROM stocks", con)
    df = sd.merge(meta, on="code", how="inner")
    df["close"] = df["close"].abs()
    df["sector"] = df["sector"].fillna("").replace("", "(미분류)")

    for raw, short in ACTORS:
        df[short] = df[raw].astype(float) * df["close"] / EOK
    for raw, _ in INST_DETAIL:
        df[raw] = df[raw].astype(float) * df["close"] / EOK
    df["tv"] = df["acc_trde_qty"].astype(float) * df["close"] / EOK
    df["ret"] = df["flu_rt"].astype(float) / 100.0   # flu_rt 는 등락률 × 100 (bp)
    return df, dates


def sector_cap(con, df: pd.DataFrame, last_date: str) -> pd.DataFrame:
    """기간말 섹터 시가총액(억) — 비중 변화의 분모."""
    codes = df[["code", "sector", "market"]].drop_duplicates("code").copy()
    codes["date"] = last_date
    codes["cap"] = market_cap_asof_bulk(con, codes[["code", "date"]]).to_numpy() / EOK
    return codes.dropna(subset=["cap"])


def build_payload(df: pd.DataFrame, dates: list[str], caps: pd.DataFrame) -> dict:
    sectors = sorted(df["sector"].unique())
    markets = sorted(df["market"].dropna().unique())
    di = {d: i for i, d in enumerate(dates)}
    n = len(dates)

    def zeros():
        return [0.0] * n

    flows: dict = {m: {s: {k: zeros() for k in ("indiv", "forgn", "inst", "etc", "tv")}
                       for s in sectors} for m in markets}
    g = df.groupby(["market", "sector", "date"], sort=False)[
        ["indiv", "forgn", "inst", "etc", "tv"]].sum()
    for (m, s, d), row in g.iterrows():
        if m not in flows:
            continue
        i = di[d]
        for k in ("indiv", "forgn", "inst", "etc", "tv"):
            flows[m][s][k][i] = round(float(row[k]), 2)

    detail: dict = {m: {s: {k: zeros() for k, _ in INST_DETAIL} for s in sectors}
                    for m in markets}
    gd = df.groupby(["market", "sector", "date"], sort=False)[
        [k for k, _ in INST_DETAIL]].sum()
    for (m, s, d), row in gd.iterrows():
        if m not in detail:
            continue
        i = di[d]
        for k, _ in INST_DETAIL:
            detail[m][s][k][i] = round(float(row[k]), 2)

    cap = {m: {s: round(float(v), 1) for s, v in
               caps[caps["market"] == m].groupby("sector")["cap"].sum().items()}
           for m in markets}

    # 종목 드릴다운 — 구간이 임의라 클라이언트가 합산할 수 있게 일별로 실어보낸다.
    top = (df.groupby("code")["tv"].sum().nlargest(TOP_NAMES).index)
    nd = df[df["code"].isin(top)]
    names: dict = {}
    meta = nd.drop_duplicates("code").set_index("code")[["name", "sector", "market"]]
    for code, grp in nd.groupby("code", sort=False):
        row = meta.loc[code]
        rec = {"name": row["name"], "sector": row["sector"], "market": row["market"]}
        for k in ("indiv", "forgn", "inst", "etc", "tv"):
            arr = zeros()
            for d, v in zip(grp["date"], grp[k]):
                arr[di[d]] = round(float(v), 2)
            rec[k] = arr
        names[code] = rec

    # 섹터 등락률(거래대금 가중) — 일별
    ret: dict = {m: {s: zeros() for s in sectors} for m in markets}
    for (m, s, d), grp in df.groupby(["market", "sector", "date"], sort=False):
        if m not in ret:
            continue
        w = grp["tv"].sum()
        ret[m][s][di[d]] = round(float((grp["ret"] * grp["tv"]).sum() / w) if w else 0.0, 3)

    return {
        "dates": dates,
        "sectors": sectors,
        "markets": markets,
        "flows": flows,
        "detail": detail,
        "ret": ret,
        "cap": cap,
        "names": names,
        "n_names": int(df["code"].nunique()),
        "finalized": _finalized(dates[-1]),
        "detail_labels": {k: ko for k, ko in INST_DETAIL},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("KR_QUANT_DB") or db_default())
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--json")
    ap.add_argument("--html")
    a = ap.parse_args()

    con = connect(a.db)
    df, dates = load(con, a.days)
    caps = sector_cap(con, df, dates[-1])
    con.close()
    payload = build_payload(df, dates, caps)

    if a.html:
        tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "templates", "sector_flow.html")
        with open(tpl, encoding="utf-8") as f:
            html = f.read()
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(html.replace("__DATA__", json.dumps(payload, ensure_ascii=False)))
        print(f"wrote {a.html}  ({dates[0]} ~ {dates[-1]}, {len(dates)}거래일, "
              f"{payload['n_names']}종목, 확정={payload['finalized']})")
        return

    out = a.json or "-"
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"wrote {out}  ({dates[0]} ~ {dates[-1]}, {len(dates)}거래일)")
        return

    last = len(dates) - 1
    print(f"# 섹터 자금흐름 — {dates[0]} ~ {dates[-1]} ({len(dates)}거래일)\n")
    print("| 섹터 | 시장 | 기관누적(억) | 비중변화(%p) | 거래대금(억) |")
    print("|---|---|---:|---:|---:|")
    rows = []
    for m in payload["markets"]:
        for s in payload["sectors"]:
            f = payload["flows"][m][s]
            tot = sum(f["inst"])
            cap = payload["cap"].get(m, {}).get(s, 0.0)
            rows.append((s, m, tot, (tot / cap * 100) if cap else 0.0, sum(f["tv"])))
    for s, m, tot, sh, tv in sorted(rows, key=lambda x: -x[3])[:15]:
        print(f"| {s} | {m} | {tot:+,.0f} | {sh:+.2f} | {tv:,.0f} |")


if __name__ == "__main__":
    main()
