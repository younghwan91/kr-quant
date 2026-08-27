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
INDEX_RET: dict = {}
DEFAULT_DAYS = 260        # 거래일 ≈ 1년
TOP_BY_TURNOVER = 8      # 섹터당 대표종목(거래대금 상위)
TOP_BY_FLOW = 4          # 섹터당 기관 순매수 절대값 상위 — 대형주가 아닌 실제 견인주
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


def attach_daily_cap(con, df: pd.DataFrame) -> pd.DataFrame:
    """종목별 일별 시총과 **전일** 시총을 붙인다 — 섹터 수익률의 가중치.

    거래대금 가중은 못 쓴다. 그날 급등한 종목이 거래대금도 크므로 가중치와 수익률이
    양의 상관을 갖고, 섹터 수익률이 체계적으로 부풀려진다(실측: 3종목짜리 부동산이
    20거래일 +102%). 표준은 **전일** 시총 가중이다 — 가중치가 수익률보다 먼저
    정해져야 편향이 안 생긴다.
    """
    key = df[["code", "date"]].drop_duplicates()
    key["cap"] = market_cap_asof_bulk(con, key).to_numpy() / EOK
    out = df.merge(key, on=["code", "date"], how="left").sort_values(["code", "date"])
    out["cap_lag"] = out.groupby("code")["cap"].shift(1)
    return out


def index_returns(con, dates: list[str], sectors, markets) -> dict:
    """KRX 업종지수의 일별 수익률 — 섹터 수익률의 **실측 기준선**.

    처음엔 종목 등락률을 섹터로 집계해 썼는데, 그러면 가중치 선택이 그대로 오차가
    된다(같은 날 거래대금 가중은 급등주에 가중치를 몰아 20거래일 +102% 같은 값을
    만들었고, 전일 시총 가중으로 바꿔도 20일은 정확했지만 60일에서 KRX 대비 중앙값
    17%p 벌어졌다). 벤더 지수가 DB 에 있으므로 그걸 쓴다.

    이름이 같은 코드가 둘이면 작은 코드가 거래소, 큰 코드가 코스닥이다. 코드가
    하나뿐인 업종(보험·증권 등 코스피 전용)은 거래소에만 붙인다. 지수가 아예 없는
    섹터는 ``None`` 으로 남겨, 소비자가 **섞지 않고 제외**할 수 있게 한다.
    """
    si = pd.read_sql_query("SELECT code, name, date, close FROM sector_index", con)
    si["date"] = si["date"].astype(str)
    lookup: dict = {m: {} for m in markets}
    for name, g in si.groupby("name"):
        codes = sorted(g["code"].unique())
        mapping = ({"거래소": codes[0], "코스닥": codes[1]} if len(codes) == 2
                   else {"거래소": codes[0]})
        for m, c in mapping.items():
            if m not in lookup:
                continue
            ser = (g[g["code"] == c].set_index("date")["close"]
                   .reindex(dates).astype(float).ffill())
            r = ser.pct_change() * 100.0
            lookup[m][name] = [None if pd.isna(v) else round(float(v), 4) for v in r]
    return {m: {s: lookup[m].get(s) for s in sectors} for m in markets}


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
    # **섹터마다** 대표종목이 나와야 한다. 전체 거래대금 상위 N 으로 자르면 대형
    # 섹터가 명단을 독점하고 작은 섹터는 종목이 하나도 안 나온다. 그래서 섹터별로
    # (거래대금 상위) ∪ (기관 순매수 절대값 상위) 를 뽑는다 — 앞은 그 섹터를
    # 대표하는 유동주, 뒤는 실제로 자금을 끌어당긴 견인주다(둘은 자주 다르다).
    per = df.groupby(["sector", "code"])[["tv", "inst"]].sum().reset_index()
    per["inst_abs"] = per["inst"].abs()
    top = pd.Index(sorted(set(
        per.groupby("sector", group_keys=False)
           .apply(lambda g: g.nlargest(TOP_BY_TURNOVER, "tv"), include_groups=False)["code"]
    ) | set(
        per.groupby("sector", group_keys=False)
           .apply(lambda g: g.nlargest(TOP_BY_FLOW, "inst_abs"), include_groups=False)["code"]
    )))
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

    # 참고용 자체 계산(전일 시총 가중). 화면의 수익률은 KRX 업종지수를 쓴다 —
    # 아래 값은 지수가 없는 섹터의 폴백이자 대조용이다.
    ret: dict = {m: {s: zeros() for s in sectors} for m in markets}
    rw: dict = {m: {s: zeros() for s in sectors} for m in markets}   # 그날 유효 가중합
    for (m, s, d), grp in df.groupby(["market", "sector", "date"], sort=False):
        if m not in ret:
            continue
        g = grp.dropna(subset=["cap_lag"])
        w = float(g["cap_lag"].sum())
        ret[m][s][di[d]] = round(float((g["ret"] * g["cap_lag"]).sum() / w) if w else 0.0, 3)
        rw[m][s][di[d]] = round(w, 1)

    # (시장, 섹터)별 실제 종목 수 — 표에서 "섹터로 읽을 만한가" 를 가리는 기준.
    counts = {m: {} for m in markets}
    for (m, sec), g in df.groupby(["market", "sector"], sort=False):
        if m in counts:
            counts[m][sec] = int(g["code"].nunique())

    # (시장, 섹터)별 실제 종목 수 — 표에서 "섹터로 읽을 만한가" 를 가리는 기준.
    counts = {m: {} for m in markets}
    for (m, sec), g in df.groupby(["market", "sector"], sort=False):
        if m in counts:
            counts[m][sec] = int(g["code"].nunique())

    return {
        "dates": dates,
        "sectors": sectors,
        "n_by_sector": counts,
        "n_by_sector": counts,
        "markets": markets,
        "flows": flows,
        "detail": detail,
        "ret": ret,
        "retw": rw,
        "iret": INDEX_RET,
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
    df = attach_daily_cap(con, df)
    caps = sector_cap(con, df, dates[-1])
    global INDEX_RET
    INDEX_RET = index_returns(con, dates, sorted(df["sector"].unique()),
                              sorted(df["market"].dropna().unique()))
    con.close()
    payload = build_payload(df, dates, caps)
    covered = sum(1 for m in payload["markets"] for s in payload["sectors"]
                  if payload["iret"][m].get(s))
    print(f"업종지수 커버리지: {covered} / "
          f"{len(payload['markets']) * len(payload['sectors'])} (시장×섹터)")

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
