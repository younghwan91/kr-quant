#!/usr/bin/env python
"""섹터별 자금흐름 — 오늘 국장에서 어느 업종으로 돈이 들어갔나.

DB 읽기 전용. 수급 정문(:func:`kr_quant.storage.read_supply_demand`)으로 순매매량을
읽어 **종가를 곱해 금액으로 환산**한 뒤 `stocks.sector` 로 묶는다.

⚠️ **순매매대금은 근사다.** DB 가 주는 것은 순매매 *수량*이라, 금액은 종가를 곱해
복원한다. 실제 체결은 하루 종일 여러 가격에 일어나므로 참값은 VWAP 가중이다.
방향과 상대 크기를 보는 용도이며, 원 단위 정확도를 주장하지 않는다.

Run:  python scripts/sector_flow.py               # 마크다운 요약
      python scripts/sector_flow.py --json out.json
      python scripts/sector_flow.py --date 2026-08-26
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from kr_quant.storage import connect, db_default, read_supply_demand

EOK = 1e8  # 원 → 억원
#: 기관 세부. ``natn``(국가)은 90일 내내 전부 0 이라 뺀다(quant-airflow 실측).
INST_DETAIL = ("fnnc_invt", "insrnc", "invtrt", "bank", "penfnd_etc", "samo_fund")
DETAIL_KO = {"fnnc_invt": "금투", "insrnc": "보험", "invtrt": "투신", "bank": "은행",
             "penfnd_etc": "연기금", "samo_fund": "사모"}
FINALIZED_KST_HOUR = 17  # 16:00 DAG + ~48분 → 그날 수급은 16:50 경 확정


def latest_date(con) -> str:
    cur = con.cursor()
    cur.execute("SELECT max(date) FROM supply_demand")
    return str(cur.fetchone()[0])


def load(con, date: str) -> pd.DataFrame:
    """그 날짜의 종목별 수급 + 섹터 + 종가·등락률."""
    sd = read_supply_demand(
        con,
        cols=("code", "date", "close", "flu_rt", "acc_trde_qty",
              "individual", "foreign_", "institution", "etc_corp") + INST_DETAIL,
        start=date, end=date,
        # 개인·기관세부는 키움에만 있다. 소스를 안 좁히면 지표마다 유니버스가 달라진다
        # (institution 은 폐지분 포함, individual 은 NULL 로 탈락) — 분모가 어긋난다.
        sources=("kiwoom",),
        allow_individual_survivorship=True,
        require_delisted=False,
    )
    meta = pd.read_sql_query("SELECT code, name, sector, market FROM stocks", con)
    df = sd.merge(meta, on="code", how="inner")
    df["close"] = df["close"].abs()
    # flu_rt 는 등락률 × 100 (bp) 이다 — 삼성전자 175 = +1.75% (전일 257,000 → 261,500).
    df["ret_pct"] = df["flu_rt"].astype(float) / 100.0

    for col in ("individual", "foreign_", "institution", "etc_corp") + INST_DETAIL:
        df[f"{col}_eok"] = df[col].astype(float) * df["close"] / EOK
    df["turnover_eok"] = df["acc_trde_qty"].astype(float) * df["close"] / EOK
    return df


def by_sector(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sector")
    out = pd.DataFrame({
        "n": g.size(),
        "turnover_eok": g["turnover_eok"].sum(),
        "inst_eok": g["institution_eok"].sum(),
        "foreign_eok": g["foreign__eok"].sum(),
        "indiv_eok": g["individual_eok"].sum(),
        "etc_eok": g["etc_corp_eok"].sum(),
    })
    # 등락률은 거래대금 가중 — 대형주가 섹터를 대표하게 한다.
    w = df.groupby("sector").apply(
        lambda x: (x["ret_pct"] * x["turnover_eok"]).sum()
        / max(x["turnover_eok"].sum(), 1e-9),
        include_groups=False)
    out["ret_pct"] = w
    for c in INST_DETAIL:
        out[f"{c}_eok"] = g[f"{c}_eok"].sum()
    out["inst_share"] = out["inst_eok"] / out["turnover_eok"].replace(0.0, float("nan"))
    return out.sort_values("inst_eok", ascending=False).reset_index()


def top_names(df: pd.DataFrame, sector: str, k: int = 5) -> list[dict]:
    """섹터 안에서 기관이 가장 많이 담은 종목(그리고 가장 많이 던진 종목)."""
    s = df[df["sector"] == sector]
    buys = s.nlargest(k, "institution_eok")
    sells = s.nsmallest(k, "institution_eok")
    def rows(x):
        return [{"code": r.code, "name": r["name"],
                 "inst_eok": round(float(r.institution_eok), 1),
                 "turnover_eok": round(float(r.turnover_eok), 1),
                 "ret_pct": round(float(r.ret_pct), 2)}
                for _, r in x.iterrows()]
    return {"buys": rows(buys), "sells": rows(sells)}


def _finalized(date: str) -> bool:
    """그 날짜 수급이 확정됐나 — 오늘이면 16:50(KST) 이후여야 한다.

    quant-airflow 의 16:00 DAG 가 전 종목을 다시 덮으므로 그 전에는 부분일 수 있다.
    catchup(10:05) 은 뒤처진 종목만 건드리지만 상한이 없어 장중 행이 들어올 수 있다.
    """
    from datetime import datetime, timedelta, timezone
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    return date != now_kst.strftime("%Y-%m-%d") or now_kst.hour >= FINALIZED_KST_HOUR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("KR_QUANT_DB") or db_default())
    ap.add_argument("--date")
    ap.add_argument("--json")
    ap.add_argument("--html", help="뷰어 HTML 생성 경로")
    a = ap.parse_args()

    con = connect(a.db)
    date = a.date or latest_date(con)
    df = load(con, date)
    con.close()

    sec = by_sector(df)
    payload = {
        "date": date,
        "finalized": _finalized(date),
        "n_names": int(len(df)),
        "total_turnover_eok": round(float(df["turnover_eok"].sum()), 1),
        "total_inst_eok": round(float(df["institution_eok"].sum()), 1),
        "total_foreign_eok": round(float(df["foreign__eok"].sum()), 1),
        "total_indiv_eok": round(float(df["individual_eok"].sum()), 1),
        "total_etc_eok": round(float(df["etc_corp_eok"].sum()), 1),
        "sectors": [
            {**{k: (round(float(v), 2) if isinstance(v, float) else v)
                for k, v in row.items()},
             "top": top_names(df, row["sector"])}
            for row in sec.to_dict("records")
        ],
    }

    if a.html:
        tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "templates", "sector_flow.html")
        with open(tpl, encoding="utf-8") as f:
            html = f.read()
        html = html.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"wrote {a.html}  ({date}, {len(df)}종목, 확정={payload['finalized']})")
        return

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"wrote {a.json}  ({date}, {len(df)}종목)")
        return

    print(f"# 섹터 자금흐름 — {date} ({len(df)}종목)\n")
    print(f"거래대금 {payload['total_turnover_eok']:,.0f}억 · "
          f"기관 {payload['total_inst_eok']:+,.0f}억 · "
          f"외국인 {payload['total_foreign_eok']:+,.0f}억 · "
          f"개인 {payload['total_indiv_eok']:+,.0f}억 · "
          f"기타법인 {payload['total_etc_eok']:+,.0f}억\n")
    print("| 섹터 | 종목 | 거래대금(억) | 기관(억) | 외국인(억) | 개인(억) | 등락(%) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in sec.itertuples():
        print(f"| {r.sector} | {r.n} | {r.turnover_eok:,.0f} | {r.inst_eok:+,.0f} | "
              f"{r.foreign_eok:+,.0f} | {r.indiv_eok:+,.0f} | {r.ret_pct:+.2f} |")
    if not payload["finalized"]:
        print("\n⚠️ 오늘 수급은 아직 확정 전이다(16:50 KST 이후 확정) — 부분일 수 있다.")


if __name__ == "__main__":
    main()
