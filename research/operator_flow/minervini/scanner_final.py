"""최종 미너비니 규칙 시스템 — 오늘의 진입 후보 스캐너 (순수 규칙, ML 불필요).

GOAL 루프 1-13에서 수렴한 시스템:
  수급 스텔스매집(60일, informed 5투자자 앙상블, 하위30% 제외)
  + 미너비니 추세템플릿(종가>MA50>MA150>MA200, MA200 우상향, 52주저점+25%, 52주고점-25%내)
  + 100억↑ 주도주 유동성
  + 20일 종가 돌파 + 거래량 수축(vc<1)
  → 5% 손절, 장기보유, 2% 리스크 사이징(하프켈리 ≈ 종목당 40% 자본), breadth 레짐 스위치.

시장 breadth(유동주 close>MA50 비율)로 위험-온/오프를 판정 — 위험-오프면 신규진입 중단.
CLI: python scanner_final.py --db <DSN>
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

INF = ["institution", "foreign_", "etc_corp", "invtrt", "fnnc_invt"]


def scan(con, *, adv_floor=10000.0, top_frac=0.70, lookback=450):
    cols = ",".join(INF)
    sd = pd.read_sql(
        f"SELECT code,date,{cols} FROM supply_demand "
        f"WHERE date > (SELECT MAX(date) FROM supply_demand) - INTERVAL '{lookback} days'", con)
    db = pd.read_sql(
        f"SELECT code,date,high,low,close,volume,trade_value FROM daily_bars "
        f"WHERE date > (SELECT MAX(date) FROM daily_bars) - INTERVAL '{lookback} days'", con)
    sd["date"] = sd["date"].astype(str); db["date"] = db["date"].astype(str)
    df = db.merge(sd, on=["code", "date"], how="inner").sort_values(["code", "date"])
    dates = sorted(df["date"].unique())
    asof = dates[-1]; ti = len(dates) - 1
    if ti < 252:
        raise ValueError("need >=252 trading days of history")

    rows = []
    liq_above50 = []  # breadth 계산용
    for code, sub in df.groupby("code"):
        sub = sub.set_index("date").reindex(dates)
        c = sub["close"].to_numpy(float)
        if not np.isfinite(c[ti]):
            continue
        vol = sub["volume"].to_numpy(float); tval = sub["trade_value"].to_numpy(float)
        high = sub["high"].to_numpy(float)
        adv = np.nanmean(tval[ti - 20:ti])
        ma50 = np.nanmean(c[ti - 49:ti + 1])
        if np.isfinite(adv) and adv >= 3000:  # breadth 유니버스(30억)
            liq_above50.append(1.0 if c[ti] > ma50 else 0.0)
        if not np.isfinite(adv) or adv < adv_floor:  # 100억 주도주만 진입후보
            continue
        ma150 = np.nanmean(c[ti - 149:ti + 1]); ma200 = np.nanmean(c[ti - 199:ti + 1])
        ma200_prev = np.nanmean(c[ti - 220:ti - 20])
        hh252 = np.nanmax(high[ti - 251:ti + 1]); ll252 = np.nanmin(sub["low"].to_numpy(float)[ti - 251:ti + 1])
        # 추세템플릿
        tt = (c[ti] > ma50 > ma150 > ma200 and ma200 > ma200_prev
              and c[ti] >= 1.25 * ll252 and c[ti] >= 0.75 * hh252)
        if not tt:
            continue
        # 돌파 + 거래량 수축
        prior_high = np.nanmax(high[ti - 20:ti])
        breakout = np.isfinite(prior_high) and c[ti] > prior_high
        vol20 = np.nansum(vol[ti - 20:ti]); vol60 = np.nansum(vol[ti - 60:ti])
        vprior = (vol60 - vol20) / 40.0
        vc = (vol20 / 20.0) / vprior if vprior > 0 else np.nan
        if not (breakout and np.isfinite(vc) and vc < 1.0):
            continue
        # 스텔스 매집(60일 순매수/거래량, 5투자자 합)
        nb60 = sum(np.nansum(sub[inv].to_numpy(float)[ti - 60:ti]) for inv in INF)
        accum = nb60 / vol60 if vol60 > 0 else np.nan
        mom60 = c[ti - 1] / c[ti - 61] - 1 if np.isfinite(c[ti - 61]) else np.nan
        rows.append({"code": code, "close": c[ti], "accum": accum, "mom": mom60, "adv_억": adv / 100})
    if not rows:
        return asof, np.mean(liq_above50) if liq_above50 else np.nan, pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["accum", "mom"])
    if out.empty:
        return asof, np.mean(liq_above50), out
    out["stealth"] = out["accum"].rank(pct=True) - out["mom"].rank(pct=True)
    thr = out["stealth"].quantile(1 - top_frac)  # 하위30% 제외
    cand = out[out["stealth"] >= thr].copy().sort_values("stealth", ascending=False)
    cand["entry"] = cand["close"].round(0)
    cand["stop_5%"] = (cand["close"] * 0.95).round(0)
    cand["size_2%risk"] = "40% 자본"  # 2%리스크 / 5%손절 = 40%; 실전 ≤50종목 분산
    breadth = np.mean(liq_above50) if liq_above50 else np.nan
    return asof, breadth, cand[["code", "close", "entry", "stop_5%", "stealth", "adv_억"]].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="최종 미너비니 규칙 시스템 스캐너")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    from kr_quant.storage import connect, default_db_path
    con = connect(args.db or str(default_db_path()))
    asof, breadth, cand = scan(con)
    con.close()
    regime = "위험-온 (진입 가능)" if breadth > 0.5 else "위험-오프 (신규진입 중단, 현금)"
    print(f"[{asof}] 시장 breadth={breadth:.0%} → {regime}")
    if breadth <= 0.5:
        print("※ 약세장 레짐 — 미너비니 규칙상 신규진입 자제/현금.")
    if cand.empty:
        print("오늘 진입 후보 없음 (선택적 시스템).")
        return 0
    print(f"진입 후보 {len(cand)}종목 (수급스텔스+추세템플릿+100억+돌파, 5%손절):")
    print(cand.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
