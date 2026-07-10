"""오늘의 세력 스텔스 매집 + 돌파 후보 스캐너 (실행 가능한 신호).

검증된 신호(research/operator_flow/README.md)를 최신 거래일에 적용해, **오늘
진입 후보**와 ATR 기반 손절/익절 레벨을 뽑는다. 개별주 스윙용.

- 유니버스: 최근 20일 평균 거래대금 ≥ adv_floor(백만원).
- 매집강도 = 최근20일 순매수량 / 최근20일 거래량 (기본 투자자: 기타법인).
- 스텔스 = rank(매집강도) − rank(최근20일 수익)  (추격매수 제거, 숨은수요만).
- 진입 = 스텔스 상위 top_frac AND 종가 > 직전 20일 고가(돌파).
- 손절 = 종가 − k·ATR(14) / 반절 = +half_R·R / 러너 = +run_R·R  (R=k·ATR).

CLI: python scanner.py --db <DSN> [--investor etc_corp] [--asof YYYY-MM-DD]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

INVESTORS = {
    "기관": "institution", "외국인": "foreign_", "개인": "individual",
    "기타법인": "etc_corp", "연기금": "penfnd_etc", "투신": "invtrt",
    "금융투자": "fnnc_invt", "사모": "samo_fund",
}


def scan(con, *, investor: str = "etc_corp", asof: str | None = None,
         adv_floor: float = 3000.0, top_frac: float = 0.30,
         katr: float = 3.0, half_r: float = 3.0, run_r: float = 12.0,
         lookback_days: int = 90) -> pd.DataFrame:
    """Return today's operator-accumulation breakout candidates with exit levels."""
    sd = pd.read_sql(
        f"SELECT code,date,{investor} AS nb FROM supply_demand "
        f"WHERE date > (SELECT MAX(date) FROM supply_demand) - INTERVAL '{lookback_days} days'",
        con)
    db = pd.read_sql(
        f"SELECT code,date,high,low,close,volume,trade_value FROM daily_bars "
        f"WHERE date > (SELECT MAX(date) FROM daily_bars) - INTERVAL '{lookback_days} days'",
        con)
    sd["date"] = sd["date"].astype(str)
    db["date"] = db["date"].astype(str)
    df = db.merge(sd, on=["code", "date"], how="inner").sort_values(["code", "date"])
    dates = sorted(df["date"].unique())
    asof = asof or dates[-1]
    if asof not in dates:
        raise ValueError(f"{asof} not in data (latest {dates[-1]})")
    ti = dates.index(asof)
    if ti < 34:
        raise ValueError("not enough history in lookback window")

    g = df.groupby("code")
    rows = []
    for code, sub in g:
        sub = sub.set_index("date").reindex(dates)
        c = sub["close"].to_numpy(float)
        if not np.isfinite(c[ti]):
            continue
        vol = sub["volume"].to_numpy(float)
        tval = sub["trade_value"].to_numpy(float)
        nb = sub["nb"].to_numpy(float)
        high = sub["high"].to_numpy(float)
        low = sub["low"].to_numpy(float)
        adv = np.nanmean(tval[ti - 20:ti])
        if not np.isfinite(adv) or adv < adv_floor:
            continue
        volsum = np.nansum(vol[ti - 20:ti])
        if volsum <= 0:
            continue
        accum = np.nansum(nb[ti - 20:ti]) / volsum
        mom = c[ti - 1] / c[ti - 21] - 1 if np.isfinite(c[ti - 21]) else np.nan
        prior_high = np.nanmax(high[ti - 20:ti])
        # ATR(14)
        tr = np.maximum.reduce([
            high[ti - 14:ti] - low[ti - 14:ti],
            np.abs(high[ti - 14:ti] - c[ti - 15:ti - 1]),
            np.abs(low[ti - 14:ti] - c[ti - 15:ti - 1]),
        ])
        atr = np.nanmean(tr)
        breakout = np.isfinite(prior_high) and c[ti] > prior_high
        if not (np.isfinite(accum) and np.isfinite(mom) and np.isfinite(atr) and atr > 0):
            continue
        rows.append({"code": code, "accum": accum, "mom": mom, "adv": adv,
                     "atr": atr, "close": c[ti], "breakout": breakout})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["stealth"] = out["accum"].rank(pct=True) - out["mom"].rank(pct=True)
    thr = out["stealth"].quantile(1 - top_frac)
    cand = out[(out["stealth"] >= thr) & out["breakout"]].copy()
    r = katr * cand["atr"]
    cand["entry"] = cand["close"].round(0)
    cand["stop"] = (cand["close"] - r).round(0)
    cand["half_tgt"] = (cand["close"] + half_r * r).round(0)
    cand["run_tgt"] = (cand["close"] + run_r * r).round(0)
    cand["adv_억"] = (cand["adv"] / 100).round(0)
    return cand.sort_values("stealth", ascending=False)[
        ["code", "stealth", "close", "entry", "stop", "half_tgt", "run_tgt", "adv_억"]
    ].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="오늘의 세력 매집+돌파 후보 스캐너")
    ap.add_argument("--db", default=None)
    ap.add_argument("--investor", default="etc_corp", choices=list(INVESTORS.values()))
    ap.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD (기본 최신)")
    ap.add_argument("--adv-floor", type=float, default=3000.0)
    ap.add_argument("--top-frac", type=float, default=0.30)
    args = ap.parse_args()
    from kr_quant.storage import connect, default_db_path
    con = connect(args.db or str(default_db_path()))
    res = scan(con, investor=args.investor, asof=args.asof,
               adv_floor=args.adv_floor, top_frac=args.top_frac)
    con.close()
    if res.empty:
        print("오늘 조건을 만족하는 후보 없음")
        return 0
    print(f"세력 매집+돌파 후보 ({args.investor}, {len(res)}종목) — 진입/손절/반절/러너:")
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
