"""RBA 추적기 — 스캐너 픽의 실제 실현 결과를 기록 (미너비니 최종 조언).

미너비니: "이론적 가정(TBA)이 아니라 실제 매매 결과 데이터(RBA)로 리스크를 설계하라."
daily_minervini_scan DAG가 쌓은 minervini_scan.csv(날짜별 진입후보)를 읽고, 각 픽에 대해
진입 다음 거래일 시가 기준 5% 손절 / +10%(2R) 목표 / 20일 이내 결과를 판정해
minervini_rba.csv에 누적한다. 축적되면 실전 승률·기대값이 백테스트와 일치하는지 검증 가능.

CLI: python rba_tracker.py --db <DSN> --scan-csv <path> --out <path>
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

HARD = 0.05  # 5% 손절
TARGET = 0.10  # +10% = 2R
HMAX = 20  # 최대 4주


def evaluate(con, picks_by_date: dict[str, list[str]], already: set) -> list[list]:
    """각 (date, code) 픽의 실현 결과 판정. 충분한 forward 데이터가 있는 것만."""
    codes = sorted({c for cs in picks_by_date.values() for c in cs})
    if not codes:
        return []
    px = pd.read_sql(
        "SELECT code,date,open,high,low,close FROM daily_bars WHERE code = ANY(%(c)s) "
        "AND date > (SELECT MIN(date) FROM daily_bars WHERE date >= %(d0)s)",
        con, params={"c": codes, "d0": min(picks_by_date)})
    px["date"] = px["date"].astype(str)
    out = []
    for pick_date, cs in picks_by_date.items():
        for code in cs:
            key = f"{pick_date}:{code}"
            if key in already:
                continue
            g = px[px["code"] == code].sort_values("date")
            fwd = g[g["date"] > pick_date].head(HMAX + 1)
            if len(fwd) < HMAX:  # 아직 결과 미확정 → 스킵(다음 실행에)
                continue
            entry = fwd.iloc[0]["open"]  # 다음날 시가 진입
            if not np.isfinite(entry) or entry <= 0:
                continue
            stop = entry * (1 - HARD); tgt = entry * (1 + TARGET)
            outcome = "open"; exit_px = fwd.iloc[-1]["close"]; days = HMAX
            for k in range(1, len(fwd)):
                lo, hi = fwd.iloc[k]["low"], fwd.iloc[k]["high"]
                if np.isfinite(lo) and lo <= stop:
                    outcome = "stop"; exit_px = stop; days = k; break
                if np.isfinite(hi) and hi >= tgt:
                    outcome = "target_2R"; exit_px = tgt; days = k; break
            ret = exit_px / entry - 1
            out.append([pick_date, code, round(entry), round(exit_px), outcome,
                        round(ret * 100, 1), days])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RBA 추적기 — 스캐너 픽 실현결과 기록")
    ap.add_argument("--db", default=None)
    ap.add_argument("--scan-csv", default="/opt/kr-quant/data/minervini_scan.csv")
    ap.add_argument("--out", default="/opt/kr-quant/data/minervini_rba.csv")
    args = ap.parse_args()
    if not os.path.exists(args.scan_csv):
        print("스캔 로그 없음(DAG 미실행) — RBA 축적 대기"); return 0
    picks = {}
    for row in csv.reader(open(args.scan_csv)):
        if len(row) >= 5 and row[2] == "risk_on" and row[4]:
            picks[row[0]] = row[4].split(",")
    already = set()
    if os.path.exists(args.out):
        for r in csv.reader(open(args.out)):
            if len(r) >= 2:
                already.add(f"{r[0]}:{r[1]}")
    from kr_quant.storage import connect, default_db_path
    con = connect(args.db or str(default_db_path()))
    rows = evaluate(con, picks, already)
    con.close()
    if rows:
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)
    # 누적 RBA 요약
    if os.path.exists(args.out):
        df = pd.read_csv(args.out, header=None,
                         names=["date", "code", "entry", "exit", "outcome", "ret%", "days"])
        wins = (df["outcome"] == "target_2R").sum(); n = len(df)
        if n:
            wr = wins / n
            print(f"RBA 누적: {n}건, 2R승률 {wr:.0%}, 평균수익 {df['ret%'].mean():+.1f}%, "
                  f"기대값 {3*wr-1:+.2f}R (백테스트 base 43% 대조)")
    print(f"신규 판정 {len(rows)}건 기록")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
