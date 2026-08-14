"""가격 시계열 기업행동(액면분할·무상증자) 조정 유틸리티.

## 왜 필요한가 (GOAL 루프 54-59에서 진단)

`daily_bars`의 가격은 기업행동이 **일관되게 조정되어 있지 않다**. 미조정 불연속을
전수 조사한 결과:

- 종가 대비 ±30%(한국 일일 가격제한)를 넘는 일간 변동 855건 발견 — 물리적으로
  불가능하므로 전부 데이터 아티팩트(분할/기준변경).
- 그 중 **93%가 정확히 3개 날짜(2019-02-27, 2021-07-29, 2024-01-08)에 군집** —
  서로 무관한 수백 종목이 같은 날 점프. 실제 분할은 이렇게 몰리지 않는다. 이는
  데이터가 여러 배치로 조립되며 배치 간 조정 기준이 어긋난 **수집-경계 불일치**다.
- 나머지 ~7%(≈44개 고립 날짜)가 진짜 개별 분할.

미조정 상태로 백테스트하면 분할이 가짜 −68% 손실(또는 가짜 이익)로 잡혀 절대수익률이
왜곡된다(리더 시스템 CAGR이 미조정 +20.9% → 조정 +14.0%로, 인덱스 대비 우위가 소멸).

## 무엇을 하는가

각 종목의 종가 시계열에서 ±30%를 넘고 **다음 3거래일간 새 레벨에 머무는(되돌림
스파이크 제외)** 불연속을 분할로 판정하고, 그 비율로 **이전 구간 전체를 백조정**해
연속적인 시계열을 만든다(표준 back-adjust). OHLC 전부에 동일 배수를 적용한다.

거래량은 분할 시 역으로 스케일되지만(가격 1/4 → 거래량 ×4), 대부분의 신호는 가격
기반이므로 여기서는 가격만 조정한다(거래량 조정이 필요하면 `adjust_volume=True`).

CLI: `python -m kr_quant.price_adjust --db <DSN>` 로 진단 리포트 출력.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LIMIT = 0.30  # 한국 일일 가격제한 ±30% — 이를 넘는 종가변동은 데이터 아티팩트
DOWN = 1 - LIMIT  # 0.70
UP = 1 / DOWN  # ≈1.4286 (역분할/기준상향)
PERSIST_DAYS = 3  # 새 레벨이 유지되어야 분할로 인정(일시 스파이크 배제)
PERSIST_TOL = 0.25


def _split_factors(close: np.ndarray) -> np.ndarray:
    """각 인덱스에 곱할 백조정 배수. close는 시간순 1D 배열(NaN 허용)."""
    n = len(close)
    factors = np.ones(n)
    if n <= PERSIST_DAYS + 1:
        return factors
    # ±30% 초과 종가비율만 벡터로 추려 후보로 (전체 일자 파이썬 루프 회피).
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = close[1:] / close[:-1]
    cand = np.where(np.isfinite(ratio) & ((ratio < DOWN) | (ratio > UP)))[0] + 1
    splits: list[tuple[int, float]] = []
    for t in cand:
        if t >= n - PERSIST_DAYS or not (close[t - 1] > 0):
            continue
        # 새 레벨(c1)이 이후에 유지되고 AND 이전 레벨(c0)이 그 앞 며칠과 일관되어야
        # 진짜 분할. 하루만 튀는 스파이크는 되돌림 지점에서 c0(스파이크)이 직전과 어긋나 걸러짐.
        fwd = close[t + 1:t + 1 + PERSIST_DAYS]
        fwd = fwd[np.isfinite(fwd)]
        bwd = close[max(0, t - 1 - PERSIST_DAYS):t - 1]
        bwd = bwd[np.isfinite(bwd)]
        fwd_ok = fwd.size > 0 and abs(np.median(fwd) / close[t] - 1) < PERSIST_TOL
        bwd_ok = bwd.size == 0 or abs(np.median(bwd) / close[t - 1] - 1) < PERSIST_TOL
        if fwd_ok and bwd_ok:
            splits.append((int(t), float(ratio[t - 1])))
    if not splits:
        return factors
    # index i 에는 i 보다 뒤에서 일어난 모든 분할 비율의 곱을 적용(표준 back-adjust)
    for t, r in splits:
        factors[:t] *= r
    return factors


def adjust_prices(df: pd.DataFrame, *, cols=("open", "high", "low", "close"),
                  adjust_volume: bool = False) -> pd.DataFrame:
    """long-format(code,date,OHLC[,volume]) 가격을 종목별로 기업행동 백조정해 반환.

    입력을 변경하지 않고 조정된 사본을 돌려준다. `date` 오름차순 정렬을 가정하지 않는다.
    """
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    # 정수 가격 컬럼에 실수 배수를 할당하므로 먼저 float 캐스팅(dtype 경고 방지)
    for c in present:
        out[c] = out[c].astype(float)
    if adjust_volume and "volume" in out.columns:
        out["volume"] = out["volume"].astype(float)
    for code, idx in out.groupby("code").groups.items():
        sub = out.loc[idx].sort_values("date")
        fac = _split_factors(sub["close"].to_numpy(float))
        order = sub.index
        for c in present:
            out.loc[order, c] = out.loc[order, c].to_numpy(float) * fac
        if adjust_volume and "volume" in out.columns:
            out.loc[order, "volume"] = out.loc[order, "volume"].to_numpy(float) / fac
    return out


def diagnose(df: pd.DataFrame) -> pd.DataFrame:
    """불연속(분할/기준변경) 이벤트를 (code,date,ratio)로 반환 — 진단용."""
    rows = []
    for code, sub in df.groupby("code"):
        sub = sub.sort_values("date")
        c = sub["close"].to_numpy(float)
        dts = sub["date"].astype(str).tolist()
        for t in range(1, len(c) - PERSIST_DAYS):
            if np.isfinite(c[t]) and np.isfinite(c[t - 1]) and c[t - 1] > 0:
                r = c[t] / c[t - 1]
                if r < DOWN or r > UP:
                    fwd = [c[t + k] for k in range(1, PERSIST_DAYS + 1) if np.isfinite(c[t + k])]
                    bwd = [c[t - 1 - k] for k in range(1, PERSIST_DAYS + 1) if np.isfinite(c[t - 1 - k])]
                    fwd_ok = bool(fwd) and abs(np.median(fwd) / c[t] - 1) < PERSIST_TOL
                    bwd_ok = (not bwd) or abs(np.median(bwd) / c[t - 1] - 1) < PERSIST_TOL
                    if fwd_ok and bwd_ok:
                        rows.append({"code": code, "date": dts[t], "ratio": round(r, 3)})
    return pd.DataFrame(rows)


def rebuild_adjusted_table(con: Any, *, adjust_volume: bool = False) -> int:
    """Recompute back-adjusted OHLC for ALL of ``daily_bars`` and upsert into
    ``daily_bars_adjusted``.

    Back-adjustment must see a code's *entire* history to place each split
    factor correctly (a split detected today changes the adjustment applied to
    every earlier date for that code) — so this always recomputes from scratch
    over the full table rather than incrementally, and re-upserts every row
    (existing (code,date) rows are overwritten via the natural-key upsert, so a
    later-discovered split correctly revises previously-adjusted historical
    values). Cheap enough to run periodically (weekly) since daily_bars is a
    few million rows, not billions.
    """
    from .storage import upsert_daily_bars_adjusted

    from .storage import ADJUSTED_BAR_COLUMNS

    # source 도 함께 읽어 그대로 전파한다 — 백테스트가 읽는 건 조정가 테이블이라,
    # "이 행의 trade_value 는 close*volume 근사치(폐지 종목 백필)"라는 사실이 여기
    # 없으면 ADV 문턱을 다루는 코드가 그걸 알 방법이 없다.
    df = pd.read_sql(
        "SELECT code,date,open,high,low,close,volume,trade_value,source FROM daily_bars", con)
    df["date"] = df["date"].astype(str)
    adjusted = adjust_prices(df, adjust_volume=adjust_volume)
    records = list(adjusted[ADJUSTED_BAR_COLUMNS].itertuples(index=False, name=None))
    return upsert_daily_bars_adjusted(con, records)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="기업행동 미조정 진단 리포트 / DB 조정가 테이블 재생성")
    ap.add_argument("--db", default=None)
    ap.add_argument("--rebuild-db", action="store_true",
                    help="진단만 하지 않고 daily_bars_adjusted 테이블을 전체 재계산해 upsert")
    args = ap.parse_args()
    from kr_quant.storage import connect, db_default
    con = connect(args.db or db_default())

    if args.rebuild_db:
        n = rebuild_adjusted_table(con)
        print(f"daily_bars_adjusted 재생성 완료: {n}행 upsert")
        con.close()
        return 0

    df = pd.read_sql("SELECT code,date,close FROM daily_bars", con)
    con.close()
    df["date"] = df["date"].astype(str)
    ev = diagnose(df)
    print(f"미조정 불연속 {len(ev)}건, 고유종목 {ev['code'].nunique()}개")
    top = ev["date"].value_counts().head(5)
    print("상위 날짜:\n" + top.to_string())
    if len(ev):
        print(f"상위3 날짜 집중도: {top.head(3).sum() / len(ev) * 100:.0f}% (높으면 수집-경계 불일치)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
