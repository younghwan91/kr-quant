#!/usr/bin/env python
"""국면 오프스위치 — 배포형 PEAD 북의 MDD 를 랜덤 널보다 잘 깎는가.

사전등록: ``research/logs/regime_switch/VERDICT.md`` (2026-08-27, 결과 보기 전 커밋).

이 러너는 **알파를 찾지 않는다.** 이미 검증된 PEAD 를 배포 가능한 형태(롱온리 +
인버스헤지)로 만든 뒤, 지수가 자기 200일 이평 아래인 달에 북을 내려두는 오버레이를
씌우고 **그게 타이밍인지 그냥 노출 축소인지**를 가른다.

판별기는 always-on 이 아니라 회전 널(:func:`kr_quant.strategies.regime.rotation_null`)이다 —
같은 상태 수열을 시간축으로 돌려 듀티사이클·런렝스·스위치 횟수를 그대로 두고 수익과의
정렬만 깨뜨린다. 노출 축소 효과는 회전에도 살아남고, 타이밍 효과만 죽는다.

Run:  python research/experiments/regime_switch.py
      python research/experiments/regime_switch.py --out report.md
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from kr_quant.engine.metrics import max_drawdown, newey_west_t
from kr_quant.features.fundamentals import _yoy_vec, earnings_yoy_panel
from kr_quant.storage import connect, db_default, read_earnings, read_prices
from kr_quant.strategies.combo import series_metrics
from kr_quant.strategies.hedge import (
    INVERSE_ANNUAL_FEE, inverse_hedged_return, universe_market_return)
from kr_quant.strategies.lowvol import lowvol_backtest
from kr_quant.strategies.pead import pead_backtest
from kr_quant.strategies.regime import (
    apply_switch, ma_regime_state, market_index_level, monthly_state,
    percentile_of, rotation_null)

# --- 사전등록 config (이게 테스트) ---------------------------------------
LIQ_FLOOR = 5000.0     # 50억 (백만원)
HORIZON = 21           # 월 케이던스
COST = 0.0034          # 34bp one-way
SWITCH_COST = 0.0034   # 스위칭 왕복 1회분
MA_WINDOW = 200        # 거래일
HELD_OUT_MONTHS = 24   # 손 안 댄 창
START_INDEX = 130

SENSITIVITY_WINDOWS = (100, 150, 200, 250, 300)


# ------------------------------------------------------------------ 데이터

def load_frames(db: str):
    con = connect(db)
    prices = read_prices(con, cols=("code", "date", "close", "trade_value"))
    prices["date"] = prices["date"].astype(str)
    ea = read_earnings(con, cols=("code", "period", "avail_date", "netinc",
                                  "netinc_prior", "knowledge_date"))
    con.close()
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    dates = sorted(prices["date"].unique())
    panel = earnings_yoy_panel(ea.dropna(subset=["yoy"]), dates,
                              knowledge_col="knowledge_date")
    return prices, panel


def to_monthly(periods: pd.DataFrame, col: str = "net") -> pd.Series:
    s = periods[["date", col]].copy()
    s["m"] = pd.PeriodIndex(pd.to_datetime(s["date"]), freq="M")
    return s.groupby("m")[col].apply(lambda x: (1.0 + x).prod() - 1.0)


def market_monthly(daily: pd.Series) -> pd.Series:
    df = daily.rename("r").reset_index()
    df.columns = ["date", "r"]
    df["m"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    return df.groupby("m")["r"].apply(lambda x: (1.0 + x).prod() - 1.0)


def hedged_monthly(longonly_excess: pd.Series, mkt: pd.Series) -> pd.Series:
    """롱온리 초과수익 → 절대수익 복원 → 인버스헤지(확장창 베타 + TER)."""
    common = longonly_excess.index.intersection(mkt.index)
    book_abs = longonly_excess.loc[common] + mkt.loc[common]
    hedged, _ = inverse_hedged_return(
        book_abs, mkt.loc[common], leverage=1.0, beta=None, beta_window=None,
        fee_per_period=INVERSE_ANNUAL_FEE / 12.0, min_obs=12)
    return hedged.dropna()


# ------------------------------------------------------------------ 평가

def mdd_of(s: pd.Series) -> float:
    return max_drawdown(s.dropna().to_numpy(float))


def sharpe_of(s: pd.Series) -> float:
    return series_metrics(s)["sharpe"]


def row(name: str, s: pd.Series) -> dict:
    m = series_metrics(s)
    x = s.dropna().to_numpy(float)
    m["nw_t"] = newey_west_t(x, lag=1)[1] if len(x) > 6 else float("nan")
    m["name"] = name
    return m


def fmt(m: dict) -> str:
    return (f"| {m['name']} | {m['n']} | {m['ann_ret']:+.1%} | {m['ann_vol']:.1%} | "
            f"{m['sharpe']:.2f} | {m['mdd']:.1%} | {m['nw_t']:+.2f} |")


def evaluate(book: pd.Series, state_m: pd.Series, *, switch_cost=SWITCH_COST) -> dict:
    """오버레이 결과 + 회전 널 대비 위치."""
    on = pd.Series(1.0, index=state_m.index)
    always, _ = apply_switch(book, on, switch_cost=switch_cost)
    sw, exp = apply_switch(book, state_m, switch_cost=switch_cost)

    mdd_nulls, mdd_act = rotation_null(book, state_m, mdd_of, switch_cost=switch_cost)
    shp_nulls, shp_act = rotation_null(book, state_m, sharpe_of, switch_cost=switch_cost)
    return {
        "always": always, "switched": sw, "exposure": exp,
        "duty_on": float(exp.mean()),
        "n_switches": int((exp.diff().abs() > 0).sum()),
        "mdd_delta": mdd_act - mdd_of(always),      # MDD 는 음수 → 클수록 개선
        "mdd_pct": percentile_of(mdd_act, mdd_nulls),
        "mdd_null_med": float(np.median(mdd_nulls)) if mdd_nulls else float("nan"),
        "mdd_null_p90": float(np.percentile(mdd_nulls, 90)) if mdd_nulls else float("nan"),
        "shp_act": shp_act,
        "shp_pct": percentile_of(shp_act, shp_nulls),
        "shp_null_med": float(np.median(shp_nulls)) if shp_nulls else float("nan"),
        "shp_always": sharpe_of(always),
    }


# ------------------------------------------------------------------ 리포트

def build_report(prices, panel) -> str:
    L: list[str] = []
    w = L.append

    daily_mkt = universe_market_return(prices, adv_floor=LIQ_FLOOR)
    mkt = market_monthly(daily_mkt)
    level = market_index_level(daily_mkt)

    pe_lo, _ = pead_backtest(prices, panel, horizon=HORIZON, adv_floor=LIQ_FLOOR,
                             cost_one_way=COST, long_only=True, start_index=START_INDEX)
    pead_hedged = hedged_monthly(to_monthly(pe_lo), mkt)

    lv_lo, _ = lowvol_backtest(prices, horizon=HORIZON, adv_floor=LIQ_FLOOR,
                               cost_one_way=COST, long_only=True, start_index=START_INDEX)
    lowvol_hedged = hedged_monthly(to_monthly(lv_lo), mkt)

    state_daily = ma_regime_state(level, window=MA_WINDOW)
    state_m = monthly_state(state_daily)

    w("# 국면 오프스위치 — 실측 리포트")
    w("")
    w(f"사전등록 config: MA{MA_WINDOW}일 이진 스위치, lag 1개월, 스위칭 비용 "
      f"{SWITCH_COST*1e4:.0f}bp, 유동성 하한 {LIQ_FLOOR:,.0f}백만원.")
    w("")

    # ---- 1. 판정 대상: 배포형 PEAD
    res = evaluate(pead_hedged, state_m)
    w("## 1. 판정 대상 — PEAD 롱온리 + 인버스헤지")
    w("")
    w("| 북 | 월수 | 연율수익 | 연율변동성 | 샤프 | MDD | NW-t |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    w(fmt(row("always-on (오버레이 없음)", res["always"])))
    w(fmt(row(f"국면 스위치 (MA{MA_WINDOW})", res["switched"])))
    w("")
    w(f"- 켜져 있던 비율(듀티사이클): **{res['duty_on']:.1%}**, 스위칭 횟수: {res['n_switches']}회")
    w(f"- MDD 변화: {res['mdd_delta']:+.1%}p (양수 = 개선)")
    w("")
    w("### 회전 널 대비 — 이게 판별기다")
    w("")
    w("| 지표 | 실제 | 널 중앙값 | 널 p90 | 실제의 널 내 백분위 |")
    w("|---|---:|---:|---:|---:|")
    w(f"| MDD | {mdd_of(res['switched']):.1%} | {res['mdd_null_med']:.1%} | "
      f"{res['mdd_null_p90']:.1%} | {res['mdd_pct']:.0%} |")
    w(f"| Sharpe | {res['shp_act']:.2f} | {res['shp_null_med']:.2f} | — | {res['shp_pct']:.0%} |")
    w("")
    w("> 회전 널은 같은 상태 수열을 시간축으로 돌린 것이라 듀티사이클·스위치 횟수가 동일하다."
      " 백분위가 50% 근처면 국면 신호에 타이밍 정보가 없다는 뜻이다.")
    w("")

    # ---- 2. 비용 2배
    res2 = evaluate(pead_hedged, state_m, switch_cost=SWITCH_COST * 2)
    w("## 2. 스위칭 비용 2배")
    w("")
    w("| 북 | 월수 | 연율수익 | 연율변동성 | 샤프 | MDD | NW-t |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    w(fmt(row("국면 스위치 (68bp)", res2["switched"])))
    w("")
    w(f"- MDD 변화 {res2['mdd_delta']:+.1%}p, 널 내 백분위 {res2['mdd_pct']:.0%}, "
      f"Sharpe {res2['shp_act']:.2f} (always-on {res2['shp_always']:.2f})")
    w("")

    # ---- 3. 레짐 분해 + 손 안 댄 창
    w("## 3. 레짐 분해 · 손 안 댄 창")
    w("")
    w("| 구간 | 월수 | always-on MDD | 스위치 MDD | 변화 | 스위치 Sharpe |")
    w("|---|---:|---:|---:|---:|---:|")
    idx = pead_hedged.index
    mid = idx[len(idx) // 2]
    held_start = idx[-HELD_OUT_MONTHS]
    for label, sl in (("전반", pead_hedged[idx < mid]),
                      ("후반", pead_hedged[idx >= mid]),
                      (f"탐색창(마지막 {HELD_OUT_MONTHS}개월 제외)",
                       pead_hedged[idx < held_start]),
                      (f"손 안 댄 창(마지막 {HELD_OUT_MONTHS}개월)",
                       pead_hedged[idx >= held_start])):
        if len(sl) < 8:
            continue
        r = evaluate(sl, state_m)
        w(f"| {label} | {len(r['switched'])} | {mdd_of(r['always']):.1%} | "
          f"{mdd_of(r['switched']):.1%} | {r['mdd_delta']:+.1%}p | {r['shp_act']:.2f} |")
    w("")

    # ---- 4. 민감도 (승자선택 금지 — 격자는 민감도 전용)
    w("## 4. 민감도 — MA 창 (격자는 민감도 전용, 승자선택 금지)")
    w("")
    w("| MA | 듀티 | 스위치 | MDD | MDD 변화 | 널 백분위 | Sharpe |")
    w("|---:|---:|---:|---:|---:|---:|---:|")
    for win in SENSITIVITY_WINDOWS:
        sm = monthly_state(ma_regime_state(level, window=win))
        r = evaluate(pead_hedged, sm)
        star = " ←사전등록" if win == MA_WINDOW else ""
        w(f"| {win}{star} | {r['duty_on']:.0%} | {r['n_switches']} | "
          f"{mdd_of(r['switched']):.1%} | {r['mdd_delta']:+.1%}p | "
          f"{r['mdd_pct']:.0%} | {r['shp_act']:.2f} |")
    w("")
    w("| 3단(1.0/0.5/0) | 듀티 | 스위치 | MDD | MDD 변화 | 널 백분위 | Sharpe |")
    w("|---:|---:|---:|---:|---:|---:|---:|")
    sm3 = monthly_state(ma_regime_state(level, window=MA_WINDOW, mid_window=60))
    r3 = evaluate(pead_hedged, sm3)
    w(f"| MA{MA_WINDOW}/60 | {r3['duty_on']:.0%} | {r3['n_switches']} | "
      f"{mdd_of(r3['switched']):.1%} | {r3['mdd_delta']:+.1%}p | "
      f"{r3['mdd_pct']:.0%} | {r3['shp_act']:.2f} |")
    w("")

    # ---- 5. 경쟁 가설: 인버스헤지 자체 / 저변동
    w("## 5. 참고 — 다른 북에도 같은 오버레이")
    w("")
    w("| 북 | always-on MDD | 스위치 MDD | 변화 | 널 백분위 | always Sharpe | 스위치 Sharpe |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    # ⚠️ 사후 탐색(판정 아님). 위 두 북은 **이미 시장중립**이다 — 엔진의 롱온리 gross 는
    # 유니버스 평균 대비 초과수익이고, 헤지본은 베타까지 명시적으로 제거했다. 국면
    # 스위치는 원래 *시장노출*을 타이밍하는 장치이므로, 메커니즘이 이 데이터에서
    # 아예 작동하지 않는지 보려면 시장노출을 가진 절대수익 북에서도 봐야 한다.
    pead_abs = (to_monthly(pe_lo).reindex(mkt.index) + mkt).dropna()
    for label, bk in (("PEAD 롱온리(초과수익 — 이미 시장중립)", to_monthly(pe_lo)),
                      ("저변동 롱온리+헤지", lowvol_hedged),
                      ("[사후] PEAD 절대수익 북(시장노출 있음)", pead_abs),
                      ("[사후] 시장 자체(등가중 유니버스)", mkt)):
        r = evaluate(bk, state_m)
        w(f"| {label} | {mdd_of(r['always']):.1%} | {mdd_of(r['switched']):.1%} | "
          f"{r['mdd_delta']:+.1%}p | {r['mdd_pct']:.0%} | "
          f"{r['shp_always']:.2f} | {r['shp_act']:.2f} |")
    w("")
    w("---")
    w("")
    w("재현: `python research/experiments/regime_switch.py`")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("KR_QUANT_DB") or db_default())
    ap.add_argument("--out")
    a = ap.parse_args()
    prices, panel = load_frames(a.db)
    rep = build_report(prices, panel)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(rep + "\n")
        print(f"wrote {a.out}")
    else:
        print(rep)


if __name__ == "__main__":
    main()
