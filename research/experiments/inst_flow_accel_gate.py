#!/usr/bin/env python
"""기관 수급 가속 — 임펄스가 아니라 가속도가 예측하는가.

사전등록: ``research/logs/inst_flow_accel/VERDICT.md`` (2026-08-27, 결과 보기 전 커밋).

**물리 프레이밍이 신호를 강제한다.** 질량 m=시가총액, 외력 F=기관 순매수 유입,
가속도 a=F/m, 임펄스 J=∫F dt. 같은 임펄스라도 질량이 작으면 Δv 가 크므로 수익률을
예측해야 하는 건 순매수 *금액*이 아니라 *금액/시총*이다. 금액을 그대로 쓰면 이미
기각된 사이즈 팩터를 다시 잡는다.

  signal = accel(t) − accel(t−20),  accel(t) = Σ_{t−19..t} 기관순매수금액 / 시총(t)

판정 배터리는 재발명하지 않는다 — ``prop_gate`` 하버스가 음성대조·비용스윕·
손안댄창·R분포·fragility·DSR 을 한 번에 낸다.

Run:  python research/experiments/inst_flow_accel_gate.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prop_gate import prop_gate, random_entry_control  # noqa: E402

from kr_quant.engine.panels import panel_pivot  # noqa: E402
from kr_quant.storage import (  # noqa: E402
    SHARES_RESIDUAL_OK, connect, db_default, market_cap_asof_bulk, read_prices,
    read_supply_demand, shares_backfill_pending)

OUT_DIR = "research/logs/inst_flow_accel"

# --- 사전등록 config (이게 테스트) ---
PRE_WIN = 20          # 가속도 창(거래일)
PRE_LAG = 20          # 직전 창과의 간격 = 변화를 재는 축
PRE_HOLD = 21         # 보유(거래일)
PRE_STEP = 21         # 리밸 간격 = 월
PRE_TOPQ = 0.10       # 상위 10% 롱
PRE_STOP = 0.08       # 하드손절 8% (R 정규화 분모)
ADV_FLOOR = 5000.0    # 백만원 = 50억
ADV_WIN = 20
START_I = 80          # 워밍업(질량창 20 + LAG 20 + WIN 20 + 여유)
MASS_LAG = 40         # v2: 질량을 두 유량 창보다 앞선 [t−59..t−40] 에서 잰다


def load(db: str):
    con = connect(db)
    prices = read_prices(con, cols=("code", "date", "close", "trade_value"))
    prices["date"] = prices["date"].astype(str)
    # 기관만 읽는다 — individual 을 섞으면 폐지 종목이 NULL 로 빠진다(정문이 막는다).
    sd = read_supply_demand(con, cols=("code", "date", "close", "institution"))
    sd["date"] = sd["date"].astype(str)
    return con, prices, sd


def build(prices: pd.DataFrame, sd: pd.DataFrame):
    # ⚠️ panel_pivot 은 **code × date** 를 준다(index=code). 시계열 rolling 을 하려면
    # date × code 로 세워야 한다 — 전치를 빠뜨리면 코드와 날짜가 서로 뒤바뀌어
    # reindex 가 전부 NaN 이 되고 트레이드가 0 건이 된다(실제로 한 번 밟았다).
    close = panel_pivot(prices, "close").T           # date × code
    tv = panel_pivot(prices, "trade_value").T
    sd = sd.copy()
    sd["amt"] = sd["institution"].astype(float) * sd["close"].abs() / 1e8   # 억원
    flow = (sd.pivot_table(index="date", columns="code", values="amt", aggfunc="sum")
              .reindex(index=close.index, columns=close.columns))
    adv = tv.rolling(ADV_WIN, min_periods=ADV_WIN).mean()
    return close, adv, flow


def trades(con, close, adv, flow, mass: str = "cap", *,
           win: int = PRE_WIN, lag: int = PRE_LAG, topq: float = PRE_TOPQ,
           hold: int = PRE_HOLD, step: int = PRE_STEP):
    """``mass="cap"`` = 시가총액(v1), ``"adv"`` = 직전 평균 거래대금(v2).

    v2 의 질량은 **두 유량 창보다 앞선** 구간에서 잰다. 분자와 분모를 같은 창에서
    재면 둘이 함께 움직여 신호가 뭉개지고, 물리적으로도 질량은 힘보다 먼저 정해져
    있어야 한다. adv 는 rolling(20).mean() 이므로 ``adv[t − MASS_LAG]`` 가 곧
    [t−59..t−40] 평균이다. 단위는 백만원이라 억원으로 맞춘다(÷100).
    """
    dates = list(close.index)
    codes = np.array(close.columns)
    C = close.to_numpy(float)
    A = adv.to_numpy(float)
    F = flow.to_numpy(float)
    # 누적합으로 창 합을 O(1) 에 — Σ_{a..b} = cs[b+1] − cs[a]
    cs = np.vstack([np.zeros(F.shape[1]), np.nancumsum(np.nan_to_num(F), axis=0)])

    rebal = list(range(START_I, len(dates) - hold - 1, step))
    capmap = {}
    if mass == "cap":
        caps = pd.concat([pd.DataFrame({"code": codes, "date": dates[t]}) for t in rebal],
                         ignore_index=True)
        caps["cap"] = market_cap_asof_bulk(con, caps).to_numpy() / 1e8        # 억원
        capmap = {(r.code, r.date): r.cap for r in caps.itertuples()}

    ent_d, rets, sigs = [], [], []
    for t in rebal:
        d = dates[t]
        if mass == "cap":
            m = np.array([capmap.get((c, d), np.nan) for c in codes])
        else:
            m = A[t - MASS_LAG] / 100.0        # 백만원 → 억원 (일평균 거래대금)
        cap = m
        recent = cs[t + 1] - cs[t + 1 - win]                       # 최근 win 일 유입
        prior = cs[t + 1 - lag] - cs[t + 1 - lag - win]
        with np.errstate(invalid="ignore", divide="ignore"):
            sig = (recent - prior) / cap                           # Δ가속도 (배수)
        ok = (A[t] >= ADV_FLOOR) & np.isfinite(sig) & np.isfinite(C[t + 1]) \
            & np.isfinite(C[t + 1 + hold]) & (cap > 0)
        n = int(ok.sum())
        if n < 30:
            continue
        idx = np.where(ok)[0]
        k = max(1, int(round(n * topq)))
        pick = idx[np.argsort(sig[idx])[::-1][:k]]
        r = C[t + 1 + hold, pick] / C[t + 1, pick] - 1.0
        ent_d += [dates[t + 1]] * len(pick)
        rets += list(r)
        sigs += list(sig[pick])
    return np.array(ent_d), np.array(rets, float), np.array(sigs, float)


def run(date_from: str | None = None, mass: str = "cap",
        allow_partial_shares: bool = False):
    """``date_from`` 을 주면 그 이후 진입만 남긴다 — 시총이 실제로 계산되는 구간으로
    좁혀 **기술 통계**를 보기 위한 것이다. 폴드·손안댄창이 무너지므로 **판정이 아니다.**
    사전등록 신호는 한 글자도 바뀌지 않는다(분모를 갈아끼우면 사후 선택이 된다)."""
    db = os.environ.get("KR_QUANT_DB") or db_default()
    con, prices, sd = load(db)

    # mass="cap" 은 시가총액이 분모다 — 백필이 도는 중이면 유니버스가 시대별로
    # 다르게 좁혀지고, 그 상태의 판정은 신호가 아니라 처리 진행률을 재게 된다.
    if mass == "cap":
        pending = shares_backfill_pending(con)
        print(f"[사전검사] 과거 주식수 미처리 상장 종목: {pending}건 "
              f"(정상 종료 기대치 ~60)")
        if pending > SHARES_RESIDUAL_OK and not allow_partial_shares:
            con.close()
            print("  → 백필이 아직 도는 중이다. 지금 돌리면 유니버스를 신호가 아니라 "
                  "데이터 가용성이 고른다(v1 VOID 의 원인). 중단한다.\n"
                  "     완료 후 재실행하거나, 의도한 것이면 --allow-partial-shares 를 준다.")
            return None, None

    close, adv, flow = build(prices, sd)
    ent, ret, _sig = trades(con, close, adv, flow, mass=mass)
    con.close()
    if date_from:
        keep = ent >= date_from
        ent, ret = ent[keep], ret[keep]
        print(f"[구간 제한] 진입 >= {date_from}")
    if len(ret) == 0:
        print("트레이드 0건 — 보고할 것이 없다")
        return None, None
    print(f"트레이드 {len(ret)}건 · {min(ent)} ~ {max(ent)}")

    # 민감도 격자를 **먼저** — 원장 N 이 사전등록 게이트보다 앞서 쌓여야 DSR 이 걸린다.
    # ⚠️ 2026-08-27 수정. 이전 판본은 win·topq·stop 을 **라벨 dict 에만** 넣고
    # trades() 에 넘기지 않았다. 트레이드는 모듈 상수로 한 번만 계산됐으므로
    # 격자 전체가 **동일한 계산**이었고, 민감도는 어디에서도 산출되지 않은 채
    # 원장 N 만 부풀렸다. 린트 (e)는 config= 가 있어 통과했다 — 글자는 검사하고
    # 취지는 놓친 사례다. 이제 칸마다 실제로 다시 시뮬레이션하고 결과를 보고한다.
    grid = []
    for gwin, gtopq, gstop in ((10, PRE_TOPQ, PRE_STOP), (40, PRE_TOPQ, PRE_STOP),
                               (PRE_WIN, 0.05, PRE_STOP), (PRE_WIN, PRE_TOPQ, 0.05)):
        g = dict(win=gwin, lag=PRE_LAG, topq=gtopq, hold=PRE_HOLD, stop=gstop,
                 step=PRE_STEP, adv_floor=ADV_FLOOR, mass=mass)
        con2 = connect(db)
        gent, gret, _ = trades(con2, close, adv, flow, mass=mass,
                               win=gwin, topq=gtopq)
        con2.close()
        if date_from is not None:
            keep = gent >= date_from
            gent, gret = gent[keep], gret[keep]
        if len(gret) < 30:
            print(f"  [민감도] {g} — 트레이드 {len(gret)}건, 건너뜀")
            continue
        grep = prop_gate(gent, gret, gstop, label="inst_flow_accel_sens",
                         log_dir=OUT_DIR, config=g, verbose=False)
        grid.append((g, len(gret), grep))

    rep = prop_gate(ent, ret, PRE_STOP, label="inst_flow_accel", log_dir=OUT_DIR, config={
        "win": PRE_WIN, "lag": PRE_LAG, "topq": PRE_TOPQ, "hold": PRE_HOLD,
        "stop": PRE_STOP, "step": PRE_STEP, "adv_floor": ADV_FLOOR, "mass": mass,
    })
    ctrl = random_entry_control(ent, ret, PRE_STOP, n_per_draw=len(ret),
                                n_draws=200, seed=7)

    # 민감도는 **보고돼야** 민감도다. 이전 판본은 verbose=False 리포트를 버렸다.
    # 민감도는 **보고돼야** 민감도다. 이전 판본은 verbose=False 리포트를 버렸다.
    def _oos(r):
        cs = r.get("cost_sweep") or [{}]
        return cs[0].get("oos_expectancy_R")

    print("\n민감도 격자 — 칸마다 다시 시뮬레이션 (승자선택 금지 · 기준비용 OOS expR)")
    print("  win topq  stop |     n | OOS expR")
    print(f"  {PRE_WIN:3d} {PRE_TOPQ:.2f} {PRE_STOP:.2f} | {len(ret):5d} | "
          f"{_oos(rep):+.3f}   ←사전등록")
    for g, n, r in grid:
        v = _oos(r)
        v = f"{v:+.3f}" if v is not None else "  n/a"
        print(f"  {g['win']:3d} {g['topq']:.2f} {g['stop']:.2f} | {n:5d} | {v}")
    if not grid:
        print("  (격자 없음)")
    return rep, ctrl


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from",
                    help="이 날짜 이후 진입만 (기술 통계용 — 판정 아님)")
    ap.add_argument("--mass", default="cap", choices=("cap", "adv"),
                    help="질량: cap=시가총액(v1), adv=직전 평균 거래대금(v2)")
    ap.add_argument("--allow-partial-shares", action="store_true",
                    help="주식수 백필이 미완이어도 강행(판정용으로 쓰지 말 것)")
    a = ap.parse_args()
    run(a.date_from, a.mass, a.allow_partial_shares)
