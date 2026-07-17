#!/usr/bin/env python
"""개미 반대매매(Contrarian-to-Retail) 백테스트 — 개인 순매수를 정반대로.

가설(사용자 지정): 개인투자자(개미)가 순매수하는 종목을 회피/숏하고, 순매도하는
종목을 롱하면 인덱스를 이긴다("무조건 정반대"). 이 스크립트는 그 가설을 이 레포의
투자자별 수급 데이터(``supply_demand.individual``, 2017~2026, 2338거래일)로 검증한다.

정직성 원칙 (이 레포의 북극성): 정반대가 알파면 알파로, 아니면 아님으로 보고한다.
"무조건 정반대"는 **검증할 가설**이지 커브핏으로 양수 만들 명령이 아니다.

핵심 사실 (DB 실측): 개인 순매수는 (외국인+기관)과 94.6%가 반대 부호. 즉 "개미 반대"는
구조적으로 "스마트머니(외국인+기관) 편승"과 거의 같다.

Scratch research. src/kr_quant/는 건드리지 않는다. engine 회계(staggered_backtest)를
**재사용**한다 — 진입가·벤치마크·연율화를 새로 구현하지 않는다(docs/backtest-engine.md 규칙).

신호 설계
---------
- retail_intensity(code, t) = Σ individual[t-W..t] / Σ acc_trde_qty[t-W..t]
  (개인 순매수 주식수 / 거래량, 윈도 W일 누적 — 거래량 중 개미 순매수 비중, 단위 일관)
- contrarian_signal(code, t) = -retail_intensity   (개미가 판 종목이 높은 점수 → 롱)
- **룩어헤드 방지:** date t의 신호는 t-LAG일까지의 수급만 사용(LAG≥1). staggered_backtest는
  close[t]에 진입하므로, t일 수급(장마감 후 확정)으로 t일 진입하면 동시성 = 룩어헤드.
  LAG=1로 t-1일까지의 수급만 써서 인과성 보장.

가격
----
분할조정 필수 — daily_bars_adjusted(원자료 daily_bars는 분할이 가짜 −68%로 잡힘,
docs/pead-refinement.md / MULTI_ALPHA.md §"반드시 지킬 전제" #1).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _load_env_db() -> str:
    """``.env``의 KR_QUANT_DB를 로드해 반환(셸에 export 안 돼 있어도 동작)."""
    v = os.environ.get("KR_QUANT_DB")
    if not v and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("KR_QUANT_DB"):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["KR_QUANT_DB"] = v
    return v or ""


# --- Baseline parameters ----------------------------------------------------
PRICE_TABLE = "daily_bars_adjusted"   # 분할조정 (원자료 daily_bars 금지)
SIGNAL_WINDOW = 5      # 개미 순매수 누적 윈도(거래일)
SIGNAL_LAG = 1         # 룩어헤드 방지 래그(거래일)
BASELINE = dict(horizon=20, step=5, top_n=40, adv_floor=20000.0, start_index=130, min_names=20)


def load_data(db: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DB에서 분할조정 가격 + 개인 순매수 수급을 로드.

    Returns ``(prices, flow)``:
      - ``prices``: long ``code``/``date``/``close``/``trade_value`` (분할조정)
      - ``flow``:   long ``code``/``date``/``individual``/``volume``
    """
    from kr_quant.storage import connect, db_default
    con = connect(db or db_default())
    prices = pd.read_sql_query(
        f"SELECT code, date, close, trade_value FROM {PRICE_TABLE}", con  # noqa: S608 — 신뢰 상수
    )
    flow = pd.read_sql_query(
        "SELECT code, date, individual, acc_trde_qty AS volume FROM supply_demand", con
    )
    con.close()
    for df in (prices, flow):
        df["code"] = df["code"].astype(str)
        df["date"] = df["date"].astype(str)
    return prices, flow


def build_contrarian_signal(
    flow: pd.DataFrame,
    *,
    window: int = SIGNAL_WINDOW,
    lag: int = SIGNAL_LAG,
    sign: int = -1,
) -> pd.DataFrame:
    """개인 순매수 → (반전) 반대매매 신호 패널(long code/date/signal).

    signal = sign × [Σ individual(window) / Σ volume(window)], t-lag일까지만 사용.
    sign=-1: 반대매매(개미 순매도 종목이 고점수 → 롱). sign=+1: 개미 추종(대조군).

    윈도 합/래그는 종목별 시계열로 계산 → 룩어헤드 없음(shift(lag)로 t는 t-lag까지만 봄).
    """
    f = flow.sort_values(["code", "date"]).copy()
    g = f.groupby("code", sort=False)
    ind_sum = g["individual"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    vol_sum = g["volume"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    intensity = ind_sum / vol_sum.where(vol_sum > 0)
    # 래그: date t의 신호는 t-lag까지의 수급만 반영
    f["signal"] = sign * g_shift(intensity, f["code"], lag)
    out = f[["code", "date", "signal"]].dropna(subset=["signal"])
    return out


def g_shift(series: pd.Series, codes: pd.Series, lag: int) -> pd.Series:
    """종목별 shift(lag) — 미래 정보 유입 없이 t를 t-lag로 민다."""
    return series.groupby(codes, sort=False).shift(lag)


def run_arm(
    prices: pd.DataFrame,
    signal_panel: pd.DataFrame,
    *,
    cost_one_way: float = 0.0,
    **params,
) -> dict:
    """engine의 staggered_backtest로 1개 arm 실행 → summary dict 반환.

    회계(진입가 close[t]·벤치마크 유니버스평균·연율화·excess)는 전부 engine 소유.
    여기선 신호만 주입한다.
    """
    from kr_quant.strategies.pead import staggered_backtest
    p = {**BASELINE, **params}
    periods, summary = staggered_backtest(prices, earnings_panel=None, signal_panel=signal_panel, **p)
    if cost_one_way and not periods.empty:
        # 비용반영: step마다 회전율×비용 차감(1방향). engine의 net은 gross와 동일(무비용)이므로
        # 여기서 사후 차감 컬럼을 별도 계산(정직성: 1차 비교는 무비용, 비용은 2차).
        turn = periods["turnover"] if "turnover" in periods else 1.0
        net = periods["gross"] - turn * cost_one_way
        summary = {**summary, "cum_net_cost": float((1 + net).prod() - 1),
                   "sharpe_cost": float(net.mean() / net.std() * np.sqrt(252 / p["step"]))
                   if net.std() > 0 else float("nan")}
    return summary


def _fmt(s: dict) -> str:
    return (f"n={s.get('n', 0):>4}  Sharpe={s.get('sharpe', float('nan')):+.3f}  "
            f"t={s.get('t_stat', float('nan')):+.3f}  cum={s.get('cum_net', float('nan')):+.3f}  "
            f"hit={s.get('hit_rate', float('nan')):.3f}  worst={s.get('worst', float('nan')):+.3f}")


def _mdd(period_net: np.ndarray) -> float:
    """기간수익 시계열의 최대낙폭(음수)."""
    if len(period_net) == 0:
        return float("nan")
    eq = np.cumprod(1 + period_net)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def _panels(prices: pd.DataFrame, signal_panel: pd.DataFrame):
    """close·trade_value·signal을 동일 (code×date) 그리드로 정렬해 반환."""
    from kr_quant.engine.panels import panel_pivot
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    codes, dates = list(close.index), list(close.columns)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    S = (signal_panel.pivot_table(index="code", columns="date", values="signal", aggfunc="first")
         .reindex(index=codes, columns=dates).to_numpy(float))
    return C, V, S, dates


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ar = pd.Series(a[m]).rank().to_numpy().astype(float)
    br = pd.Series(b[m]).rank().to_numpy().astype(float)
    ar = ar - ar.mean()
    br = br - br.mean()
    d = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / d) if d > 0 else float("nan")


def diagnose(prices, signal_panel, *, horizon=20, step=5, adv_floor=20000.0,
             adv_window=20, start_index=130, min_names=20, top_n=40, date_lo=None, date_hi=None):
    """부호 격리 진단: 랭크-IC + 롱숏 스프레드. signal_panel은 이미 반대매매(-1) 신호.

    - rank-IC: 매 리밸런스 corr(반대매매신호, 미래h수익) over 유동성 유니버스. 양수 IC =
      반대매매 부호가 미래수익을 맞힘(개미 순매수 종목이 저조).
    - 롱숏: top_n(개미판=고신호) 롱 − bottom_n(개미산=저신호) 숏 스프레드. 유니버스 중립 →
      순수 부호 엣지. ≈0이면 부호 무의미(공통 교란), 강양수면 반대매매 부호가 진짜 알파.
    """
    C, V, S, dates = _panels(prices, signal_panel)
    nD = C.shape[1]
    di = np.array([str(d) for d in dates])
    ics, spreads = [], []
    for t in range(start_index, nD - horizon - 1, step):
        if date_lo and di[t] < date_lo:
            continue
        if date_hi and di[t] >= date_hi:
            continue
        adv = np.nanmean(V[:, t - adv_window:t], axis=1)
        elig = adv >= adv_floor
        s_t = np.where(elig, S[:, t], np.nan)
        if np.isfinite(s_t).sum() < min_names:
            continue
        fwd = C[:, t + horizon] / C[:, t] - 1.0
        ics.append(_spearman(s_t, fwd))
        order = np.argsort(np.where(np.isfinite(s_t), -s_t, -np.inf))  # 고신호 우선
        valid = order[np.isfinite(s_t[order])]
        # 롱·숏 disjoint 보장: 유니버스가 2*top_n 미만이면 k를 1/3로 제한(중간 1/3 남김) →
        # 겹침으로 스프레드가 0으로 편향되는 아티팩트 방지(유동성 유니버스 중앙값 101,
        # 33.8% 구간이 80 미만이라 필수).
        k = min(top_n, len(valid) // 3)
        if k < 1:
            continue
        longs, shorts = valid[:k], valid[-k:]
        lr, sr = np.nanmean(fwd[longs]), np.nanmean(fwd[shorts])
        if np.isfinite(lr) and np.isfinite(sr):
            spreads.append(lr - sr)
    ics = np.array([x for x in ics if np.isfinite(x)])
    spreads = np.array(spreads)
    per_year = 252 / step
    def _stat(x):
        if len(x) == 0 or x.std() == 0:
            return dict(n=len(x), mean=float("nan"), t=float("nan"), sharpe=float("nan"))
        return dict(n=len(x), mean=float(x.mean()),
                    t=float(x.mean() / (x.std() / np.sqrt(len(x)))),
                    sharpe=float(x.mean() / x.std() * np.sqrt(per_year)))
    ic = _stat(ics)
    ls = _stat(spreads)
    ic["pct_pos"] = float((ics > 0).mean()) if len(ics) else float("nan")
    ls["cum"] = float(np.prod(1 + spreads) - 1) if len(spreads) else float("nan")
    ls["mdd"] = _mdd(spreads)
    return {"rank_ic": ic, "long_short": ls}


def main() -> int:
    ap = argparse.ArgumentParser(description="개미 반대매매 백테스트")
    ap.add_argument("--db", default=None)
    ap.add_argument("--window", type=int, default=SIGNAL_WINDOW)
    ap.add_argument("--cost", type=float, default=0.0023, help="1방향 거래비용(2차 비교용)")
    ap.add_argument("--out", default="docs/contrarian-retail.md")
    args = ap.parse_args()

    _load_env_db()
    print("=== 데이터 로드 ===")
    prices, flow = load_data(args.db)
    print(f"  가격 {len(prices):,}행({PRICE_TABLE}), 수급 {len(flow):,}행")

    contr = build_contrarian_signal(flow, window=args.window, sign=-1)
    follow = build_contrarian_signal(flow, window=args.window, sign=+1)
    print(f"  신호 패널: 반대매매 {len(contr):,}행, 추종 {len(follow):,}행 (window={args.window}, lag={SIGNAL_LAG})")

    print("\n=== 3-arm 롱온리 excess (staggered, 무비용 1차) ===")
    s_contr = run_arm(prices, contr, cost_one_way=args.cost)
    s_follow = run_arm(prices, follow)
    print(f"  [반대매매] {_fmt(s_contr)}")
    print(f"  [개미추종] {_fmt(s_follow)}")
    if "sharpe_cost" in s_contr:
        print(f"  [반대매매·비용후] Sharpe={s_contr['sharpe_cost']:+.3f}  cum={s_contr['cum_net_cost']:+.3f}")
    print("  ⚠️ 둘 다 양수면 부호가 아니라 공통 교란(관심집중) — 아래 부호 격리 진단이 판정.")

    print("\n=== 부호 격리 진단 (핵심 판정) ===")
    d = diagnose(prices, contr)
    ic, ls = d["rank_ic"], d["long_short"]
    print(f"  [랭크-IC] mean={ic['mean']:+.4f}  t={ic['t']:+.2f}  Sharpe={ic['sharpe']:+.2f}  "
          f"IC>0비율={ic['pct_pos']:.1%}  (양수=반대매매 부호 유효)")
    print(f"  [롱숏]   mean={ls['mean']:+.4f}  t={ls['t']:+.2f}  Sharpe={ls['sharpe']:+.2f}  "
          f"cum={ls['cum']:+.3f}  MDD={ls['mdd']:+.3f}  (≈0=부호무의미, 강양수=진짜 알파)")

    print("\n=== 하위기간 롱숏 (견고성) ===")
    for lo, hi, name in [("2017-01-01", "2022-01-01", "2017-2021"),
                         ("2022-01-01", "2027-01-01", "2022-2026")]:
        dd = diagnose(prices, contr, date_lo=lo, date_hi=hi)["long_short"]
        print(f"  {name}: 롱숏 Sharpe={dd['sharpe']:+.2f}  t={dd['t']:+.2f}  mean={dd['mean']:+.4f}")

    print("\n=== 파라미터 스윕 (부호 엣지가 어디서도 살아나나) ===")
    print(f"  {'window':>6} {'horizon':>7} {'IC_t':>7} {'LS_t':>7} {'LS_Sharpe':>9} {'LS_cum':>8}")
    sweep_rows = []
    for w in (1, 5, 20):
        sig_w = build_contrarian_signal(flow, window=w, sign=-1)
        for h in (5, 20, 60):
            dg = diagnose(prices, sig_w, horizon=h, step=5)
            r = {"window": w, "horizon": h, "ic_t": dg["rank_ic"]["t"],
                 "ls_t": dg["long_short"]["t"], "ls_sharpe": dg["long_short"]["sharpe"],
                 "ls_cum": dg["long_short"]["cum"]}
            sweep_rows.append(r)
            print(f"  {w:>6} {h:>7} {r['ic_t']:>+7.2f} {r['ls_t']:>+7.2f} "
                  f"{r['ls_sharpe']:>+9.2f} {r['ls_cum']:>+8.3f}")
    mx = max(abs(r["ls_t"]) for r in sweep_rows)
    print(f"\n  전 격자 |롱숏 t| 최대 = {mx:.2f}  "
          f"→ {'2 미만이면 어떤 파라미터도 부호 엣지 없음(가설 기각)' if mx < 2 else '2 이상 존재 — 추가검토'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
