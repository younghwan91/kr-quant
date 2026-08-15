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
    from kr_quant.storage import connect, db_default, read_prices
    con = connect(db or db_default())
    prices = read_prices(con, cols=("code", "date", "close", "trade_value"))
    # individual IS NOT NULL 을 명시하는 이유: 폐지 종목 수급은 네이버에서 부분만
    # 받아(기관·외국인) individual 이 NULL 이다. 필터가 없어도 NaN 전파로 결국
    # 빠지지만, 그러면 "왜 폐지 종목이 이 연구에 없는가"가 코드에 안 드러난다.
    # 이 전략은 개인 순매매가 신호 자체라 부분 데이터로는 재현할 수 없다.
    flow = pd.read_sql_query(
        "SELECT code, date, individual, acc_trde_qty AS volume FROM supply_demand "
        "WHERE individual IS NOT NULL", con
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


def build_behavior_signals(
    flow: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    window: int = SIGNAL_WINDOW,
    lag: int = SIGNAL_LAG,
) -> dict[str, pd.DataFrame]:
    """개미 '심리적 실수'를 가격맥락과의 상호작용으로 신호화 → 각각 정반대(반대매매).

    단순 순매수 부호가 아니라 **행동**을 뒤집는다. 개미 강도 ri = Σ개인/Σ거래량(window),
    과거수익 pr = close[t]/close[t-window]-1. 둘 다 t-lag까지만(룩어헤드 없음).

    - **antichase(추격 반대):** -ri × relu(pr). 오른 종목(pr>0)에서 개미 페이드 —
      개미가 산 급등주는 숏(음수), 개미가 판 급등주는 롱(양수). "고점 추격매수"의 반대.
    - **antiknife(물타기·투매 반대):** -ri × relu(-pr). 내린 종목(pr<0)에서 개미 페이드 —
      개미가 받는 낙폭주는 숏(물타기 회피), 개미가 던진 낙폭주는 롱(투매 역발상 매수).
    - **composite(등락 극단 페이드):** -ri × |pr|. 등락이 클수록 개미를 강하게 페이드
      (개미의 실수는 큰 움직임 끝에 집중된다는 가설).

    signal 컬럼으로 long code/date/signal 패널 dict 반환(engine 주입 형식).
    """
    px = prices[["code", "date", "close"]]
    m = flow.merge(px, on=["code", "date"], how="inner").sort_values(["code", "date"])
    g = m.groupby("code", sort=False)
    ind_sum = g["individual"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    vol_sum = g["volume"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    ri = ind_sum / vol_sum.where(vol_sum > 0)
    pr = g["close"].transform(lambda s: s / s.shift(window) - 1.0)
    ri_l = g_shift(ri, m["code"], lag).to_numpy()
    pr_l = g_shift(pr, m["code"], lag).to_numpy()
    relu_p = np.where(pr_l > 0, pr_l, 0.0)
    relu_n = np.where(pr_l < 0, -pr_l, 0.0)
    m["antichase"] = -ri_l * relu_p
    m["antiknife"] = -ri_l * relu_n
    m["composite"] = -ri_l * np.abs(pr_l)
    sigs = {}
    for name in ("antichase", "antiknife", "composite"):
        sigs[name] = (m[["code", "date", name]]
                      .rename(columns={name: "signal"})
                      .dropna(subset=["signal"]))
    return sigs


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


def build_control_signals(
    flow: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    window: int = SIGNAL_WINDOW,
    lag: int = SIGNAL_LAG,
) -> dict[str, pd.DataFrame]:
    """교란 통제 신호: antichase가 '개미 추격 페이드'인지 순수 단기반전인지 가른다.

    - mom_reversal: -pr (개미 수급 無, 순수 과거수익 반전). antichase IC와 비슷하면 confound.
    - flow_in_winners: pr>0 종목 안에서만 -ri (모멘텀 중립화한 순수 개미 수급 방향).
      유의하면 개미 수급 자체가 오른 종목에서 예측력 있다는 뜻 → 진짜 '추격 페이드'.
    - flow_in_losers: pr<0 종목 안에서만 -ri (하락장 개미 수급 방향).
    """
    px = prices[["code", "date", "close"]]
    m = flow.merge(px, on=["code", "date"], how="inner").sort_values(["code", "date"])
    g = m.groupby("code", sort=False)
    ind_sum = g["individual"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    vol_sum = g["volume"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    ri = g_shift(ind_sum / vol_sum.where(vol_sum > 0), m["code"], lag).to_numpy()
    pr = g_shift(g["close"].transform(lambda s: s / s.shift(window) - 1.0), m["code"], lag).to_numpy()
    m["mom_reversal"] = -pr
    win = pr > 0
    los = pr < 0
    m["flow_in_winners"] = np.where(win, -ri, np.nan)
    m["flow_in_losers"] = np.where(los, -ri, np.nan)
    sigs = {}
    for name in ("mom_reversal", "flow_in_winners", "flow_in_losers"):
        sigs[name] = (m[["code", "date", name]]
                      .rename(columns={name: "signal"})
                      .dropna(subset=["signal"]))
    return sigs


def build_smart_signal(
    flow: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    window: int = SIGNAL_WINDOW,
    lag: int = SIGNAL_LAG,
    with_momentum: float = 0.0,
) -> pd.DataFrame:
    """학습 통합 신호: signal = -ri × sign(pr)  (오른데선 개미반대, 내린데선 개미편승).

    - ri>0(개미매수)·pr>0(상승) = 추격 → -ri<0 → 숏(추격 페이드)
    - ri<0(개미매도)·pr>0(상승) = 익절·존버 → +>0 → 롱(개미가 던진 급등주 매집)
    - ri>0(개미매수)·pr<0(하락) = 물타기 → +>0 → 롱(반등 편승)
    - ri<0(개미매도)·pr<0(하락) = 투매 → -<0 → 숏(패닉엔 동참 안함)

    with_momentum>0이면 순수 단기반전(-pr의 z점수)을 그만큼 가중 결합(보조 신호).
    """
    px = prices[["code", "date", "close"]]
    m = flow.merge(px, on=["code", "date"], how="inner").sort_values(["code", "date"])
    g = m.groupby("code", sort=False)
    ind_sum = g["individual"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    vol_sum = g["volume"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    ri = g_shift(ind_sum / vol_sum.where(vol_sum > 0), m["code"], lag).to_numpy()
    pr = g_shift(g["close"].transform(lambda s: s / s.shift(window) - 1.0), m["code"], lag).to_numpy()
    sig = -ri * np.sign(pr)
    if with_momentum > 0:
        # 두 신호를 종목횡단이 아니라 시계열 표준화로 대략 정규화 후 결합(간이)
        rev = -pr
        sig = _zc(sig) + with_momentum * _zc(rev)
    m["signal"] = sig
    return m[["code", "date", "signal"]].dropna(subset=["signal"])


def _zc(x: np.ndarray) -> np.ndarray:
    """유한값 표준화(z)."""
    fin = np.isfinite(x)
    if fin.sum() < 2:
        return x
    mu, sd = np.nanmean(x[fin]), np.nanstd(x[fin])
    return (x - mu) / sd if sd > 0 else x - mu


def ic_weighted_book(
    prices: pd.DataFrame,
    signal_panel: pd.DataFrame,
    *,
    horizon: int = 20,
    step: int = 5,
    adv_floor: float = 20000.0,
    adv_window: int = 20,
    start_index: int = 130,
    min_names: int = 20,
    cost_one_way: float = 0.0,
    date_lo: str | None = None,
    date_hi: str | None = None,
) -> dict:
    """랭크가중 마켓뉴트럴 북 — 강한 IC를 실제 수익으로 전환(단순 상하위 롱숏보다 IC 활용).

    매 리밸런스: 유동성 유니버스에서 신호를 랭크 → 평균제거 → Σ|w|=1로 정규화(달러중립,
    Σw=0). 포트수익 = Σ w_i·fwd_i. 회전율×비용 차감. 룩어헤드 없음(신호는 t-lag까지).
    """
    C, V, S, dates = _panels(prices, signal_panel)
    nD = C.shape[1]
    di = np.array([str(d) for d in dates])
    rets, turns = [], []
    prev_w: dict[int, float] = {}
    for t in range(start_index, nD - horizon - 1, step):
        if date_lo and di[t] < date_lo:
            continue
        if date_hi and di[t] >= date_hi:
            continue
        adv = np.nanmean(V[:, t - adv_window:t], axis=1)
        elig = adv >= adv_floor
        s_t = np.where(elig, S[:, t], np.nan)
        fin = np.isfinite(s_t)
        if fin.sum() < min_names:
            continue
        idx = np.where(fin)[0]
        r = pd.Series(s_t[idx]).rank().to_numpy()
        w = r - r.mean()
        gross = np.abs(w).sum()
        if gross == 0:
            continue
        w = w / gross  # Σ|w|=1, Σw=0 (달러중립)
        fwd = C[idx, t + horizon] / C[idx, t] - 1.0
        valid = np.isfinite(fwd)
        rets.append(float(np.sum(w[valid] * fwd[valid])))
        wt = {int(idx[i]): float(w[i]) for i in range(len(idx))}
        allk = set(wt) | set(prev_w)
        turns.append(float(sum(abs(wt.get(k, 0.0) - prev_w.get(k, 0.0)) for k in allk)))
        prev_w = wt
    rets = np.array(rets)
    turns = np.array(turns)
    gross_rets = rets.copy()
    if cost_one_way:
        rets = rets - turns * cost_one_way
    per_year = 252 / step
    if len(rets) == 0 or rets.std() == 0:
        return {"n": len(rets), "sharpe": float("nan"), "t": float("nan"),
                "cum": float("nan"), "mdd": float("nan"), "turnover": float("nan")}
    return {
        "n": len(rets),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(per_year)),
        "t": float(rets.mean() / (rets.std() / np.sqrt(len(rets)))),
        "cum": float(np.prod(1 + rets) - 1),
        "mdd": _mdd(rets),
        "turnover": float(turns.mean()),
        "sharpe_gross": float(gross_rets.mean() / gross_rets.std() * np.sqrt(per_year)),
    }


def _run_behavior(prices: pd.DataFrame, flow: pd.DataFrame) -> int:
    """심리적 실수(추격·물타기·투매) 반대 신호를 진단. 각 신호 정반대로 롱숏·IC."""
    labels = {"antichase": "추격 반대(오른데서 개미 페이드)",
              "antiknife": "물타기·투매 반대(내린데서 개미 페이드)",
              "composite": "등락극단 페이드(-ri×|과거수익|)"}
    print("\n=== 심리 반대매매 진단 (부호 격리: 롱숏 유니버스중립 + 랭크-IC) ===")
    results = {}
    for w in (5, 20):
        sigs = build_behavior_signals(flow, prices, window=w)
        for name, panel in sigs.items():
            for h in (5, 20):
                dg = diagnose(prices, panel, horizon=h, step=5)
                ic, ls = dg["rank_ic"], dg["long_short"]
                results[(w, name, h)] = (ic, ls)
                print(f"  [{name:9} w{w:>2} h{h:>2}] IC t={ic['t']:+.2f}  "
                      f"롱숏 t={ls['t']:+.2f} Sharpe={ls['sharpe']:+.2f} cum={ls['cum']:+.3f}  "
                      f"({labels[name]})")
    mx = max(abs(ls['t']) for _, ls in results.values())
    best = max(results.items(), key=lambda kv: abs(kv[1][1]['t']))
    print(f"\n  전 조합 |롱숏 t| 최대 = {mx:.2f} @ {best[0]}")
    print(f"  → {'2 미만이면 심리 반대매매도 알파 아님' if mx < 2 else '2 이상 — 하위기간·다중검정 보정 필요'}")

    if mx >= 2.0:
        w, name, h = best[0]
        print(f"\n=== 최고 신호 하위기간 견고성: {name} w{w} h{h} ===")
        sigs = build_behavior_signals(flow, prices, window=w)
        for lo, hi, nm in [("2017-01-01", "2022-01-01", "2017-2021"),
                           ("2022-01-01", "2027-01-01", "2022-2026")]:
            dd = diagnose(prices, sigs[name], horizon=h, step=5, date_lo=lo, date_hi=hi)["long_short"]
            print(f"  {nm}: 롱숏 t={dd['t']:+.2f} Sharpe={dd['sharpe']:+.2f} cum={dd['cum']:+.3f}")
        n_combos = len(results)
        print(f"\n  ⚠️ 다중검정: {n_combos}개 조합 중 최고이므로 유효 유의수준은 대략 t≈{2 + 0.5:.1f}+ 요구"
              f" (Bonferroni 감안). 하위기간 둘 다 살아야 진짜.")

    print("\n=== 교란 통제: antichase가 개미효과인가 순수 단기반전인가 ===")
    for w in (5, 20):
        ctrl = build_control_signals(flow, prices, window=w)
        for name in ("mom_reversal", "flow_in_winners", "flow_in_losers"):
            for h in (5, 20):
                dg = diagnose(prices, ctrl[name], horizon=h, step=5)
                ic = dg["rank_ic"]
                print(f"  [{name:15} w{w:>2} h{h:>2}] IC t={ic['t']:+.2f}  롱숏 t={dg['long_short']['t']:+.2f}")
    print("  판독: mom_reversal IC ≈ antichase IC면 그냥 반전(개미 스토리 허구). "
          "flow_in_winners가 유의하면 오른종목 안 개미수급 자체가 예측력(진짜 추격 페이드).")
    return 0


def _signal_panels(prices, flow, window, lag):
    """close·trade_value·ri(개미강도)·pr(과거수익)을 동일 code×date 그리드로 (모두 lag 적용)."""
    from kr_quant.engine.panels import panel_pivot
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    codes, dates = list(close.index), list(close.columns)
    px = prices[["code", "date", "close"]]
    m = flow.merge(px, on=["code", "date"], how="inner").sort_values(["code", "date"])
    g = m.groupby("code", sort=False)
    ind = g["individual"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    vol = g["volume"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    m["ri"] = g_shift(ind / vol.where(vol > 0), m["code"], lag)
    m["pr"] = g_shift(g["close"].transform(lambda s: s / s.shift(window) - 1.0), m["code"], lag)
    def piv(col):
        return (m.pivot_table(index="code", columns="date", values=col, aggfunc="first")
                .reindex(index=codes, columns=dates).to_numpy(float))
    return (close.to_numpy(float), tval.reindex(index=codes, columns=dates).to_numpy(float),
            piv("ri"), piv("pr"), dates)


def simulate_trades(prices, flow, *, window=5, lag=1, hold=20, stop=0.08, target=0.15,
                    mom_q=0.7, ext_q=0.9, adv_floor=20000.0, adv_window=20, start_index=130,
                    cost_roundtrip=0.0046, side="both"):
    """이벤트드리븐 트레이드: 급등(모멘텀 상위) 종목 중 개미 극단 추격→숏 / 극단 투매→롱.

    진입 close[t](신호 lag로 인과적), 청산 = 손절 −stop / 목표 +target / 시간 hold 중 먼저.
    한 종목은 청산 전까지 재진입 안 함(중복 방지). 반환: 트레이드 dict 리스트.
    """
    C, V, RI, PR, dates = _signal_panels(prices, flow, window, lag)
    _, nD = C.shape
    di = np.array([str(d) for d in dates])
    busy_until: dict[int, int] = {}
    trades = []
    for t in range(start_index, nD - 1):
        adv = np.nanmean(V[:, t - adv_window:t], axis=1)
        pr_t, ri_t, c_t = PR[:, t], RI[:, t], C[:, t]
        valid = (adv >= adv_floor) & np.isfinite(pr_t) & np.isfinite(ri_t) & np.isfinite(c_t) & (c_t > 0)
        if valid.sum() < 20:
            continue
        winners = valid & (pr_t >= np.quantile(pr_t[valid], mom_q))
        if winners.sum() < 10:
            continue
        riw = ri_t[winners]
        hi, lo = np.quantile(riw, ext_q), np.quantile(riw, 1 - ext_q)
        if not hi > lo:
            continue
        chase = set(np.where(winners & (ri_t >= hi))[0])   # 개미 추격 → 숏
        dump = set(np.where(winners & (ri_t <= lo))[0])     # 개미 투매/익절 → 롱
        for idx in chase | dump:
            if busy_until.get(idx, -1) > t:
                continue
            sd = +1 if idx in dump else -1
            if (side == "long" and sd < 0) or (side == "short" and sd > 0):
                continue
            entry = C[idx, t]
            ex_t, reason = min(t + hold, nD - 1), "time"
            for u in range(t + 1, min(t + hold, nD - 1) + 1):
                cu = C[idx, u]
                if not np.isfinite(cu) or cu <= 0:
                    continue
                rr = (cu / entry - 1.0) * sd  # 포지션 방향 수익
                if rr <= -stop:
                    ex_t, reason = u, "stop"
                    break
                if rr >= target:
                    ex_t, reason = u, "target"
                    break
            r = (C[idx, ex_t] / entry - 1.0) * sd - cost_roundtrip
            trades.append({"entry": di[t], "exit": di[ex_t], "side": sd,
                           "ret": float(r), "reason": reason, "bars": ex_t - t})
            busy_until[idx] = ex_t
    return trades


def simulate_momentum_long(prices, flow, *, window=5, lag=1, hold=20, stop=0.08, target=0.15,
                           trail=0.0, top_mom=0.8, retail_rule="all", ext_q=0.8, adv_floor=20000.0,
                           adv_window=20, start_index=130, cost_roundtrip=0.0046, seed=0):
    """롱온리 모멘텀 매매 + 개미 수급 필터. 필터가 순수 모멘텀을 이기는지 검증.

    진입: 모멘텀 상위(top_mom 분위 이상 급등주) 중 retail_rule로 선별 → 롱.
      - "all":     급등주 전부 (순수 모멘텀 베이스라인)
      - "ex_fomo": 개미 극단추격(ext_q 상위) 제외
      - "dump":    개미 투매/익절(ext_q 하위)만
      - "random":  dump와 같은 개수(하위분위 크기)를 랜덤 선택 (선택이 개미때문인지 아티팩트인지 대조군)
    청산: 손절(−stop) 우선. trail>0이면 **트레일링 스탑**(고점대비 −trail, 수익권에서만)으로
    수익을 달리게 둔다(손익비 중시). target>0이면 고정 목표 익절도 병행. 시간(hold) 상한.
    한 종목 청산 전 재진입 없음.
    """
    C, V, RI, PR, dates = _signal_panels(prices, flow, window, lag)
    _, nD = C.shape
    di = np.array([str(d) for d in dates])
    rng = np.random.default_rng(seed)
    busy_until: dict[int, int] = {}
    trades = []
    for t in range(start_index, nD - 1):
        adv = np.nanmean(V[:, t - adv_window:t], axis=1)
        pr_t, ri_t, c_t = PR[:, t], RI[:, t], C[:, t]
        valid = (adv >= adv_floor) & np.isfinite(pr_t) & np.isfinite(ri_t) & np.isfinite(c_t) & (c_t > 0)
        if valid.sum() < 20:
            continue
        cand = valid & (pr_t >= _pctl(np.sort(pr_t[valid]), top_mom))
        if cand.sum() < 5:
            continue
        riw = ri_t[cand]
        rsrt = np.sort(riw)
        hi, lo = _pctl(rsrt, ext_q), _pctl(rsrt, 1 - ext_q)
        if retail_rule == "all":
            entries = np.where(cand)[0]
        elif retail_rule == "ex_fomo":
            entries = np.where(cand & (ri_t < hi))[0]
        elif retail_rule == "random":
            cidx = np.where(cand)[0]
            k = int(round((1 - ext_q) * len(cidx)))  # dump와 동일 개수(하위분위 크기)
            entries = rng.choice(cidx, size=max(k, 1), replace=False) if k >= 1 else cidx[:0]
        else:  # dump
            entries = np.where(cand & (ri_t <= lo))[0]
        for idx in entries:
            if busy_until.get(idx, -1) > t:
                continue
            entry = C[idx, t]
            ex_t, reason = min(t + hold, nD - 1), "time"
            peak = entry
            for u in range(t + 1, min(t + hold, nD - 1) + 1):
                cu = C[idx, u]
                if not np.isfinite(cu) or cu <= 0:
                    continue
                rr = cu / entry - 1.0
                if rr <= -stop:
                    ex_t, reason = u, "stop"
                    break
                peak = max(peak, cu)
                if trail > 0 and cu > entry and cu <= peak * (1 - trail):
                    ex_t, reason = u, "trail"   # 고점대비 −trail% 반납 → 수익 확정(달리게 둔 뒤)
                    break
                if target > 0 and rr >= target:
                    ex_t, reason = u, "target"
                    break
            r = (C[idx, ex_t] / entry - 1.0) - cost_roundtrip
            trades.append({"entry": di[t], "exit": di[ex_t], "side": 1,
                           "ret": float(r), "reason": reason, "bars": ex_t - t})
            busy_until[idx] = ex_t
    return trades


def _adv_panel(V: np.ndarray, adv_window: int) -> np.ndarray:
    """거래대금 배열 → 20일 평균(ADV) 배열. t시점 값은 [t-window, t) 평균(룩어헤드 없음)."""
    nC, nD = V.shape
    adv = np.full((nC, nD), np.nan)
    for t in range(adv_window, nD):
        adv[:, t] = np.nanmean(V[:, t - adv_window:t], axis=1)
    return adv


def _pctl(sorted_a, q):
    """정렬된 배열의 q분위 (np.quantile 기본 'linear' 보간과 동일 공식).

    numba/순수파이썬 양쪽 경로가 **같은 함수**를 써야 임계값이 bit-identical → 트레이드
    선택이 일치(np.quantile과 _pctl의 2e-16 차이가 임계 경계에서 종목을 뒤집어 busy상태로
    연쇄되는 것을 방지). 아래 try에서 numba 존재 시 njit으로 감싼다.
    """
    n = len(sorted_a)
    if n == 1:
        return sorted_a[0]
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = lo + 1
    if hi >= n:
        return sorted_a[n - 1]
    frac = pos - lo
    return sorted_a[lo] * (1.0 - frac) + sorted_a[hi] * frac


try:
    import numba

    _pctl = numba.njit(cache=True)(_pctl)

    @numba.njit(cache=True)
    def _sim_core(C, ADV, RI, PR, start_index, adv_floor, top_mom, ext_q,
                  stop, target, trail, hold, cost_rt):
        """simulate_momentum_long(retail_rule='dump')의 njit 등가 — 트레이드 수익·진입일 반환.

        진입: ADV통과 급등주(모멘텀 top_mom 분위↑) 중 개미강도 하위(1-ext_q 분위↓).
        청산: 손절(−stop) > 트레일(고점−trail, 수익권) > 목표(+target) > 시간(hold). 재진입 금지.
        """
        nC, nD = C.shape
        busy = np.full(nC, -1, np.int64)
        max_tr = nC * (nD // 5 + 1)
        rets = np.empty(max_tr)
        edays = np.empty(max_tr, np.int64)
        emom = np.empty(max_tr)       # 진입 모멘텀 강도(PR)
        edump = np.empty(max_tr)      # 개미 투매 강도(-RI, 클수록 개미가 많이 던짐)
        ebars = np.empty(max_tr, np.int64)
        ereason = np.empty(max_tr, np.int64)  # 0 time,1 stop,2 trail,3 target
        nt = 0
        prbuf = np.empty(nC)
        ribuf = np.empty(nC)
        for t in range(start_index, nD - 1):
            m = 0
            for i in range(nC):
                c = C[i, t]
                if (ADV[i, t] >= adv_floor and PR[i, t] == PR[i, t]
                        and RI[i, t] == RI[i, t] and c == c and c > 0):
                    prbuf[m] = PR[i, t]
                    m += 1
            if m < 20:
                continue
            mom_thr = _pctl(np.sort(prbuf[:m]), top_mom)
            w = 0
            for i in range(nC):
                c = C[i, t]
                if (ADV[i, t] >= adv_floor and PR[i, t] == PR[i, t]
                        and RI[i, t] == RI[i, t] and c == c and c > 0 and PR[i, t] >= mom_thr):
                    ribuf[w] = RI[i, t]
                    w += 1
            if w < 10:
                continue
            lo = _pctl(np.sort(ribuf[:w]), 1.0 - ext_q)
            for i in range(nC):
                c = C[i, t]
                if not (ADV[i, t] >= adv_floor and PR[i, t] == PR[i, t]
                        and RI[i, t] == RI[i, t] and c == c and c > 0
                        and PR[i, t] >= mom_thr and RI[i, t] <= lo):
                    continue
                if busy[i] > t:
                    continue
                entry = C[i, t]
                cap_t = t + hold if t + hold < nD - 1 else nD - 1
                ex_t = cap_t
                reason = 0
                peak = entry
                uu = t + 1
                while uu <= cap_t:
                    cu = C[i, uu]
                    if cu == cu and cu > 0:
                        rr = cu / entry - 1.0
                        if rr <= -stop:
                            ex_t = uu
                            reason = 1
                            break
                        if cu > peak:
                            peak = cu
                        if trail > 0 and cu > entry and cu <= peak * (1.0 - trail):
                            ex_t = uu
                            reason = 2
                            break
                        if target > 0 and rr >= target:
                            ex_t = uu
                            reason = 3
                            break
                    uu += 1
                if nt < max_tr:
                    rets[nt] = (C[i, ex_t] / entry - 1.0) - cost_rt
                    edays[nt] = t
                    emom[nt] = PR[i, t]
                    edump[nt] = -RI[i, t]
                    ebars[nt] = ex_t - t
                    ereason[nt] = reason
                    nt += 1
                busy[i] = ex_t
        return rets[:nt], edays[:nt], emom[:nt], edump[:nt], ebars[:nt], ereason[:nt]

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


def simulate_fast(prices, flow, *, window=5, lag=1, hold=20, stop=0.10, target=0.0,
                  trail=0.20, top_mom=0.8, ext_q=0.8, adv_floor=20000.0, adv_window=20,
                  start_index=130, cost_roundtrip=0.0046, _cache=None):
    """numba 가속 dump 시뮬 → (트레이드 수익 배열, 진입일 문자열 배열).

    _cache: {(window,lag,adv_window): (C, ADV, RI, PR, dates)} 딕셔너리를 넘기면 패널 빌드를
    조합별 1회로 재사용(BO 수백 trial에서 필수). RI/PR은 (window,lag), ADV는 adv_window에
    의존하므로 셋을 키로 잡아야 구조상수 스윕에서도 정확. None이면 매번 빌드.
    """
    key = (window, lag, adv_window)
    if _cache is not None and key in _cache:
        C, ADV, RI, PR, dates = _cache[key]
    else:
        C, V, RI, PR, dates = _signal_panels(prices, flow, window, lag)
        ADV = _adv_panel(V, adv_window)
        if _cache is not None:
            _cache[key] = (C, ADV, RI, PR, dates)
    rets, edays = _sim_core(C, ADV, RI, PR, start_index, adv_floor, top_mom, ext_q,
                            stop, target, trail, hold, cost_roundtrip)[:2]
    di = np.array([str(d) for d in dates])
    entry_dates = di[edays] if len(edays) else np.array([], dtype=object)
    return np.asarray(rets), entry_dates


def simulate_detailed(prices, flow, *, window=8, lag=1, hold=60, stop=0.10, target=0.0,
                      trail=0.20, top_mom=0.8, ext_q=0.85, adv_floor=20000.0, adv_window=20,
                      start_index=130, cost_roundtrip=0.0046, _cache=None):
    """분포 분석용 상세 시뮬 → 트레이드별 (수익·진입일·모멘텀강도·투매강도·보유일·청산사유).

    ret: 비용후 수익, mom: 진입 5일수익(모멘텀 강도), dump: 개미 투매 강도(-ri),
    bars: 보유일, reason: 0 time/1 stop/2 trail/3 target. 확신 사이징·꼬리 예측 분석에 사용.
    """
    key = (window, lag, adv_window)
    if _cache is not None and key in _cache:
        C, ADV, RI, PR, dates = _cache[key]
    else:
        C, V, RI, PR, dates = _signal_panels(prices, flow, window, lag)
        ADV = _adv_panel(V, adv_window)
        if _cache is not None:
            _cache[key] = (C, ADV, RI, PR, dates)
    rets, edays, emom, edump, ebars, ereason = _sim_core(
        C, ADV, RI, PR, start_index, adv_floor, top_mom, ext_q,
        stop, target, trail, hold, cost_roundtrip)
    di = np.array([str(d) for d in dates])
    return {
        "ret": np.asarray(rets),
        "entry": di[edays] if len(edays) else np.array([], dtype=object),
        "mom": np.asarray(emom),
        "dump": np.asarray(edump),
        "bars": np.asarray(ebars),
        "reason": np.asarray(ereason),
    }


def fast_stats(rets: np.ndarray, entry_dates: np.ndarray, *, date_lo=None, date_hi=None) -> dict:
    """simulate_fast 출력(수익배열)→트레이더 지표. NaN 안전(유한값만). trade_stats와 동일 스키마."""
    r, ed = rets, entry_dates
    if date_lo is not None or date_hi is not None:
        mask = np.ones(len(r), bool)
        if date_lo is not None:
            mask &= ed >= date_lo
        if date_hi is not None:
            mask &= ed < date_hi
        r, ed = r[mask], ed[mask]
    yrs = 1
    if len(ed):
        yy = sorted(str(x)[:4] for x in ed)
        yrs = max(int(yy[-1]) - int(yy[0]) + 1, 1)
    r = r[np.isfinite(r)]
    if len(r) < 1:
        return {"n": 0}
    wins, losses = r[r > 0], r[r < 0]
    return {
        "n": len(r),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(-losses.mean()) if len(losses) else 0.0,
        "payoff": float(wins.mean() / -losses.mean()) if len(losses) and len(wins) else float("nan"),
        "expectancy": float(r.mean()),
        "expectancy_t": float(r.mean() / (r.std() / np.sqrt(len(r)))) if r.std() > 0 else float("nan"),
        "trades_per_yr": len(r) / yrs,
        "sum_ret": float(r.sum()),
    }


def trade_stats(trades: list, *, date_lo=None, date_hi=None) -> dict:
    """트레이드 리스트 → 트레이더 지표: n, 승률, 손익비, 건당 기대값, 연 P&L 근사."""
    tr = [x for x in trades
          if (not date_lo or x["entry"] >= date_lo) and (not date_hi or x["entry"] < date_hi)]
    if not tr:
        return {"n": 0}
    r = np.array([x["ret"] for x in tr])
    wins, losses = r[r > 0], r[r < 0]
    bars = np.array([x["bars"] for x in tr])
    years = sorted(x["entry"][:4] for x in tr)
    yrs = int(years[-1]) - int(years[0]) + 1
    return {
        "n": len(tr),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(-losses.mean()) if len(losses) else 0.0,
        "payoff": float(wins.mean() / -losses.mean()) if len(losses) and len(wins) else float("nan"),
        "expectancy": float(r.mean()),
        "expectancy_t": float(r.mean() / (r.std() / np.sqrt(len(r)))) if r.std() > 0 else float("nan"),
        "avg_bars": float(bars.mean()),
        "trades_per_yr": len(tr) / max(yrs, 1),
        "sum_ret": float(r.sum()),
    }


def _run_trade(prices: pd.DataFrame, flow: pd.DataFrame, cost: float) -> int:
    """이벤트드리븐 트레이드 실험: 개미 극단행동 페이드를 건별 매매로."""
    rt = 2 * cost  # 왕복 비용
    print(f"\n=== 이벤트드리븐 트레이드 (급등주 개미 극단 페이드, 왕복비용 {rt*1e4:.0f}bp) ===")
    print("  진입: 모멘텀 상위30% 중 개미강도 극단(상위10%=추격→숏, 하위10%=투매→롱)")
    print("  청산: 손절 −8% / 목표 +15% / 시간 20일 중 먼저\n")
    base = simulate_trades(prices, flow, cost_roundtrip=rt)
    for side_name, sub in [("전체", base),
                           ("롱(개미투매 매집)", [t for t in base if t["side"] > 0]),
                           ("숏(개미추격 페이드)", [t for t in base if t["side"] < 0])]:
        s = trade_stats(sub)
        if s["n"] == 0:
            print(f"  [{side_name}] 트레이드 없음")
            continue
        print(f"  [{side_name:16}] n={s['n']:>4} 승률={s['win_rate']:.1%} "
              f"손익비={s['payoff']:.2f} 기대값={s['expectancy']:+.4f}(t={s['expectancy_t']:+.2f}) "
              f"평균{s['avg_bars']:.0f}일 연{s['trades_per_yr']:.0f}건 총{s['sum_ret']:+.2f}")

    print("\n  하위기간 (전체):")
    for lo, hi, nm in [("2017-01-01", "2022-01-01", "2017-2021"),
                       ("2022-01-01", "2027-01-01", "2022-2026")]:
        s = trade_stats(base, date_lo=lo, date_hi=hi)
        if s["n"]:
            print(f"    {nm}: n={s['n']:>4} 승률={s['win_rate']:.1%} "
                  f"기대값={s['expectancy']:+.4f}(t={s['expectancy_t']:+.2f}) 총{s['sum_ret']:+.2f}")

    print("\n  파라미터 스윕 (극단 임계·보유기간, 전체 건당 기대값 t):")
    print(f"  {'ext_q':>5} {'hold':>4} {'stop':>5} {'n':>5} {'win':>5} {'exp_t':>6} {'sum':>7}")
    for eq in (0.85, 0.90, 0.95):
        for hd in (10, 20, 40):
            tr = simulate_trades(prices, flow, ext_q=eq, hold=hd, cost_roundtrip=rt)
            s = trade_stats(tr)
            if s["n"]:
                print(f"  {eq:>5.2f} {hd:>4} {'8%':>5} {s['n']:>5} {s['win_rate']:>5.1%} "
                      f"{s['expectancy_t']:>+6.2f} {s['sum_ret']:>+7.2f}")
    return 0


def _boot_ci(rets: np.ndarray, *, n_boot: int = 3000, seed: int = 0) -> tuple[float, float]:
    """건당 기대값의 부트스트랩 95% 신뢰구간."""
    if len(rets) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(rets, size=(n_boot, len(rets)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _run_payoff(prices: pd.DataFrame, flow: pd.DataFrame, cost: float) -> int:
    """손익비 중심 청산 최적화: 고정목표(수익 잘림) vs 트레일링(수익 달림). 승률 무시."""
    rt = 2 * cost
    print(f"\n=== 손익비 중심: 개미투매 급등주, 청산방식 비교 (왕복 {rt*1e4:.0f}bp) ===")
    print("  승률 아닌 손익비(평균수익/평균손실)·기대값이 판정. 손절 −10% 고정, 보유상한 60일.\n")
    print(f"  {'청산방식':22} {'n':>5} {'승률':>5} {'평균수익':>7} {'평균손실':>7} {'손익비':>6} {'기대값':>9} {'총':>7}")
    configs = [
        ("고정목표 +15%(기존)", dict(target=0.15, trail=0.0, hold=20)),
        ("고정목표 +30%", dict(target=0.30, trail=0.0, hold=60)),
        ("트레일 10%(목표無)", dict(target=0.0, trail=0.10, hold=60)),
        ("트레일 15%(목표無)", dict(target=0.0, trail=0.15, hold=60)),
        ("트레일 20%(목표無)", dict(target=0.0, trail=0.20, hold=60)),
        ("트레일 15%+보유120", dict(target=0.0, trail=0.15, hold=120)),
    ]
    best = None
    for label, cfg in configs:
        tr = simulate_momentum_long(prices, flow, retail_rule="dump", stop=0.10,
                                    cost_roundtrip=rt, **cfg)
        s = trade_stats(tr)
        if not s["n"]:
            continue
        print(f"  {label:22} {s['n']:>5} {s['win_rate']:>5.1%} {s['avg_win']:>+7.3f} "
              f"{-s['avg_loss']:>+7.3f} {s['payoff']:>6.2f} {s['expectancy']:>+9.4f} {s['sum_ret']:>+7.1f}")
        if best is None or s["expectancy"] > best[1]["expectancy"]:
            best = (label, s, cfg, tr)

    label, s, cfg, tr = best
    print(f"\n  최고 기대값: {label} — 손익비 {s['payoff']:.2f}, 건당 {s['expectancy']:+.4f}, "
          f"평균보유 {s['avg_bars']:.0f}일")
    lo, hi = _boot_ci(np.array([x["ret"] for x in tr]))
    print(f"  부트스트랩 95%CI: [{lo:+.4f}, {hi:+.4f}]  ({'0 초과 — 유의' if lo > 0 else '0 걸침'})")
    print("\n  하위기간(최고명세):")
    for a, b, nm in [("2017-01-01", "2022-01-01", "2017-2021"),
                     ("2022-01-01", "2027-01-01", "2022-2026")]:
        ss = trade_stats(tr, date_lo=a, date_hi=b)
        if ss["n"]:
            print(f"    {nm}: n={ss['n']:>4} 손익비={ss['payoff']:.2f} 기대값={ss['expectancy']:+.4f}"
                  f"(t{ss['expectancy_t']:+.1f})")
    return 0


def _run_harden(prices: pd.DataFrame, flow: pd.DataFrame, cost: float) -> int:
    """강체화: (1)선택이 개미때문인지 랜덤대조, (2)명세 강건성, (3)부트스트랩 CI."""
    rt = 2 * cost
    print(f"\n=== 강체화 검증: 롱온리 모멘텀 + 개미투매 필터 (왕복 {rt*1e4:.0f}bp) ===")

    print("\n[1] 선택 아티팩트 배제 — dump vs 같은개수 랜덤 급등주")
    dump = simulate_momentum_long(prices, flow, retail_rule="dump", cost_roundtrip=rt)
    d = trade_stats(dump)
    dlo, dhi = _boot_ci(np.array([t["ret"] for t in dump]))
    rand_exps = []
    for sd in range(5):
        rtr = simulate_momentum_long(prices, flow, retail_rule="random", seed=sd, cost_roundtrip=rt)
        rand_exps.append(trade_stats(rtr)["expectancy"])
    rand_exps = np.array(rand_exps)
    print(f"    dump  : n={d['n']:>5} 기대값={d['expectancy']:+.4f} (95%CI [{dlo:+.4f},{dhi:+.4f}]) t={d['expectancy_t']:+.2f}")
    print(f"    random: 기대값 평균={rand_exps.mean():+.4f} (범위 [{rand_exps.min():+.4f},{rand_exps.max():+.4f}], 5시드)")
    verdict = "PASS — 개미 선택이 랜덤을 이김(엣지는 개미신호에서)" if d["expectancy"] > rand_exps.max() \
        else "FAIL — 랜덤과 구분 안됨(선택 무의미)"
    print(f"    → {verdict}")

    print("\n[2] 명세 강건성 — 청산룰 바꿔도 dump>all 유지되나 (dump기대값 | all기대값)")
    print(f"    {'stop':>5} {'target':>6} {'hold':>4} {'dump_exp':>9} {'all_exp':>9} {'dump>all':>8}")
    robust = 0
    total = 0
    for st in (0.05, 0.08, 0.12):
        for (tg, hd) in [(0.10, 10), (0.15, 20), (0.25, 40)]:
            dd = trade_stats(simulate_momentum_long(prices, flow, retail_rule="dump",
                             stop=st, target=tg, hold=hd, cost_roundtrip=rt))
            aa = trade_stats(simulate_momentum_long(prices, flow, retail_rule="all",
                             stop=st, target=tg, hold=hd, cost_roundtrip=rt))
            win = dd["expectancy"] > aa["expectancy"]
            robust += int(win)
            total += 1
            print(f"    {st:>5.2f} {tg:>6.2f} {hd:>4} {dd['expectancy']:>+9.4f} "
                  f"{aa['expectancy']:>+9.4f} {'✓' if win else '✗':>8}")
    print(f"    → dump이 all을 이긴 명세: {robust}/{total} "
          f"({'강건' if robust >= total - 1 else '취약'})")

    print("\n[3] 종합 판정")
    ok_sel = d["expectancy"] > rand_exps.max()
    ok_pos = dlo > 0
    ok_rob = robust >= total - 1
    print(f"    선택유효={ok_sel}  기대값CI>0={ok_pos}  명세강건={ok_rob}")
    print(f"    → {'배포후보(단, 크기는 작음)' if ok_sel and ok_rob else '엣지 실재하나 확정불가 — 추가데이터/개선 필요'}")
    return 0


def _run_money(prices: pd.DataFrame, flow: pd.DataFrame, cost: float) -> int:
    """수익 검증: 롱온리 모멘텀에 개미 필터가 순수 모멘텀(baseline)을 이기나."""
    rt = 2 * cost
    print(f"\n=== 수익 각도: 롱온리 모멘텀 + 개미 필터 (왕복 {rt*1e4:.0f}bp) ===")
    print("  진입 급등주(모멘텀 상위20%), 청산 손절-8%/목표+15%/시간20일. 필터가 baseline 이기면 엣지.\n")
    print(f"  {'rule':10} {'n':>5} {'승률':>5} {'손익비':>5} {'기대값':>9} {'exp_t':>6} {'총손익':>7}")
    rules = {"all(순수모멘텀)": "all", "ex_fomo(추격제외)": "ex_fomo", "dump(개미투매만)": "dump"}
    trs = {}
    for label, rule in rules.items():
        tr = simulate_momentum_long(prices, flow, retail_rule=rule, cost_roundtrip=rt)
        trs[rule] = tr
        s = trade_stats(tr)
        if s["n"]:
            print(f"  {label:10} {s['n']:>5} {s['win_rate']:>5.1%} {s['payoff']:>5.2f} "
                  f"{s['expectancy']:>+9.4f} {s['expectancy_t']:>+6.2f} {s['sum_ret']:>+7.2f}")

    print("\n  하위기간 (rule별 기대값 t):")
    for lo, hi, nm in [("2017-01-01", "2022-01-01", "2017-2021"),
                       ("2022-01-01", "2027-01-01", "2022-2026")]:
        row = f"    {nm}:"
        for rule in ("all", "ex_fomo", "dump"):
            s = trade_stats(trs[rule], date_lo=lo, date_hi=hi)
            row += f"  {rule}={s.get('expectancy', float('nan')):+.4f}(t{s.get('expectancy_t', float('nan')):+.1f})"
        print(row)
    print("\n  판독: ex_fomo/dump가 all보다 기대값 높으면 개미필터가 모멘텀에 값 추가 → 여기 돈."
          " 비슷하면 그냥 모멘텀(개미 무가치).")
    return 0


def _run_strategy(prices: pd.DataFrame, flow: pd.DataFrame, cost: float) -> int:
    """학습 통합 전략 실험: 여러 신호를 IC가중 마켓뉴트럴 북으로 백테스트(비용 후)."""
    W = 5
    # 신호 변형들 (모두 -1 계열 = 반대/조건부)
    variants = {
        "raw_contra(-ri)": build_contrarian_signal(flow, window=W, sign=-1),
        "winners_fade": build_control_signals(flow, prices, window=W)["flow_in_winners"],
        "mom_reversal": build_control_signals(flow, prices, window=W)["mom_reversal"],
        "smart(-ri×sign pr)": build_smart_signal(flow, prices, window=W),
        "smart+0.5mom": build_smart_signal(flow, prices, window=W, with_momentum=0.5),
    }
    H = 20  # 보유지평선 = 리밸런스주기(비겹침, step=horizon) — 한 보유기간당 비용 1회만
    print(f"\n=== 전략 실험: IC가중 마켓뉴트럴 북 (비겹침 h{H}/step{H}, 비용 후) ===")
    print(f"  {'signal':22} {'Sharpe':>7} {'t':>6} {'cum':>8} {'MDD':>7} {'turn':>6} {'Sh_gross':>8}")
    for name, panel in variants.items():
        s = ic_weighted_book(prices, panel, horizon=H, step=H, cost_one_way=cost)
        print(f"  {name:22} {s['sharpe']:>+7.2f} {s['t']:>+6.2f} {s['cum']:>+8.2f} "
              f"{s['mdd']:>+7.2f} {s['turnover']:>6.2f} {s['sharpe_gross']:>+8.2f}")

    print("\n=== 하위기간 견고성 (smart, 비겹침, 비용 후) ===")
    smart = variants["smart(-ri×sign pr)"]
    for lo, hi, nm in [("2017-01-01", "2022-01-01", "2017-2021"),
                       ("2022-01-01", "2027-01-01", "2022-2026")]:
        s = ic_weighted_book(prices, smart, horizon=H, step=H, cost_one_way=cost, date_lo=lo, date_hi=hi)
        print(f"  {nm}: Sharpe={s['sharpe']:+.2f}  t={s['t']:+.2f}  cum={s['cum']:+.2f}  MDD={s['mdd']:+.2f}")

    print("\n=== 지평선·유니버스 스윕 (smart, 비겹침 step=horizon, 비용 후 | 괄호=비용전) ===")
    print(f"  {'horizon':>7} {'adv_floor':>9} {'Sharpe':>7} {'t':>6} {'turn':>6} {'(gross)':>8}")
    for h in (5, 20, 40, 60):
        for af in (20000.0, 50000.0):
            s = ic_weighted_book(prices, smart, horizon=h, step=h, adv_floor=af, cost_one_way=cost)
            print(f"  {h:>7} {af:>9.0f} {s['sharpe']:>+7.2f} {s['t']:>+6.2f} {s['turnover']:>6.2f} "
                  f"{s['sharpe_gross']:>+8.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="개미 반대매매 백테스트")
    ap.add_argument("--db", default=None)
    ap.add_argument("--window", type=int, default=SIGNAL_WINDOW)
    ap.add_argument("--cost", type=float, default=0.0023, help="1방향 거래비용(2차 비교용)")
    ap.add_argument("--out", default="docs/contrarian-retail.md")
    ap.add_argument("--behavior", action="store_true",
                    help="심리적 실수(추격·물타기·투매) 반대 신호만 분석")
    ap.add_argument("--strategy", action="store_true",
                    help="학습 통합 전략을 IC가중 북으로 실험(비용 후)")
    ap.add_argument("--trade", action="store_true",
                    help="이벤트드리븐 트레이드(급등주 개미 극단 페이드) 실험")
    ap.add_argument("--money", action="store_true",
                    help="롱온리 모멘텀+개미필터가 순수 모멘텀을 이기는지 검증")
    ap.add_argument("--harden", action="store_true",
                    help="dump 엣지 강체화: 랜덤대조·명세강건성·부트스트랩CI")
    ap.add_argument("--payoff", action="store_true",
                    help="손익비 중심 청산 비교(고정목표 vs 트레일링, 수익 달리기)")
    args = ap.parse_args()

    _load_env_db()
    print("=== 데이터 로드 ===")
    prices, flow = load_data(args.db)
    print(f"  가격 {len(prices):,}행({PRICE_TABLE}), 수급 {len(flow):,}행")

    if args.payoff:
        return _run_payoff(prices, flow, args.cost)
    if args.harden:
        return _run_harden(prices, flow, args.cost)
    if args.money:
        return _run_money(prices, flow, args.cost)
    if args.trade:
        return _run_trade(prices, flow, args.cost)
    if args.strategy:
        return _run_strategy(prices, flow, args.cost)
    if args.behavior:
        return _run_behavior(prices, flow)

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
