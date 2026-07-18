#!/usr/bin/env python
"""Step 2 신호 — 확인된 상승추세 속 **깊은 눌림목 되돌림**(pullback mean-reversion).

가설(플랜 Step 2): 돌파(Minervini)가 레짐 종속으로 기각됐다면 그 **반대 모양** —
강한 추세의 유동성 중대형주가 일시 과매도로 눌렸다가 되돌아오는 스냅백 — 이 더
안정적일 수 있다. 되돌림이면 **승률이 높아야** 한다(가설이 맞다면 dist_shape 승률↑).

Step -1 feasibility 는 **느슨한** 눌림 규칙(close>MA50>MA200 AND (low≤MA20 OR RSI<35))
이 주당 중앙값 23건 ≫ 4~8 슬롯으로 **과공급**임을 보였다. 그래서 사전등록 규칙은
느슨판이 아니라 **더 타이트·고확신** 이어야 한다(공급 ~4~8 슬롯에 맞도록 조임):

  · 추세를 **풀 스테이지-2 템플릿**으로 조임: close>MA50>MA150>MA200 + MA200 상승
    (느슨판은 close>MA50>MA200 뿐 — MA150 추가가 첫 조임 레버).
  · 눌림을 **깊은 과매도**로 조임: RSI(14)[t-1] < 30 (느슨판 RSI<35 OR MA20터치의
    OR 를 버리고, 스테이지-2 추세에서 드문 RSI<30 단일 조건으로 = 진짜 수요선 되돌림).
  · **반전 트리거**로 진입 확인: 오늘 상승마감(close[t] > close[t-1]) = 스냅백 시작
    (떨어지는 칼 잡지 않음). 신호 t종가 → **t+1 시가 진입**.

no-lookahead(플랜 Principle 4): 모든 MA/RSI/ADV 는 t까지 데이터만(RSI 는 shift(1) 로
전일 과매도를 참조), 진입은 t+1 시가. 동일 종목 ≥MIN_GAP 거래일 간격 de-dup 로 독립
스윙 근사. 데이터는 split-adjusted ``daily_bars_adjusted`` 만(raw 는 분할을 파국적
수익으로 읽어 MA/RSI 를 오염).

경계: 이 모듈은 신호·시뮬 로직만 담는다(research/signals). 판정 배터리는 재발명하지
않고 research/experiments/pullback_prop_gate.py 가 prop_gate 를 호출한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- 사전등록 파라미터 기본값 (러너가 명시적으로 넘기지만, 여기 문서화) ----------
ADV_WINDOW = 20
ADV_FLOOR = 20000.0        # trade_value 단위(~20B KRW), Step -1/Step 1 과 동일
MIN_GAP = 10               # 동일 종목 재진입 최소 거래일 간격(독립 스윙 프록시)
RSI_LEN = 14
RSI_MAX = 30.0             # 전일 과매도 임계(깊은 눌림 조임 레버)


def build_panels(prices: pd.DataFrame) -> dict:
    """date×code 패널 + 눌림 신호에 필요한 파생 피처(전부 trailing = no-lookahead)."""
    def piv(col: str) -> pd.DataFrame:
        # Kiwoom signed close → abs (repo 관용).
        return prices.pivot_table(index="date", columns="code", values=col, aggfunc="first").abs()

    close = piv("close").sort_index()
    high = piv("high").reindex_like(close)
    low = piv("low").reindex_like(close)
    open_ = piv("open").reindex_like(close)
    tval = piv("trade_value").reindex_like(close)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    ma200_rising = ma200 > ma200.shift(20)          # 200MA 상승(확인된 추세)
    adv = tval.rolling(ADV_WINDOW).mean()

    # RSI(14) — SMA 기반(진입 신호엔 충분; feasibility 와 동일 정의).
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(RSI_LEN).mean()
    avg_loss = loss.rolling(RSI_LEN).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss > 0, 100.0)            # 전부 상승한 창 → RSI 100

    return {
        "close": close, "high": high, "low": low, "open": open_,
        "ma20": ma20, "ma50": ma50, "ma150": ma150, "ma200": ma200,
        "ma200_rising": ma200_rising, "adv": adv, "rsi": rsi,
        "dates": list(close.index), "codes": list(close.columns),
    }


def build_pullback_signal(
    p: dict,
    *,
    rsi_max: float = RSI_MAX,
    adv_floor: float = ADV_FLOOR,
    full_template: bool = True,
    require_reversal: bool = True,
) -> pd.DataFrame:
    """사전등록 눌림 진입 마스크(date×code bool). 모든 항목 t까지 데이터만.

    Args:
        rsi_max: 전일(RSI[t-1]) 과매도 임계. 사전등록 30.
        adv_floor: 유동성 하한(ADV 20d).
        full_template: True 면 close>MA50>MA150>MA200(스테이지-2), False 면
            느슨판 close>MA50>MA200 (탐색적 민감도 전용).
        require_reversal: True 면 반전 트리거(close[t]>close[t-1]) 요구.

    Returns:
        date×code bool 마스크. True = 신호일 t(진입은 러너가 t+1 시가로 지연).
    """
    c = p["close"]
    if full_template:
        trend = (c > p["ma50"]) & (p["ma50"] > p["ma150"]) & (p["ma150"] > p["ma200"])
    else:
        trend = (c > p["ma50"]) & (p["ma50"] > p["ma200"])
    trend = trend & p["ma200_rising"]

    oversold = p["rsi"].shift(1) < rsi_max          # 전일 깊은 과매도 = 진짜 눌림
    mask = trend & oversold
    if require_reversal:
        mask = mask & (c > c.shift(1))              # 오늘 상승마감 = 스냅백 시작
    liquid = p["adv"] >= adv_floor
    return mask & liquid


def dedup_entries(
    mask: pd.DataFrame, date_pos: dict[str, int], min_gap: int = MIN_GAP,
) -> list[tuple[str, str]]:
    """마스크 True → (신호일 t, code), 동일 종목 ≥min_gap 거래일 간격만 유지(시간순)."""
    codes = np.asarray(mask.columns)
    raw: list[tuple[str, str]] = []
    for d, row in zip(mask.index, mask.to_numpy(), strict=True):
        for c in codes[row]:
            raw.append((str(d), str(c)))
    kept: list[tuple[str, str]] = []
    last: dict[str, int] = {}
    for d, c in sorted(raw):                          # 시간순
        pos = date_pos[d]
        if c in last and pos - last[c] < min_gap:
            continue
        kept.append((d, c))
        last[c] = pos
    return kept


def simulate_pullback_trades(
    entries: list[tuple[str, str]],
    p: dict,
    *,
    stop: float,
    target: float,
    hold_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """각 신호를 t+1 시가 진입 → 하드손절 | 이익목표 | 시간청산 → gross 수익.

    되돌림은 빠르게 스냅백하므로 청산은 타이트 손절 + 이익목표 + 시간상한이다.
    동일봉에서 손절·목표가 함께 닿으면 **보수적으로 손절 우선**(장중 순서 불가지 →
    최악 가정). 갭이면 시가로 체결. 비용은 prop_gate 안에서 사후 차감(gross 1회).

    Returns:
        (fill_dates=진입일 t+1, gross_returns) — 정렬된 numpy 배열.
    """
    dates = p["dates"]
    date_pos = {str(d): i for i, d in enumerate(dates)}
    code_pos = {str(c): j for j, c in enumerate(p["codes"])}
    Op = p["open"].to_numpy(float)
    Hi = p["high"].to_numpy(float)
    Lo = p["low"].to_numpy(float)
    Cl = p["close"].to_numpy(float)
    n = len(dates)

    fill_dates: list[str] = []
    rets: list[float] = []
    for sig_date, code in entries:
        ti = date_pos.get(sig_date)
        cj = code_pos.get(code)
        if ti is None or cj is None:
            continue
        e = ti + 1                                    # t+1 진입 바
        if e >= n:
            continue
        entry_px = Op[e, cj]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        stop_px = entry_px * (1.0 - stop)
        target_px = entry_px * (1.0 + target)
        end = min(e + hold_max, n - 1)                # 시간청산 상한 바
        exit_px = None
        for k in range(e, end + 1):
            lo_k, hi_k, op_k = Lo[k, cj], Hi[k, cj], Op[k, cj]
            # 1) 하드손절 우선(보수적): 저가가 손절 관통 → 갭다운이면 시가, 아니면 손절가.
            if np.isfinite(lo_k) and lo_k <= stop_px:
                exit_px = op_k if (np.isfinite(op_k) and op_k < stop_px) else stop_px
                break
            # 2) 이익목표: 고가가 목표 관통 → 갭업이면 시가, 아니면 목표가.
            if np.isfinite(hi_k) and hi_k >= target_px:
                exit_px = op_k if (np.isfinite(op_k) and op_k > target_px) else target_px
                break
        if exit_px is None:                           # 미청산 → 시간청산 바 시가
            exit_px = Op[end, cj]
            if not np.isfinite(exit_px):              # 마지막 유한 종가 폴백
                col = Cl[e:end + 1, cj]
                fin = col[np.isfinite(col)]
                if not len(fin):
                    continue
                exit_px = fin[-1]
        fill_dates.append(str(dates[e]))              # 진입일 = t+1(자본 투입 시점)
        rets.append(float(exit_px / entry_px - 1.0))
    return np.asarray(fill_dates), np.asarray(rets, float)
