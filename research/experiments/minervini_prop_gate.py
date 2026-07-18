#!/usr/bin/env python
"""Step 1 — 미너비니 돌파 셋업을 **개별-트레이드 게이트**에 태우는 얇은 러너.

플랜(.omc/plans/prop-swing-alpha.md) Step 1 + CONSENSUS REVISIONS(R1~R4) 를 따른다.

R1(사전등록·비협상): 게이트 출력을 보기 **전에** TRAIN-only/정론 추론으로 딱 하나의
1차 config 를 골라 못박는다. 그 하나의 config 가 THE 테스트다. 파라미터 그리드는
**탐색적 민감도(plateau 확인)** 로만 — 2차·라벨링, 승자픽 금지. 그리드 argmax 로만
사는 config 는 합격 아님.

R4(데이터 위험): research/signals/operator_flow/minervini/scanner_final.py 는 LIVE
시점 스캐너(비조정 데이터 접촉 가능성) — 백테스트 추출기로 재사용 금지. 이 파일은
split-adjusted ``daily_bars_adjusted`` 위에 새로 만든 **역사적** 추출기다. 트렌드
템플릿/RS/돌파/거래량수축 규칙은 kr_quant.features(vcp·rs_rating) 정론과 동일 사상으로
패널 벡터화; no-lookahead(모든 MA/고저/RSI/거래량은 t까지, 신호 t종가 → t+1 시가 진입,
동일 종목 ≥10거래일 간격 de-dup).

판정은 재발명 금지 — research/experiments/prop_gate.py 의 prop_gate/random_entry_control
을 그대로 호출한다. 결과는 research/logs/minervini_prop/VERDICT.md 로(사람이 읽는 리포트,
코딩된 판정 없음 — R3).

실행(DB 필요): docker start kr-quant-airflow-timescaledb-1 후
    uv run python research/experiments/minervini_prop_gate.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# 형제 모듈(같은 research/experiments) — 스크립트 실행시 그 디렉터리가 sys.path[0].
from prop_gate import prop_gate, random_entry_control

from kr_quant.storage import connect, db_default

# ===========================================================================
# 사전등록 1차 CONFIG (R1) — 게이트 출력을 보기 전에 못박음. 이것이 THE 테스트.
# ---------------------------------------------------------------------------
# 근거(정론 + 기존 HONEST_VERDICT):
#  · 진입 = 미너비니 트렌드 템플릿(스테이지2) + IBD RS≥70(8번째 게이트) + 50일고
#    돌파 + 돌파 직전 거래량 수축(dry-up). vcp.py 가 "거래량이 진짜 게이트"라 결론냈으므로
#    가격기하(연속 수축 모양)가 아니라 거래량 dry-up 을 수축 프록시로 사전등록.
#  · 손절 = 7% 하드손절(미너비니 정론 7~8%; HONEST 는 5% 였으나 7% 가 캔들 노이즈 대비
#    중앙값적 선택). R 정규화 분모 = 0.07.
#  · 청산 = 하드손절 + 시간청산(10거래일). 스윙 1~10일 범위의 상한, 파라미터 최소 →
#    재현성 최대. 트레일링은 2차 민감도로 강등(R1: 1차는 딱 하나).
#  · 유니버스 = ADV(20d) ≥ 20B KRW(trade_value floor 20000) — Step -1 feasibility 와 동일.
# ===========================================================================
PRICE_TABLE = "daily_bars_adjusted"     # split-adjusted (raw daily_bars 금지 — R4)
ADV_WINDOW = 20
ADV_FLOOR = 20000.0                      # trade_value 단위(~20B KRW), Step -1 과 동일
MIN_GAP = 10                             # 동일 종목 재진입 최소 간격(독립 스윙 프록시)

PRE_STOP = 0.07                          # 1차 하드손절폭 (= R 정규화 분모)
PRE_HOLD_MAX = 10                        # 1차 시간청산(거래일)
PRE_EXIT = "time"                        # 1차 청산 = 하드손절 + 시간청산 (트레일링 아님)

RS_MIN = 70.0                            # IBD RS rating 게이트(트렌드템플릿 8번째)
NEAR_HIGH = 0.90                         # close ≥ 0.90 × 252일고
ABOVE_LOW = 1.25                         # close ≥ 1.25 × 252일저
VOL_DRYUP = 0.70                         # 돌파 직전 5일 평균거래량 < 0.70 × 50일 평균(t-1까지)

TRAIN_HI = "2022-01-01"
OUT_DIR = "research/logs/minervini_prop"

# IBD 프론트웨이티드 RS 블렌드(rs_rating.py 와 동일 가중/룩백).
RS_WEIGHTS = (2.0, 1.0, 1.0, 1.0)
RS_LOOKBACKS = (63, 126, 189, 252)


def _load_env_db() -> None:
    """.env 의 KR_QUANT_DB 를 환경에 실어줌(다른 러너와 동일 관용)."""
    if os.environ.get("KR_QUANT_DB") or not os.path.exists(".env"):
        return
    for line in open(".env"):
        if line.startswith("KR_QUANT_DB"):
            os.environ["KR_QUANT_DB"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            break


def load_prices() -> pd.DataFrame:
    """split-adjusted OHLC + volume + trade_value 를 long 으로 로드."""
    _load_env_db()
    con = connect(db_default())
    prices = pd.read_sql_query(
        f"SELECT code, date, open, high, low, close, volume, trade_value FROM {PRICE_TABLE}",  # noqa: S608 — 신뢰 상수
        con,
    )
    con.close()
    prices["code"] = prices["code"].astype(str)
    prices["date"] = prices["date"].astype(str)
    return prices


# ---------------------------------------------------------------------------
# 패널(rows=date, cols=code) — 롤링이 시간축을 따라 trailing 으로 도므로 no-lookahead.
# ---------------------------------------------------------------------------
def build_panels(prices: pd.DataFrame) -> dict:
    """date×code 패널 + 사전등록 진입규칙에 필요한 파생 피처."""
    def piv(col: str) -> pd.DataFrame:
        # Kiwoom signed close → abs (panel_pivot 관용과 동일).
        return prices.pivot_table(index="date", columns="code", values=col, aggfunc="first").abs()

    close = piv("close").sort_index()
    high = piv("high").reindex_like(close)
    low = piv("low").reindex_like(close)
    open_ = piv("open").reindex_like(close)
    vol = piv("volume").reindex_like(close)
    tval = piv("trade_value").reindex_like(close)

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    ma200_rising = ma200 > ma200.shift(20)          # 200MA 상승(정론 트렌드템플릿)
    hi252 = high.rolling(252).max()                 # 252일 최고(오늘 포함)
    low252 = low.rolling(252).min()                 # 252일 최저(오늘 포함)
    prior_hi50 = high.rolling(50).max().shift(1)     # [t-50, t-1] 최고(돌파 기준, t 제외)
    adv = tval.rolling(ADV_WINDOW).mean()

    # 돌파 직전 거래량 dry-up: 5일평균(t-1까지) < 0.70 × 50일평균(t-1까지).
    # shift(1) 로 돌파일 t 의 거래량 팽창을 제외 → "베이스에서 말라붙었나"를 잰다(vcp.py 사상).
    vol5 = vol.rolling(5).mean().shift(1)
    vol50 = vol.rolling(50).mean().shift(1)
    dryup = vol5 < (VOL_DRYUP * vol50)

    # IBD RS rating: 프론트웨이티드 트레일링 수익 블렌드의 그날 횡단면 백분위 ×100.
    blend = None
    for w, lb in zip(RS_WEIGHTS, RS_LOOKBACKS):
        r = close / close.shift(lb) - 1.0
        blend = (w * r) if blend is None else (blend + w * r)
    rs = blend.rank(axis=1, pct=True) * 100.0        # 그날 유한값 횡단면 순위

    return {
        "close": close, "high": high, "low": low, "open": open_,
        "ma50": ma50, "ma150": ma150, "ma200": ma200, "ma200_rising": ma200_rising,
        "hi252": hi252, "low252": low252, "prior_hi50": prior_hi50,
        "adv": adv, "dryup": dryup, "rs": rs,
        "dates": list(close.index), "codes": list(close.columns),
    }


def entry_mask(p: dict) -> pd.DataFrame:
    """사전등록 진입 마스크(date×code bool). 모든 항목 t까지 데이터만 사용."""
    c = p["close"]
    trend = (
        (c > p["ma50"]) & (p["ma50"] > p["ma150"]) & (p["ma150"] > p["ma200"])
        & p["ma200_rising"]
        & (c >= NEAR_HIGH * p["hi252"])
        & (c >= ABOVE_LOW * p["low252"])
    )
    rs_ok = p["rs"] >= RS_MIN
    breakout = c > p["prior_hi50"]                    # 50일 신고 돌파(오늘 종가)
    liquid = p["adv"] >= ADV_FLOOR
    return trend & rs_ok & breakout & p["dryup"] & liquid


def dedup_entries(mask: pd.DataFrame, date_pos: dict[str, int]) -> list[tuple[str, str]]:
    """마스크 True → (신호일 t, code) 리스트, 동일 종목 ≥MIN_GAP 거래일 간격만 유지."""
    codes = np.asarray(mask.columns)
    raw: list[tuple[str, str]] = []
    for d, row in zip(mask.index, mask.to_numpy(), strict=True):
        for c in codes[row]:
            raw.append((str(d), str(c)))
    kept: list[tuple[str, str]] = []
    last: dict[str, int] = {}
    for d, c in sorted(raw):                          # 시간순
        pos = date_pos[d]
        if c in last and pos - last[c] < MIN_GAP:
            continue
        kept.append((d, c))
        last[c] = pos
    return kept


# ---------------------------------------------------------------------------
# 개별-트레이드 시뮬레이션 — 신호 t종가 → t+1 시가 진입, 하드손절 + (시간청산 | 트레일링).
# 반환: (fill_dates=진입일 t+1, gross_returns). 비용은 prop_gate 안에서 사후 차감.
# ---------------------------------------------------------------------------
def simulate_trades(
    entries: list[tuple[str, str]],
    p: dict,
    *,
    stop: float,
    hold_max: int,
    exit_mode: str = "time",
    trail_pct: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """entries 각각을 t+1 시가 진입 → 하드손절/시간청산 시뮬 → gross 수익."""
    dates = p["dates"]
    date_pos = {str(d): i for i, d in enumerate(dates)}
    code_pos = {str(c): j for j, c in enumerate(p["codes"])}
    Op = p["open"].to_numpy(float)
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
        end = min(e + hold_max, n - 1)                # 시간청산 상한 바
        peak = entry_px                               # 트레일링용 진입후 최고 종가
        exit_px = None
        for k in range(e, end + 1):
            lo_k, op_k, cl_k = Lo[k, cj], Op[k, cj], Cl[k, cj]
            # 1) 하드손절: 당일 저가가 손절 관통 → 손절가(갭다운이면 시가)로 청산.
            if np.isfinite(lo_k) and lo_k <= stop_px:
                exit_px = op_k if (np.isfinite(op_k) and op_k < stop_px) else stop_px
                break
            # 2) 트레일링(2차 모드): 진입후 최고 종가 대비 trail_pct 하락 → 종가 청산.
            if exit_mode == "trail" and np.isfinite(cl_k):
                peak = max(peak, cl_k)
                if cl_k <= peak * (1.0 - trail_pct):
                    exit_px = cl_k
                    break
        if exit_px is None:                           # 미청산 → 시간청산 바 시가
            exit_px = Op[end, cj]
            if not np.isfinite(exit_px):              # 마지막 유한 종가로 폴백
                col = Cl[e:end + 1, cj]
                fin = col[np.isfinite(col)]
                if not len(fin):
                    continue
                exit_px = fin[-1]
        fill_dates.append(str(dates[e]))              # 진입일 = t+1(자본 투입 시점)
        rets.append(float(exit_px / entry_px - 1.0))
    return np.asarray(fill_dates), np.asarray(rets, float)


# ---------------------------------------------------------------------------
# 리포트 조립
# ---------------------------------------------------------------------------
def _regime_split(fill_dates: np.ndarray, rets: np.ndarray, cost: float, stop: float) -> dict:
    """레짐 정직성(플랜 Step 1): 2018-2022 vs 2023-2026 건당 기대값(gross·R)."""
    out = {}
    for name, lo, hi in [("2018-2022", "2018-01-01", "2023-01-01"),
                         ("2023-2026", "2023-01-01", "2027-01-01")]:
        m = (fill_dates >= lo) & (fill_dates < hi)
        r = rets[m]
        r = r[np.isfinite(r)]
        R = (r - cost) / stop
        out[name] = {
            "n": int(len(r)),
            "gross_mean": float(r.mean()) if len(r) else float("nan"),
            "expectancy_R": float(R.mean()) if len(R) else float("nan"),
            "win_rate": float((R > 0).mean()) if len(R) else float("nan"),
        }
    return out


def _sensitivity(entries: list[tuple[str, str]], p: dict) -> list[dict]:
    """탐색적 민감도(2차, R1) — stop×hold_max 격자 + 트레일링 변형. plateau 확인용.

    승자픽 아님: 사전등록 config 가 THE 테스트. 여기선 OOS expR·clean-fold 만 표로.
    """
    rows = []
    grid = [(s, h, "time") for s in (0.05, 0.07, 0.10) for h in (5, 10, 15)]
    grid += [(0.07, 10, "trail")]                    # 트레일링 변형 1개(비교용)
    for s, h, mode in grid:
        fd, rr = simulate_trades(entries, p, stop=s, hold_max=h, exit_mode=mode)
        rep = prop_gate(fd, rr, s, label=f"sens_{s}_{h}_{mode}", verbose=False)
        cs0 = rep["cost_sweep"][0]
        fb = rep["folds"]
        rows.append({
            "stop": s, "hold": h, "mode": mode, "n": rep["n_total"],
            "oos_expR": cs0["oos_expectancy_R"],
            "clean_k": fb["clean_oos_positive"], "clean_v": fb["clean_oos_valid"],
            "raw_k": fb["raw_positive"], "raw_v": fb["raw_valid"],
        })
    return rows


def _fmt_verdict(
    rep: dict, ctrl: dict, regime: dict, sens: list[dict], n_entries_raw: int,
) -> str:
    """VERDICT.md — 사전등록 블록 먼저, 그 아래 게이트 숫자(코딩된 판정 없음, R3)."""
    L: list[str] = []
    L.append("# Minervini 돌파 — 개별-트레이드 게이트 VERDICT (Step 1)")
    L.append("")
    L.append("> **리포터 문서다 (R3).** 아래 숫자를 사람이 읽고 kill/continue 를 판단한다. "
             "코딩된 PASS/FAIL 임계값 없음.")
    L.append("")
    # --- 사전등록(결과를 보기 전에 못박은 것) ---
    L.append("## 사전등록 1차 CONFIG (R1 — 결과를 보기 전에 확정)")
    L.append("")
    L.append("이 **딱 하나의 config 가 THE 테스트다.** 그리드 argmax 로만 사는 값은 합격 아님.")
    L.append("")
    L.append("**진입(정론 트렌드 템플릿 + RS + 돌파 + 거래량수축, 전부 t까지 데이터):**")
    L.append("- close > MA50 > MA150 > MA200, 그리고 MA200 상승(200MA[t] > 200MA[t-20])")
    L.append(f"- close ≥ {NEAR_HIGH:.2f}×(252일고)  AND  close ≥ {ABOVE_LOW:.2f}×(252일저)")
    L.append(f"- IBD RS rating ≥ {RS_MIN:.0f} (프론트웨이티드 3/6/9/12M 블렌드 그날 횡단면 백분위)")
    L.append("- 돌파: close > 직전 50일 신고가(max high over [t-50, t-1])")
    L.append(f"- 돌파 직전 거래량 dry-up: mean vol[t-5..t-1] < {VOL_DRYUP:.2f} × mean vol[t-50..t-1]")
    L.append(f"- 유니버스: ADV({ADV_WINDOW}d) ≥ {ADV_FLOOR:.0f} trade_value(~20B KRW)")
    L.append("")
    L.append("**타이밍/독립성:** 신호 t종가 → **t+1 시가 진입**; 동일 종목 ≥"
             f"{MIN_GAP}거래일 간격 de-dup.")
    L.append(f"**청산(1차):** 하드손절 {PRE_STOP:.0%} + 시간청산 {PRE_HOLD_MAX}거래일 "
             "(트레일링 아님 — 트레일링은 2차 민감도로 강등). R 정규화 분모 = 손절폭.")
    L.append("")
    L.append(f"원신호(de-dup 전) {n_entries_raw} → de-dup 후 트레이드 {rep['n_total']}건, "
             f"진입일 {rep['entry_range'][0]}~{rep['entry_range'][1]}.")
    L.append("")
    L.append("---")
    L.append("")

    # --- 게이트 결과 ---
    L.append("## 게이트 결과 (prop_gate, 사전등록 config)")
    L.append("")
    L.append(f"- **총 트레이드 n = {rep['n_total']}**, 손절폭 stop = {rep['stop']:.0%}, "
             f"기준비용 {int(round(rep['primary_cost'] * 1e4))}bp")
    edies = rep["cost_edge_dies"]
    L.append(f"- **비용 스윕 — 엣지 사망 비용:** "
             f"{'전 구간 생존' if edies is None else f'{int(round(edies * 1e4))}bp 에서 OOS expR ≤ 0'}")
    L.append("")
    L.append("### [1] 슬리피지 스윕 (비용별 OOS 진입≥2022 기대값 R + 폴드 일관성)")
    L.append("")
    L.append("| cost | oos_n | oos_expR | raw k/6 | clean-OOS k/N |")
    L.append("|---|---|---|---|---|")
    for cs in rep["cost_sweep"]:
        bp = int(round(cs["cost"] * 1e4))
        L.append(f"| {bp}bp | {cs['oos_n']} | {cs['oos_expectancy_R']:+.3f} | "
                 f"{cs['raw_fold_positive']}/{cs['raw_fold_valid']} | "
                 f"{cs['clean_fold_positive']}/{cs['clean_fold_valid']} |")
    L.append("")
    fb = rep["folds"]
    L.append("### [2] 폴드 재현성 (R2 — raw k/6 과 clean-OOS k/N 둘 다)")
    L.append("")
    L.append("| TEST 창 | n | expR | 양수? | 구분 |")
    L.append("|---|---|---|---|---|")
    for row in fb["rows"]:
        tag = "INSIDE-TRAIN" if row["inside_train"] else "clean OOS"
        L.append(f"| {row['test_window']} | {row['n']} | {row['expectancy_R']:+.3f} | "
                 f"{'●' if row['positive'] else '○'} | {tag} |")
    L.append("")
    L.append(f"- **raw {fb['raw_positive']}/{fb['raw_valid']}  |  "
             f"clean-OOS {fb['clean_oos_positive']}/{fb['clean_oos_valid']}** "
             f"(inside-train {fb['inside_train_windows']} 제외 — 6클린폴드 주장 금지, R2)")
    L.append("")
    d = rep["distribution"]
    fr = rep["fragility"]
    tr = fr["tail_removal"]
    L.append("### [3] 분포 모양 + 취약성 (기준비용, OOS 트레이드)")
    L.append("")
    L.append(f"- n={d['n']}  기대값R={d['expectancy_R']:+.3f}  승률={d['win_rate']:.0%}  "
             f"손익비={d['payoff']:.2f}  왜도={d['skew']:+.2f}  왼꼬리비중={d['left_tail_share']:.0%}")
    L.append(f"- monster top5={fr['monster_share_top5']:.0%}  top20={fr['monster_share_top20']:.0%}  "
             f"최장연패={fr['max_loss_streak']}  중앙값R={fr['median_trade']:+.3f}")
    L.append(f"- 꼬리제거(상위20): 기대값 {tr['expectancy_full']:+.3f} → {tr['expectancy_ex']:+.3f} "
             f"({'생존' if tr['expectancy_ex'] > 0 else '붕괴'})")
    L.append("")
    u = rep["untouched"]
    L.append(f"### [4] 미접촉 최종창 (R1 held-out) [{u['lo']}~{u['hi']}) — 폴드와 별개")
    L.append("")
    L.append(f"- n={u['n']}  기대값R={u['expectancy_R']:+.3f}  "
             f"승률={u['dist']['win_rate']:.0%}  왜도={u['dist']['skew']:+.2f}")
    L.append("")
    gr = rep["gate_report"]
    ci = gr["expectancy_ci"]
    fc = gr["fold_consistency"]
    L.append("### [5] gate_report 배포신호 (REPORTER)")
    L.append("")
    L.append(f"- OOS n={gr['n']}  기대값R={gr['expectancy_R']:+.3f}  "
             f"95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    L.append(f"- 폴드 {fc['n_positive']}/{fc['n_folds']} 양수  monster={gr['monster_share']:.0%}  "
             f"최장연패={gr['max_loss_streak']}")
    L.append("")

    # --- 음성대조 ---
    L.append("## vs 랜덤 음성대조 (R1 — 게이트 자체 위양성률 보정)")
    L.append("")
    L.append("같은 유니버스/타이밍의 marginal 을 흉내낸 무작위 진입↔수익 페어링을 "
             "동일 게이트에 태운 것. 무작위가 이 바를 실제 셋업만큼 자주 넘으면 바가 느슨.")
    L.append("")
    L.append(f"- 무작위 raw ≥5/6 폴드 도달: **{ctrl['raw_ge5_frac']:.1%}** of draws")
    L.append(f"- 무작위 clean-OOS 전폴드(≥{ctrl['clean_fold_valid_median']}/"
             f"{ctrl['clean_fold_valid_median']}) 양수: **{ctrl['clean_all_positive_frac']:.1%}** of draws")
    L.append(f"- 무작위 OOS 기대값R: 평균 {ctrl['oos_expectancy_R_mean']:+.3f}  "
             f"p95 {ctrl['oos_expectancy_R_p95']:+.3f}")
    L.append(f"- **실제 Minervini clean-OOS = {fb['clean_oos_positive']}/{fb['clean_oos_valid']}, "
             f"OOS expR = {gr['expectancy_R']:+.3f}** — 위 무작위 분포 대비 어디에 서 있나(사람 판단).")
    L.append("")

    # --- 레짐 ---
    L.append("## 레짐 정직성 (플랜 Step 1 — 건당 기대값 레짐별)")
    L.append("")
    L.append("HONEST_VERDICT 는 포트 레벨 +1.0%(2018-22) vs +47.8%(2023-26) 였다 — "
             "**건당** 엣지도 레짐 포로인가?")
    L.append("")
    L.append("| 레짐 | n | gross 평균 | 기대값R | 승률 |")
    L.append("|---|---|---|---|---|")
    for name, r in regime.items():
        L.append(f"| {name} | {r['n']} | {r['gross_mean']:+.3%} | {r['expectancy_R']:+.3f} | "
                 f"{r['win_rate']:.0%} |")
    L.append("")

    # --- 민감도(2차) ---
    L.append("## 탐색적 민감도 (2차, R1 — plateau vs spike, 승자픽 아님)")
    L.append("")
    L.append("사전등록 config(stop 0.07 / hold 10 / time)가 격자에서 **고원(plateau)** 위에 "
             "있는지, 아니면 홀로 튀는 spike 인지 확인용. spike 면 신뢰도 낮음.")
    L.append("")
    L.append("| stop | hold | mode | n | OOS expR | clean-OOS k/N | raw k/6 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in sens:
        star = "  ← 사전등록" if (s["stop"] == PRE_STOP and s["hold"] == PRE_HOLD_MAX
                                and s["mode"] == "time") else ""
        L.append(f"| {s['stop']:.2f} | {s['hold']} | {s['mode']} | {s['n']} | "
                 f"{s['oos_expR']:+.3f} | {s['clean_k']}/{s['clean_v']} | "
                 f"{s['raw_k']}/{s['raw_v']}{star} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 사람이 읽는 kill/continue 메모 (코딩된 임계값 아님, R3)")
    L.append("")
    L.append("_아래는 게이트 러너가 남기는 관찰 프롬프트다. 최종 판단은 team-lead/사람이 한다._")
    L.append("- clean-OOS 폴드 카운트가 무작위 음성대조의 도달 빈도를 유의미하게 넘는가?")
    L.append("- 기대값이 monster top5/top20 소수 트레이드에 얹혀 있나(꼬리제거시 붕괴)?")
    L.append("- 건당 엣지가 2023-2026 레짐에만 사는가(2018-2022 은 ~0/음수)?")
    L.append("- 미접촉창 [2025-07~2026-07) 이 폴드 결과와 같은 방향인가?")
    L.append("- 사전등록 config 가 민감도 격자에서 고원 위인가, 홀로 튀는 spike 인가?")
    L.append("")
    return "\n".join(L)


def run() -> None:
    prices = load_prices()
    p = build_panels(prices)
    date_pos = {str(d): i for i, d in enumerate(p["dates"])}

    mask = entry_mask(p)
    n_entries_raw = int(mask.to_numpy().sum())
    entries = dedup_entries(mask, date_pos)

    # 사전등록 1차 config 시뮬 → prop_gate.
    fill_dates, rets = simulate_trades(
        entries, p, stop=PRE_STOP, hold_max=PRE_HOLD_MAX, exit_mode=PRE_EXIT)
    print(f"\n[extractor] 원신호(de-dup 전) {n_entries_raw} → de-dup {len(entries)} → "
          f"시뮬 트레이드 {len(rets)}")

    rep = prop_gate(fill_dates, rets, PRE_STOP, label="minervini")

    # 음성대조 — 실제 트레이드 수만큼 무작위 draw.
    ctrl = random_entry_control(
        fill_dates, rets, PRE_STOP, n_per_draw=len(rets), n_draws=200, seed=7)

    # 레짐 정직성.
    regime = _regime_split(fill_dates, rets, rep["primary_cost"], PRE_STOP)

    # 탐색적 민감도(2차).
    sens = _sensitivity(entries, p)

    report = _fmt_verdict(rep, ctrl, regime, sens, n_entries_raw)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "VERDICT.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    run()
