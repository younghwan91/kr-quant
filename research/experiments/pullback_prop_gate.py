#!/usr/bin/env python
"""Step 2 — 눌림목 되돌림 셋업을 **개별-트레이드 게이트**에 태우는 얇은 러너.

플랜(.omc/plans/prop-swing-alpha.md) Step 2 + CONSENSUS REVISIONS(R1~R3) 를 따른다.

R1(사전등록·비협상): 게이트 출력을 보기 **전에** TRAIN-only/정론 추론으로 딱 하나의
1차 config 를 골라 못박는다. 그 하나가 THE 테스트다. 파라미터 그리드는 **탐색적
민감도(plateau 확인)** 로만 — 승자픽 금지.

신호·시뮬 로직은 research/signals/pullback_swing.py 에 있다(이 파일은 얇은 러너).
판정 배터리는 재발명 금지 — research/experiments/prop_gate.py 의 prop_gate/
random_entry_control 을 그대로 호출한다. 결과는 research/logs/pullback_prop/VERDICT.md
로(사람이 읽는 리포트, 코딩된 판정 없음 — R3).

실행(DB 필요): docker start quant-airflow-timescaledb-1 후
    uv run python research/experiments/pullback_prop_gate.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# 스크립트로 실행하면 sys.path[0]=이 파일 디렉터리(research/experiments)뿐이라
# 형제 `prop_gate` 는 되지만 `research.signals` 패키지 import 는 실패한다. repo 루트를
# 경로에 실어 둘 다 되게 한다(문서화된 실행: uv run python research/experiments/...).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 형제 모듈(같은 research/experiments) — 스크립트 실행시 그 디렉터리가 sys.path[0].
from prop_gate import prop_gate, random_entry_control  # noqa: E402 — sys.path 부트스트랩 뒤
from prop_swing_common import (  # noqa: E402 — sys.path 부트스트랩 뒤
    gate_sim,
    load_env_db,
    regime_split,
    render_control,
    render_dist_fragility,
    render_gate_core,
    render_gate_report,
    render_untouched,
    weekly_count,
)

# 신호 모듈(research/signals) — 위에서 repo 루트를 경로에 실어 패키지 경로로 import.
from research.signals.pullback_swing import (  # noqa: E402 — sys.path 부트스트랩 뒤
    build_panels,
    build_pullback_signal,
    dedup_entries,
    simulate_pullback_trades,
)

from kr_quant.storage import connect, db_default, read_prices  # noqa: E402 — sys.path 부트스트랩 뒤

# ===========================================================================
# 사전등록 1차 CONFIG (R1) — 게이트 출력을 보기 전에 못박음. 이것이 THE 테스트.
# ---------------------------------------------------------------------------
# 근거(정론 + Step -1 feasibility 의 과공급 관찰):
#  · Step -1 은 느슨한 눌림 규칙이 주당 중앙값 23건 ≫ 4~8 슬롯으로 과공급임을 보였다.
#    → 사전등록은 느슨판이 아니라 3개 레버로 조인 고확신 규칙:
#      (a) 추세: 풀 스테이지-2 close>MA50>MA150>MA200 + MA200 상승 (느슨판은 MA150 없음).
#      (b) 눌림: RSI(14)[t-1] < 30 (느슨판 RSI<35 OR MA20터치의 OR 를 버림) —
#          스테이지-2 추세에서 드문 깊은 과매도 = 진짜 수요선 되돌림.
#      (c) 반전: 오늘 상승마감 close[t]>close[t-1] = 스냅백 시작(떨어지는 칼 회피).
#  · 청산 = 되돌림은 빠르게 스냅백하므로 타이트: 4% 손절 + 5% 목표 + 8거래일 시간상한.
#    R 정규화 분모 = 손절폭 0.04. 되돌림 가설이 맞으면 승률↑(손익비 낮아도 무방).
#  · 유니버스 = ADV(20d) ≥ 20B KRW(trade_value floor 20000) — Step -1/Step 1 과 동일.
# ===========================================================================
PRICE_TABLE = "daily_bars_adjusted"     # split-adjusted (raw daily_bars 금지)
ADV_FLOOR = 20000.0                      # trade_value 단위(~20B KRW)
MIN_GAP = 10                             # 동일 종목 재진입 최소 간격(독립 스윙 프록시)

RSI_MAX = 30.0                           # 전일 과매도 임계(깊은 눌림)
PRE_STOP = 0.04                          # 1차 하드손절폭 (= R 정규화 분모)
PRE_TARGET = 0.05                        # 1차 이익목표
PRE_HOLD_MAX = 8                         # 1차 시간청산(거래일)

TRAIN_HI = "2022-01-01"
OUT_DIR = "research/logs/pullback_prop"


def load_prices() -> pd.DataFrame:
    """split-adjusted OHLC + trade_value 를 long 으로 로드."""
    load_env_db()
    con = connect(db_default())
    prices = read_prices(con, cols=("code", "date", "open", "high", "low", "close", "trade_value"))
    con.close()
    prices["code"] = prices["code"].astype(str)
    prices["date"] = prices["date"].astype(str)
    return prices


def _sensitivity(entries: list[tuple[str, str]], p: dict) -> list[dict]:
    """탐색적 민감도(2차, R1) — stop×target×hold 격자. plateau 확인용(승자픽 아님).

    사전등록 config(stop0.04/target0.05/hold8)가 격자에서 고원 위인지 홀로 튀는
    spike 인지만 표로. 진입 신호는 고정(사전등록) — 청산만 스윕한다.
    """
    rows = []
    grid = [(s, tgt, h)
            for s in (0.03, 0.04, 0.05)
            for tgt in (0.05, 0.08)
            for h in (5, 8, 10)]
    for s, tgt, h in grid:
        fd, rr = simulate_pullback_trades(entries, p, stop=s, target=tgt, hold_max=h)
        rep, cs0, fb, d = gate_sim(fd, rr, s, f"sens_{s}_{tgt}_{h}")
        rows.append({
            "stop": s, "target": tgt, "hold": h, "n": rep["n_total"],
            "oos_expR": cs0["oos_expectancy_R"], "win_rate": d["win_rate"],
            "clean_k": fb["clean_oos_positive"], "clean_v": fb["clean_oos_valid"],
            "raw_k": fb["raw_positive"], "raw_v": fb["raw_valid"],
        })
    return rows


def _fmt_verdict(
    rep: dict, ctrl: dict, regime: dict, sens: list[dict],
    n_entries_raw: int, weekly_median: float, weekly_max: int,
) -> str:
    """VERDICT.md — 사전등록 블록 먼저, 그 아래 게이트 숫자(코딩된 판정 없음, R3)."""
    L: list[str] = []
    L.append("# 눌림목 되돌림 — 개별-트레이드 게이트 VERDICT (Step 2)")
    L.append("")
    L.append("> **리포터 문서다 (R3).** 아래 숫자를 사람이 읽고 kill/continue 를 판단한다. "
             "코딩된 PASS/FAIL 임계값 없음.")
    L.append("")
    # --- 사전등록 ---
    L.append("## 사전등록 1차 CONFIG (R1 — 결과를 보기 전에 확정)")
    L.append("")
    L.append("이 **딱 하나의 config 가 THE 테스트다.** 그리드 argmax 로만 사는 값은 합격 아님. "
             "Step -1 은 느슨한 눌림 규칙이 주당 중앙값 23건(≫4~8슬롯)으로 과공급임을 보였다 "
             "→ 사전등록은 3개 레버로 조인 고확신 규칙:")
    L.append("")
    L.append("**진입(확인된 상승추세 + 깊은 눌림 + 반전, 전부 t까지 데이터):**")
    L.append("- 추세(풀 스테이지-2, 느슨판보다 조임): close>MA50>MA150>MA200 AND MA200 상승"
             "(MA200[t]>MA200[t-20])")
    L.append(f"- 눌림(깊은 과매도): RSI(14)[t-1] < {RSI_MAX:.0f} "
             "(느슨판의 RSI<35 OR MA20터치를 버리고 단일·깊은 조건으로 조임)")
    L.append("- 반전 트리거: 오늘 상승마감 close[t] > close[t-1] (스냅백 시작 — 떨어지는 칼 회피)")
    L.append(f"- 유니버스: ADV(20d) ≥ {ADV_FLOOR:.0f} trade_value(~20B KRW)")
    L.append("")
    L.append(f"**타이밍/독립성:** 신호 t종가 → **t+1 시가 진입**; 동일 종목 ≥{MIN_GAP}거래일 간격 de-dup.")
    L.append(f"**청산(1차):** 하드손절 {PRE_STOP:.0%} + 이익목표 {PRE_TARGET:.0%} + "
             f"시간청산 {PRE_HOLD_MAX}거래일. R 정규화 분모 = 손절폭 {PRE_STOP:.0%}. "
             "동일봉 손절·목표 동시 → 보수적 손절 우선.")
    L.append("")
    L.append(f"원신호(de-dup 전) {n_entries_raw} → de-dup 후 트레이드 {rep['n_total']}건, "
             f"진입일 {rep['entry_range'][0]}~{rep['entry_range'][1]}.")
    L.append(f"공급 체크: 주당 진입 중앙값 {weekly_median:.0f} / 최대 {weekly_max} "
             "(느슨판 23 대비 — 4~8 슬롯 용량에 맞게 조여졌나).")
    L.append("")
    L.append("---")
    L.append("")

    # --- 게이트 결과 (공통 렌더) ---
    L.extend(render_gate_core(rep))
    L.extend(render_dist_fragility(
        rep, win_bold=True,
        note="_되돌림 가설이 맞다면 **승률이 높아야** 한다(손익비는 낮아도 무방)._"))
    L.extend(render_untouched(rep))
    L.extend(render_gate_report(rep))

    # --- 음성대조 (공통 렌더, 서론·라벨만 전략별) ---
    L.extend(render_control(
        rep, ctrl,
        intro=("같은 유니버스/타이밍의 marginal 을 흉내낸 무작위 진입↔수익 페어링을 "
               "동일 게이트에 태운 것. 무작위가 이 바를 실제 셋업만큼 자주 넘으면 바가 느슨. "
               "**Step 1(돌파)은 이 음성대조를 넘지 못했다** — 실 셋업 clean-OOS 2/4, OOS expR "
               "-0.029 가 무작위 평균 +0.078 아래였다. 눌림은 넘는가?"),
        actual_label="눌림"))

    # --- 레짐 ---
    L.append("## 레짐 정직성 (건당 기대값 레짐별)")
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
    L.append(f"사전등록 config(stop {PRE_STOP:.0%} / target {PRE_TARGET:.0%} / hold {PRE_HOLD_MAX})"
             "가 격자에서 **고원(plateau)** 위인지 홀로 튀는 spike 인지 확인용. 진입 신호는 고정, "
             "청산만 스윕.")
    L.append("")
    L.append("| stop | target | hold | n | OOS expR | 승률 | clean-OOS k/N | raw k/6 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in sens:
        star = ("  ← 사전등록" if (s["stop"] == PRE_STOP and s["target"] == PRE_TARGET
                                and s["hold"] == PRE_HOLD_MAX) else "")
        L.append(f"| {s['stop']:.2f} | {s['target']:.2f} | {s['hold']} | {s['n']} | "
                 f"{s['oos_expR']:+.3f} | {s['win_rate']:.0%} | "
                 f"{s['clean_k']}/{s['clean_v']} | {s['raw_k']}/{s['raw_v']}{star} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 사람이 읽는 kill/continue 메모 (코딩된 임계값 아님, R3)")
    L.append("")
    L.append("_아래는 게이트 러너가 남기는 관찰 프롬프트다. 최종 판단은 team-lead/사람이 한다._")
    L.append("- **되돌림 가설의 핵심**: 승률이 실제로 높은가(≥55~60%)? 낮으면 '되돌림'이 아니다.")
    L.append("- clean-OOS 폴드 카운트가 무작위 음성대조 도달 빈도를 유의미하게 넘는가"
             "(Step 1 은 넘지 못함)?")
    L.append("- 기대값이 monster top5/top20 소수 트레이드에 얹혀 있나(꼬리제거시 붕괴)?")
    L.append("- 건당 엣지가 특정 레짐에만 사는가(2018-2022 vs 2023-2026)?")
    L.append("- 미접촉창 [2025-07~2026-07) 이 폴드 결과와 같은 방향인가?")
    L.append("- 사전등록 config 가 민감도 격자에서 고원 위인가, 홀로 튀는 spike 인가?")
    L.append("")
    return "\n".join(L)


def _weekly_supply(entries: list[tuple[str, str]]) -> tuple[float, int]:
    """de-dup 후 진입의 주당(ISO주) 건수 중앙값·최대 — 공급 vs 4~8 슬롯 체크."""
    wk = weekly_count(entries)
    if not len(wk):
        return 0.0, 0
    return float(wk.median()), int(wk.max())


def run() -> None:
    prices = load_prices()
    p = build_panels(prices)
    date_pos = {str(d): i for i, d in enumerate(p["dates"])}

    mask = build_pullback_signal(p, rsi_max=RSI_MAX, adv_floor=ADV_FLOOR,
                                 full_template=True, require_reversal=True)
    n_entries_raw = int(mask.to_numpy().sum())
    entries = dedup_entries(mask, date_pos, min_gap=MIN_GAP)
    weekly_median, weekly_max = _weekly_supply(entries)

    # 사전등록 1차 config 시뮬 → prop_gate.
    fill_dates, rets = simulate_pullback_trades(
        entries, p, stop=PRE_STOP, target=PRE_TARGET, hold_max=PRE_HOLD_MAX)
    print(f"\n[extractor] 원신호(de-dup 전) {n_entries_raw} → de-dup {len(entries)} → "
          f"시뮬 트레이드 {len(rets)}  (주당 중앙값 {weekly_median:.0f}/최대 {weekly_max})")

    # 사전등록 config 를 넘기면 다중검정 원장(research/logs/pullback_prop/TRIALS.jsonl)에
    # 기록되고 N 이 거기서 읽힌다 — DSR·t-haircut 의 입력을 손으로 세지 않는다.
    rep = prop_gate(fill_dates, rets, PRE_STOP, label="pullback", log_dir=OUT_DIR, config={
        "adv_floor": ADV_FLOOR, "min_gap": MIN_GAP, "rsi_max": RSI_MAX,
        "stop": PRE_STOP, "target": PRE_TARGET, "hold_max": PRE_HOLD_MAX,
    })

    # 음성대조 — 실제 트레이드 수만큼 무작위 draw.
    ctrl = random_entry_control(
        fill_dates, rets, PRE_STOP, n_per_draw=len(rets), n_draws=200, seed=7)

    # 레짐 정직성.
    regime = regime_split(fill_dates, rets, rep["primary_cost"], PRE_STOP)

    # 탐색적 민감도(2차).
    sens = _sensitivity(entries, p)

    report = _fmt_verdict(rep, ctrl, regime, sens, n_entries_raw, weekly_median, weekly_max)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "VERDICT.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    run()
