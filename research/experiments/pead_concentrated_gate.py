#!/usr/bin/env python
"""Step 3 — PEAD 를 **집중형 프랍 스윙**으로 재해석해 개별-트레이드 게이트에 태우는 얇은 러너.

플랜(.omc/plans/prop-swing-alpha.md) Step 3 + CONSENSUS REVISIONS(R1~R3) 를 따른다.

동기(pead_gate.py 와의 대비):
  pead_gate.py 는 PEAD 를 **분산형 북**으로 잰다 — 매 리밸런스마다 top_n=40 종목 전부를
  개별표본으로, 벤치마크-초과수익으로, 손절 없이. 결과: 폴드 재현성은 있으나(4/6) 건당
  +0.46% 로 얇다 = 프랍에 **틀린 모양**. 이 파일의 가설: 매 실적서프라이즈 리밸런스에서
  **확신 최상위 top_n(=1~3) 이름만** 골라 **절대수익**으로 **하드손절+보유상한** 스윙하면
  왼꼬리가 −1R 에 절단되고 오른꼬리만 남아 **두꺼운(프랍형)** 분포가 된다는 것.

  핵심 차이 3가지(모두 pead_gate 와 반대):
   1. top_n 을 40 → 1~3 으로 집중(분산 평균화 제거).
   2. 벤치마크-초과 → **절대수익**(그날 유니버스 평균을 빼지 않는다).
   3. 손절 없음 → **하드손절 + 보유상한**(왼꼬리 절단 = R-멀티플이 −1R 근처).

Step -1(feasibility) 경고: 이 셋업은 THIN/temporally sparse — clean-OOS 폴드당 ~36 트레이드,
연 ~12 distinct 리밸런스일. CI 는 넓다. 표본을 정직하게 보고하고 억지 판정하지 않는다.

R1(사전등록·비협상): 게이트 출력을 보기 **전에** 딱 하나의 1차 config 를 정론으로 못박는다.
그 하나가 THE 테스트다. 파라미터 그리드는 **탐색적 민감도(plateau 확인)** 로만 — 승자픽 금지.

no-lookahead(Principle 4): 신호(YoY 서프라이즈)는 리밸런스일 t 종가에 알려짐(earnings_yoy_panel
이 avail_date 로 이미 lookahead-safe). **다음 세션(t+1) 종가 진입**, 이후 하드손절/보유상한 청산.
서프라이즈 임계값(top-decile)은 **TRAIN(진입<2022) 에서만 학습**해 전방 적용.

데이터: split-adjusted (pead_refinement.PRICE_TABLE=daily_bars_adjusted). raw daily_bars 금지.
_context 는 종가(C)만 노출하므로 손절/청산은 **종가 기준**(장중 저가 관통 미탐지) — 정직한 caveat.

판정은 재발명 금지 — research/experiments/prop_gate.py 의 prop_gate/random_entry_control 을
그대로 호출한다. 결과는 research/logs/pead_concentrated/VERDICT.md 로(사람이 읽는 리포트,
코딩된 판정 없음 — R3).

라이브러리 경계: src/kr_quant 를 수정하지 않고 research 형제 모듈만 재사용한다.

실행(DB 필요): docker start quant-airflow-timescaledb-1 후
    uv run python research/experiments/pead_concentrated_gate.py
"""

from __future__ import annotations

import os

import numpy as np

# 형제 모듈(같은 research/experiments) — 스크립트 실행시 그 디렉터리가 sys.path[0].
from pead_gate import extract_trades as diversified_extract
from pead_refinement import BASELINE, START_INDEX, _context, load_data
from prop_gate import prop_gate, random_entry_control
from prop_swing_common import (
    gate_sim,
    load_env_db,
    render_control,
    render_dist_fragility,
    render_gate_core,
    render_gate_report,
    render_untouched,
)

from kr_quant.diagnostics.fragility import monster_share
from kr_quant.diagnostics.r_distribution import dist_shape

# ===========================================================================
# 사전등록 1차 CONFIG (R1) — 게이트 출력을 보기 전에 못박음. 이것이 THE 테스트.
# ---------------------------------------------------------------------------
# 근거(정론 + Step -1 feasibility + pead_gate 대비):
#  · top_n = 3 — Step -1 에서 top3 가 "MEASURABLE BUT THIN"(clean 폴드 ~36). top_n=1 은
#    표본이 1/3 로 붕괴해 CI 가 측정 불가; top_n=3 이 "집중형이면서 측정가능"의 균형점.
#  · hold = 10 거래일 — 프랍 스윙(1~10일)의 상한. PEAD 드리프트는 60일 효과지만 여기선
#    "스윙"으로 재해석하므로 짧은 보유 + 손절이 본질(드리프트 전량회수가 목적이 아님).
#  · stop = 0.10 — 실적서프라이즈 종목은 고변동. 10% 손절이 캔들 노이즈에 조기 털리지
#    않으면서 왼꼬리를 −1R 에 절단. R 정규화 분모 = 0.10.
#  · 서프라이즈 임계값 = TRAIN top-decile(90분위) YoY 절대 floor — "최고확신"만 진입.
#    TRAIN 에서만 학습해 전방 적용(no-lookahead). top_n=3 이 이미 강한 순 상위라
#    이 floor 는 대개 거의 안 물린다(투명하게 제거 트레이드 수를 보고).
#  · 리밸런스 = pead BASELINE step(20거래일), 유니버스 ADV floor = BASELINE adv_floor —
#    Step -1 / pead_gate 와 동일 카덴스·유니버스로 apples-to-apples.
# ===========================================================================
PRE_TOP_N = 3                       # 집중도(1차). pead_gate 는 40 = 분산형.
PRE_HOLD = 10                       # 보유상한(거래일)
PRE_STOP = 0.10                     # 하드손절폭 (= R 정규화 분모)
PRE_SURPRISE_Q = 0.90              # TRAIN top-decile YoY 절대 floor(전방 적용)

REBALANCE_STEP = BASELINE["step"]           # 20 — pead BASELINE 과 동일 카덴스
ADV_FLOOR = BASELINE["adv_floor"]           # 20000 (~20B KRW) — 동일 유니버스
TRAIN_HI = "2022-01-01"                      # no-lookahead 경계(진입<이 날짜 = TRAIN)
OUT_DIR = "research/logs/pead_concentrated"


def _rebalance_ts(nD: int, step: int, hold: int) -> list[int]:
    """리밸런스 신호일 t 목록: START_INDEX 부터 step 간격, t+1 진입 + hold 청산이 데이터 안."""
    return [t for t in range(START_INDEX, nD - hold - 2, step)]


def train_surprise_floor(ctx: dict, *, step: int, adv_floor: float, q: float,
                         train_hi: str, hold: int) -> float:
    """TRAIN(진입<train_hi) 리밸런스일의 eligible YoY 서프라이즈 q-분위 = 전방 적용 floor.

    no-lookahead: 임계값을 TRAIN 에서만 학습한다(Principle 4). 진입일 = dates[t+1] 로 판정.
    """
    sig_m, adv, dates, nD = ctx["sig_m"], ctx["adv"], ctx["dates"], ctx["nD"]
    vals: list[float] = []
    for t in _rebalance_ts(nD, step, hold):
        if str(dates[t + 1])[:10] >= train_hi:      # 진입일(t+1)이 OOS 면 학습 제외
            continue
        ok = np.isfinite(sig_m[:, t]) & (adv[:, t] >= adv_floor)
        vals.extend(sig_m[ok, t].tolist())
    if not vals:
        return float("-inf")                          # TRAIN 표본 없으면 floor 무력화
    return float(np.quantile(np.asarray(vals, float), q))


def extract_concentrated(
    ctx: dict, *, top_n: int, hold: int, stop: float, step: int,
    adv_floor: float, surprise_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """집중형 개별-트레이드 추출 — 매 리밸런스 top_n 이름, **절대수익**, 하드손절+보유상한.

    pead_gate.extract_trades 와 정반대 3점: (a) top_n 소수(분산 아님), (b) 벤치마크 안 뺌
    (절대수익), (c) 손절/보유상한(왼꼬리 −1R 절단). 신호 t종가 → **t+1 종가 진입** → 이후
    종가로 손절/보유상한 청산(_context 는 종가만 → 장중 저가 미탐지, caveat).

    반환: (entry_dates=t+1 ISO, returns=절대 gross, meta) — 비용은 prop_gate 안에서 사후 차감.
    """
    C, sig_m, adv, dates, nD = ctx["C"], ctx["sig_m"], ctx["adv"], ctx["dates"], ctx["nD"]

    ent: list[str] = []
    ret: list[float] = []
    n_floor_removed = 0                               # floor 가 걷어낸 (리밸런스 슬롯) 수
    n_rebal_used = 0
    for t in _rebalance_ts(nD, step, hold):
        ok = np.isfinite(sig_m[:, t]) & (adv[:, t] >= adv_floor)
        idx = np.where(ok)[0]
        if idx.size == 0:
            continue
        # 강한 순 top_n 후보 → 서프라이즈 floor 미달은 절사(집중형이므로 대개 통과).
        cand = idx[np.argsort(-sig_m[idx, t])[:top_n]]
        keep = cand[sig_m[cand, t] >= surprise_floor]
        n_floor_removed += int(len(cand) - len(keep))
        if len(keep) == 0:
            continue
        n_rebal_used += 1
        e = t + 1                                     # t+1 종가 진입(다음 세션)
        for i in keep:
            entry_px = C[i, e]
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue
            stop_px = entry_px * (1.0 - stop)
            end = min(e + hold, nD - 1)               # 보유상한 바
            exit_px = None
            for k in range(e + 1, end + 1):           # 진입 다음날부터 종가로 청산 검사
                px = C[i, k]
                if not np.isfinite(px):               # NaN/상폐: 직전 유효 종가로 청산
                    exit_px = C[i, k - 1]
                    break
                if px <= stop_px:                     # 하드손절(종가 관통) → 그 종가 청산
                    exit_px = px
                    break
            if exit_px is None:                       # 미청산 → 보유상한 종가
                exit_px = C[i, end]
                if not np.isfinite(exit_px):
                    col = C[i, e:end + 1]
                    fin = col[np.isfinite(col)]
                    if not len(fin):
                        continue
                    exit_px = fin[-1]
            if not np.isfinite(exit_px) or exit_px <= 0:
                continue
            ent.append(str(dates[e])[:10])            # 진입일 = t+1(자본 투입 시점)
            ret.append(float(exit_px / entry_px - 1.0))   # 절대수익(벤치마크 안 뺌)
    meta = {"n_floor_removed": n_floor_removed, "n_rebalances_used": n_rebal_used}
    return np.asarray(ent), np.asarray(ret, float), meta


def diversified_baseline_shape(ctx: dict) -> dict:
    """pead_gate.py 분산형 북의 분포 모양(대조군) — 집중이 분포를 무엇으로 바꿨나.

    pead_gate.extract_trades(BASELINE): top_n=40, 벤치마크-초과, 손절 없음. 여기선 그
    **초과수익 분포의 모양**(왜도·승률·꼬리·monster)만 뽑아 집중형 R-분포와 대조한다.
    NOTE: 초과수익(%)이지 R 이 아니다 — 손절이 없어 R 정규화가 의미 없음. 모양만 비교.
    """
    ent, exc = diversified_extract(ctx, **BASELINE)
    oos = ent >= TRAIN_HI                              # 집중형과 동일하게 OOS 만
    d = dist_shape(exc[oos])
    return {
        "n_all": int(len(exc)), "n_oos": int(oos.sum()),
        "mean_excess": float(np.mean(exc[oos])) if oos.any() else float("nan"),
        "win_rate": d["win_rate"], "skew": d["skew"],
        "left_tail_share": d["left_tail_share"],
        "monster_top5": monster_share(exc[oos], k=5),
        "monster_top20": monster_share(exc[oos], k=20),
    }


def _sensitivity(ctx: dict, floor: float) -> list[dict]:
    """탐색적 민감도(2차, R1) — top_n×hold×stop 격자. plateau 확인용, 승자픽 아님.

    사전등록 config(top3/hold10/stop0.10)가 격자에서 고원 위인지 홀로 튀는 spike 인지.
    같은 TRAIN-학습 floor 를 전 격자에 적용(no-lookahead 유지).
    """
    rows: list[dict] = []
    grid = [(tn, h, s)
            for tn in (1, 2, 3)
            for h in (5, 10, 20)
            for s in (0.05, 0.07, 0.10)]
    for tn, h, s in grid:
        ent, ret, _ = extract_concentrated(
            ctx, top_n=tn, hold=h, stop=s, step=REBALANCE_STEP,
            adv_floor=ADV_FLOOR, surprise_floor=floor)
        rep, cs0, fb, _ = gate_sim(ent, ret, s, f"sens_{tn}_{h}_{s}")
        rows.append({
            "top_n": tn, "hold": h, "stop": s, "n": rep["n_total"],
            "oos_expR": cs0["oos_expectancy_R"],
            "clean_k": fb["clean_oos_positive"], "clean_v": fb["clean_oos_valid"],
            "raw_k": fb["raw_positive"], "raw_v": fb["raw_valid"],
        })
    return rows


def _fmt_verdict(rep, ctrl, div, sens, floor, meta) -> str:
    """VERDICT.md — 사전등록 블록 먼저, 그 아래 게이트 숫자(코딩된 판정 없음, R3)."""
    L: list[str] = []
    L.append("# PEAD 집중형 — 개별-트레이드 게이트 VERDICT (Step 3)")
    L.append("")
    L.append("> **리포터 문서다 (R3).** 아래 숫자를 사람이 읽고 kill/continue 를 판단한다. "
             "코딩된 PASS/FAIL 임계값 없음.")
    L.append("")

    # --- 사전등록 ---
    L.append("## 사전등록 1차 CONFIG (R1 — 결과를 보기 전에 확정)")
    L.append("")
    L.append("이 **딱 하나의 config 가 THE 테스트다.** 그리드 argmax 로만 사는 값은 합격 아님.")
    L.append("")
    L.append(f"- **top_n = {PRE_TOP_N}** — 매 리밸런스 서프라이즈 확신 최상위 {PRE_TOP_N}개만 "
             "(pead_gate 는 40 = 분산형). Step -1: top3 = MEASURABLE BUT THIN.")
    L.append(f"- **보유상한 hold = {PRE_HOLD}거래일**, **하드손절 stop = {PRE_STOP:.0%}** "
             "(R 정규화 분모). 왼꼬리를 −1R 에 절단.")
    L.append(f"- **서프라이즈 floor = TRAIN top-decile YoY(q={PRE_SURPRISE_Q:.2f}) = "
             f"{floor:+.3f}** — TRAIN(진입<{TRAIN_HI})에서만 학습해 전방 적용(no-lookahead).")
    L.append(f"- **절대수익**(벤치마크 안 뺌), 리밸런스 {REBALANCE_STEP}거래일 간격, "
             f"유니버스 ADV≥{ADV_FLOOR:.0f}(pead BASELINE 동일).")
    L.append("- **타이밍:** 신호 t종가 → **t+1 종가 진입** → 이후 종가로 손절/보유상한 청산.")
    L.append("")
    L.append(f"floor 가 걷어낸 (top_n 후보) 슬롯 = {meta['n_floor_removed']}개, "
             f"실제 사용 리밸런스 {meta['n_rebalances_used']}회 → 트레이드 {rep['n_total']}건, "
             f"진입 {rep['entry_range'][0]}~{rep['entry_range'][1]}.")
    L.append("")
    L.append("**pead_refinement 정론 재해석 caveat:** PEAD 는 검증된 60일 드리프트 알파다. "
             f"여기서 {PRE_HOLD}일 보유 + 손절로 **스윙화**하는 것은 원 알파의 지평을 자르는 "
             "재해석 — 드리프트 전량회수가 아니라 '프랍 모양(두꺼운 건당)' 여부만 묻는다.")
    L.append("")
    L.append("---")
    L.append("")

    # --- 표본 caveat (이 셋업의 정직한 진실말하기) ---
    L.append("## 표본 크기 caveat (Step -1 경고 — 이 셋업의 핵심 진실)")
    L.append("")
    L.append("Step -1 feasibility: PEAD-top3 = **MEASURABLE BUT THIN**. clean-OOS 폴드당 "
             "~36 트레이드, **연 ~12 distinct 리밸런스일** (top_n 이 한 날짜에 뭉침). "
             "→ 유효 독립 시점 ≈ 12/년 → **폴드 CI 는 매우 넓다.** 아래 숫자는 이 한계 "
             "안에서 읽어야 하며, 좁은 신뢰구간을 가장하지 않는다.")
    L.append("")
    fb = rep["folds"]
    L.append("| 구간 | clean-OOS 폴드별 n |")
    L.append("|---|---|")
    clean_ns = ", ".join(f"{r['test_window']}={r['n']}" for r in fb["rows"] if not r["inside_train"])
    L.append(f"| 폴드별 트레이드 수 | {clean_ns} |")
    gr = rep["gate_report"]
    ci = gr["expectancy_ci"]
    L.append(f"| OOS 전체 | n={gr['n']}, 95%CI 폭 = {ci[1] - ci[0]:.3f} R |")
    L.append("")
    L.append("---")
    L.append("")

    # --- 게이트 결과 (공통 렌더) ---
    L.extend(render_gate_core(rep))
    L.extend(render_dist_fragility(rep))
    L.extend(render_untouched(
        rep, extra_note="  (Step -1: 미접촉창은 12 distinct 일에 36건 — 이 창 하나로 확정 판단 금물)"))
    L.extend(render_gate_report(rep, ci_width=True))

    # --- 음성대조 (공통 렌더, 서론·라벨만 전략별) ---
    L.extend(render_control(
        rep, ctrl,
        intro=("같은 유니버스/타이밍의 marginal 을 흉내낸 무작위 진입↔수익 페어링을 동일 게이트에 "
               "태운 것. 무작위가 이 바를 실제 셋업만큼 자주 넘으면 바가 느슨(thin 표본에선 특히 중요)."),
        actual_label="PEAD집중"))

    # 아래 분산형 대조 표가 참조하는 집중형 분포·취약성 조각.
    d = rep["distribution"]
    fr = rep["fragility"]

    # --- 분산형 대조 ---
    L.append("## vs pead_gate.py 분산형 베이스라인 (집중이 분포 모양을 무엇으로 바꿨나)")
    L.append("")
    L.append("pead_gate: top_n=40, 벤치마크-초과, 손절 없음(분산형 북). 아래는 그 **OOS 초과수익 "
             "분포 모양** vs 이 파일의 **집중형 R-분포**. NOTE: 분산형은 초과수익(%)·손절 없음이라 "
             "R 이 아님 — 모양(왜도·승률·꼬리·monster)만 대조.")
    L.append("")
    L.append("| 지표 | 분산형(pead_gate, 초과%) | 집중형(이 파일, R) |")
    L.append("|---|---|---|")
    L.append(f"| OOS n | {div['n_oos']} | {d['n']} |")
    L.append(f"| 건당 평균 | {div['mean_excess']:+.4f} (초과) | {d['expectancy_R']:+.3f} R |")
    L.append(f"| 승률 | {div['win_rate']:.0%} | {d['win_rate']:.0%} |")
    L.append(f"| 왜도 | {div['skew']:+.2f} | {d['skew']:+.2f} |")
    L.append(f"| 왼꼬리비중 | {div['left_tail_share']:.0%} | {d['left_tail_share']:.0%} |")
    L.append(f"| monster top5 | {div['monster_top5']:.0%} | {fr['monster_share_top5']:.0%} |")
    L.append(f"| monster top20 | {div['monster_top20']:.0%} | {fr['monster_share_top20']:.0%} |")
    L.append("")
    L.append("→ 집중+손절이 왼꼬리를 −1R 에 절단하고 오른꼬리 의존도(monster/왜도)를 "
             "**늘렸는지/줄였는지**를 읽는다. monster 가 크게 오르면 '두꺼워진' 게 아니라 "
             "'소수 괴물 의존'으로 옮겨간 것(프랍에 더 나쁨).")
    L.append("")

    # --- 민감도(2차) ---
    L.append("## 탐색적 민감도 (2차, R1 — plateau vs spike, 승자픽 아님)")
    L.append("")
    L.append("사전등록 config(top3/hold10/stop0.10)가 격자에서 **고원(plateau)** 위인지 홀로 튀는 "
             "spike 인지 확인용. spike 면 신뢰도 낮음. thin 표본이라 격자값도 CI 넓음(참고).")
    L.append("")
    L.append("| top_n | hold | stop | n | OOS expR | clean-OOS k/N | raw k/6 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in sens:
        star = "  ← 사전등록" if (s["top_n"] == PRE_TOP_N and s["hold"] == PRE_HOLD
                                and s["stop"] == PRE_STOP) else ""
        L.append(f"| {s['top_n']} | {s['hold']} | {s['stop']:.2f} | {s['n']} | "
                 f"{s['oos_expR']:+.3f} | {s['clean_k']}/{s['clean_v']} | "
                 f"{s['raw_k']}/{s['raw_v']}{star} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 사람이 읽는 kill/continue 메모 (코딩된 임계값 아님, R3)")
    L.append("")
    L.append("_아래는 게이트 러너가 남기는 관찰 프롬프트다. 최종 판단은 team-lead/사람이 한다._")
    L.append("- clean-OOS 폴드 카운트가 무작위 음성대조 도달 빈도를 유의미하게 넘는가? "
             "(thin 표본 → 넘어도 CI 폭을 함께 볼 것)")
    L.append("- 집중형 R-분포가 분산형보다 **왼꼬리 절단·낮은 monster** 로 '프랍형'이 됐나, "
             "아니면 monster/왜도만 커졌나(소수 괴물 의존 = 프랍에 더 나쁨)?")
    L.append("- 기대값 95%CI 가 0 을 넉넉히 넘는가, 폭이 너무 넓어(±수 R) 측정 불가인가?")
    L.append("- 미접촉창(12일·36건)이 폴드와 같은 방향인가 — 단 이 창 하나로 확정 금물.")
    L.append("- 사전등록 config 가 민감도 격자에서 고원인가, 홀로 튀는 spike 인가?")
    L.append("")
    return "\n".join(L)


def run() -> None:
    load_env_db()
    print("=== 데이터 로드 (TimescaleDB, 분할조정) ===")
    prices, yoy = load_data()
    ctx = _context(prices, yoy)
    print(f"  패널: {len(ctx['codes'])} 종목 × {ctx['nD']} 거래일" if "codes" in ctx
          else f"  패널: {ctx['nD']} 거래일")

    # 서프라이즈 floor 를 TRAIN 에서만 학습(no-lookahead).
    floor = train_surprise_floor(
        ctx, step=REBALANCE_STEP, adv_floor=ADV_FLOOR, q=PRE_SURPRISE_Q,
        train_hi=TRAIN_HI, hold=PRE_HOLD)
    print(f"  TRAIN top-decile YoY floor = {floor:+.3f}")

    # 사전등록 1차 config 추출 → prop_gate.
    ent, ret, meta = extract_concentrated(
        ctx, top_n=PRE_TOP_N, hold=PRE_HOLD, stop=PRE_STOP, step=REBALANCE_STEP,
        adv_floor=ADV_FLOOR, surprise_floor=floor)
    print(f"[extractor] 리밸런스 사용 {meta['n_rebalances_used']}회, "
          f"floor 제거 {meta['n_floor_removed']} → 트레이드 {len(ret)}건")

    # 사전등록 config → 다중검정 원장 기록 + N 자동 산출(DSR·t-haircut 입력).
    rep = prop_gate(ent, ret, PRE_STOP, label="pead_concentrated", log_dir=OUT_DIR, config={
        "top_n": PRE_TOP_N, "hold": PRE_HOLD, "stop": PRE_STOP,
        "surprise_q": PRE_SURPRISE_Q, "step": REBALANCE_STEP, "adv_floor": ADV_FLOOR,
    })

    # 음성대조 — 실제 트레이드 수만큼 무작위 draw.
    ctrl = random_entry_control(ent, ret, PRE_STOP, n_per_draw=len(ret), n_draws=200, seed=7)

    # 분산형 대조 + 탐색적 민감도(2차).
    div = diversified_baseline_shape(ctx)
    sens = _sensitivity(ctx, floor)

    report = _fmt_verdict(rep, ctrl, div, sens, floor, meta)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "VERDICT.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    run()
