#!/usr/bin/env python
"""프랍 스윙 게이트 러너 공통 헬퍼 — 세 러너에 복붙돼 있던 로직을 한 곳으로.

Step 1(minervini) / Step 2(pullback) / Step 3(pead_concentrated) 러너와 Step -1
(prop_feasibility) 는 같은 게이트 배터리(prop_gate)를 부르므로 주변 유틸(.env 로드,
레짐 분할, de-dup, 주당공급, 민감도 스윕 한 칸, VERDICT.md 게이트 섹션 렌더)이 사실상
동일했다. 이 모듈이 그 공통분모다 — **동작·숫자는 그대로**, 중복만 제거한다.

경계: research/experiments 계층에 산다(형제 prop_gate 를 import). research/signals 는
이 모듈을 import 하지 않는다(신호→실험 역방향 의존 금지). kr_quant 는 수정하지 않는다.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# 형제 모듈(같은 research/experiments) — 스크립트 실행시 그 디렉터리가 sys.path[0].
from prop_gate import prop_gate


def load_env_db() -> None:
    """.env 의 KR_QUANT_DB 를 환경에 실어줌(셸에 export 안 돼 있어도 동작).

    이미 환경에 있거나 .env 가 없으면 아무것도 안 한다. 러너들이 connect(db_default())
    전에 부르던 동일 관용을 한 곳으로 모은 것.
    """
    if os.environ.get("KR_QUANT_DB") or not os.path.exists(".env"):
        return
    for line in open(".env"):
        if line.startswith("KR_QUANT_DB"):
            os.environ["KR_QUANT_DB"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            break


def regime_split(fill_dates: np.ndarray, rets: np.ndarray, cost: float, stop: float) -> dict:
    """레짐 정직성 — 2018-2022 vs 2023-2026 건당 기대값(gross 평균·R·승률).

    각 창의 유한 gross 수익에서 상수 비용을 빼고 stop 으로 정규화한 R 로 집계한다.
    엣지가 한 레짐에만 사는지(레짐 포로) 드러내는 표.
    """
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


def dedup_gap(
    entries: list[tuple[str, str]], date_pos: dict[str, int], min_gap: int,
) -> list[tuple[str, str]]:
    """동일 종목 재진입에 ≥min_gap 거래일 간격을 강제(독립 스윙 프록시).

    ``entries`` = (date, code) 리스트. 한 종목이 min_gap 거래일 내 재진입 못 하게 —
    가장 이른 것부터 시간순으로 유지하고 그 다음은 min_gap 뒤에야 다시 잡는다.
    """
    kept: list[tuple[str, str]] = []
    last: dict[str, int] = {}
    for d, c in sorted(entries):                      # 시간순
        pos = date_pos[d]
        if c in last and pos - last[c] < min_gap:
            continue
        kept.append((d, c))
        last[c] = pos
    return kept


def weekly_count(entries: list[tuple[str, str]]) -> pd.Series:
    """(date, code) 리스트 → ISO 연-주 버킷별 진입 건수 Series. 비면 빈 Series.

    공급 vs 슬롯 용량 체크(중앙값·최대)의 공통 재료. 호출측이 median()/max()/len() 을 쓴다.
    """
    if not entries:
        return pd.Series(dtype=int)
    s = pd.to_datetime([d for d, _ in entries])
    return pd.Series(1, index=s).groupby([s.isocalendar().year, s.isocalendar().week]).sum()


def gate_sim(
    fd: np.ndarray,
    rr: np.ndarray,
    stop: float,
    label: str,
    *,
    config: dict,
    log_dir: str,
) -> tuple[dict, dict, dict, dict]:
    """민감도 스윕 한 칸 — prop_gate(verbose=False) 호출 후 자주 쓰는 조각을 함께 반환.

    반환: (rep, cost_sweep[0], folds, distribution). 각 러너의 _sensitivity 가 격자·행
    dict 는 스스로 만들고 이 게이트 호출·추출만 공유한다.

    ``config``/``log_dir`` 은 **필수**다(2026-08-16). 이전에는 둘 다 안 넘겨서 격자
    셀이 통째로 다중검정 원장을 우회했다 — pullback 은 18칸, pead_concentrated 는
    27칸을 돌면서 TRIALS.jsonl 에 각각 1줄·2줄만 남았고, ``gate_report`` 는
    ``n_trials <= 1`` 이면 deflation 을 정확히 0 으로 돌려주므로(``_expected_max_sharpe_h0``)
    두 알파의 Deflated Sharpe 에 **haircut 이 전혀 걸리지 않았다.** GUARDRAILS §6
    사전등록 템플릿이 "시도 예정 config 수(그리드는 민감도 전용): N"을 요구하는 이상
    격자 셀도 시행이다. 원장은 config fingerprint 로 dedupe 하므로 같은 격자를 다시
    돌려도 N 은 안 부풀려진다.
    """
    rep = prop_gate(
        fd, rr, stop, label=label, config=config, log_dir=log_dir, verbose=False
    )
    return rep, rep["cost_sweep"][0], rep["folds"], rep["distribution"]


# ---------------------------------------------------------------------------
# VERDICT.md 게이트 섹션 렌더 — 세 러너의 _fmt_verdict 에 복붙돼 있던 마크다운.
# 각 함수는 문자열 리스트를 돌려주고, 러너는 사전등록/전략별 행만 자기 것으로 유지한다.
# 출력은 기존과 **바이트 동일**(전략별 차이는 인자로만 노출).
# ---------------------------------------------------------------------------
def render_gate_core(rep: dict) -> list[str]:
    """"## 게이트 결과" 헤더 + [1] 슬리피지 스윕 + [2] 폴드 재현성 (세 러너 공통·동일)."""
    L: list[str] = []
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
    return L


def render_dist_fragility(rep: dict, *, win_bold: bool = False, note: str | None = None) -> list[str]:
    """[3] 분포 모양 + 취약성. ``note`` 는 헤더 뒤 삽입줄(눌림), ``win_bold`` 는 승률 강조(눌림)."""
    d = rep["distribution"]
    fr = rep["fragility"]
    tr = fr["tail_removal"]
    L: list[str] = []
    L.append("### [3] 분포 모양 + 취약성 (기준비용, OOS 트레이드)")
    L.append("")
    if note is not None:
        L.append(note)
        L.append("")
    wr = f"승률={d['win_rate']:.0%}"
    if win_bold:
        wr = f"**{wr}**"
    L.append(f"- n={d['n']}  기대값R={d['expectancy_R']:+.3f}  {wr}  "
             f"손익비={d['payoff']:.2f}  왜도={d['skew']:+.2f}  왼꼬리비중={d['left_tail_share']:.0%}")
    L.append(f"- monster top5={fr['monster_share_top5']:.0%}  top20={fr['monster_share_top20']:.0%}  "
             f"최장연패={fr['max_loss_streak']}  중앙값R={fr['median_trade']:+.3f}")
    L.append(f"- 꼬리제거(상위20): 기대값 {tr['expectancy_full']:+.3f} → {tr['expectancy_ex']:+.3f} "
             f"({'생존' if tr['expectancy_ex'] > 0 else '붕괴'})")
    L.append("")
    return L


def render_untouched(rep: dict, *, extra_note: str | None = None) -> list[str]:
    """[4] 미접촉 최종창(R1 held-out). ``extra_note`` 는 n줄 뒤 추가 주석(PEAD thin caveat)."""
    u = rep["untouched"]
    L: list[str] = []
    L.append(f"### [4] 미접촉 최종창 (R1 held-out) [{u['lo']}~{u['hi']}) — 폴드와 별개")
    L.append("")
    L.append(f"- n={u['n']}  기대값R={u['expectancy_R']:+.3f}  "
             f"승률={u['dist']['win_rate']:.0%}  왜도={u['dist']['skew']:+.2f}")
    if extra_note is not None:
        L.append(extra_note)
    L.append("")
    return L


def render_gate_report(rep: dict, *, ci_width: bool = False) -> list[str]:
    """[5] gate_report 배포신호. ``ci_width`` 면 CI 폭(thin 표본) 꼬리표를 붙인다(PEAD)."""
    gr = rep["gate_report"]
    ci = gr["expectancy_ci"]
    fc = gr["fold_consistency"]
    L: list[str] = []
    L.append("### [5] gate_report 배포신호 (REPORTER)")
    L.append("")
    ci_line = (f"- OOS n={gr['n']}  기대값R={gr['expectancy_R']:+.3f}  "
               f"95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    if ci_width:
        ci_line += f"  (**CI 폭 {ci[1] - ci[0]:.3f} R** — thin 표본)"
    L.append(ci_line)
    L.append(f"- 폴드 {fc['n_positive']}/{fc['n_folds']} 양수  monster={gr['monster_share']:.0%}  "
             f"최장연패={gr['max_loss_streak']}")
    L.append("")
    return L


def render_control(rep: dict, ctrl: dict, *, intro: str, actual_label: str) -> list[str]:
    """"## vs 랜덤 음성대조" — 3개 무작위 지표 줄은 공통, 서론/실제-라벨만 전략별 인자."""
    fb = rep["folds"]
    gr = rep["gate_report"]
    L: list[str] = []
    L.append("## vs 랜덤 음성대조 (R1 — 게이트 자체 위양성률 보정)")
    L.append("")
    L.append(intro)
    L.append("")
    L.append(f"- 무작위 raw ≥5/6 폴드 도달: **{ctrl['raw_ge5_frac']:.1%}** of draws")
    L.append(f"- 무작위 clean-OOS 전폴드(≥{ctrl['clean_fold_valid_median']}/"
             f"{ctrl['clean_fold_valid_median']}) 양수: **{ctrl['clean_all_positive_frac']:.1%}** of draws")
    L.append(f"- 무작위 OOS 기대값R: 평균 {ctrl['oos_expectancy_R_mean']:+.3f}  "
             f"p95 {ctrl['oos_expectancy_R_p95']:+.3f}")
    L.append(f"- **실제 {actual_label} clean-OOS = {fb['clean_oos_positive']}/{fb['clean_oos_valid']}, "
             f"OOS expR = {gr['expectancy_R']:+.3f}** — 위 무작위 분포 대비 어디에 서 있나(사람 판단).")
    L.append("")
    return L
