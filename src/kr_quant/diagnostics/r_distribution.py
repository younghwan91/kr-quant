"""R-멀티플 분포 진단 — 복리 자본곡선이 아니라 **개별 트레이드 = 표본**의 분포 모양.

성공한 트레이더와 평균-백테스트의 차이: 백테스트는 신호 전부를 평균 1개 숫자로
붕괴시킨다("엣지 = 모든 신호의 평균"). 트레이더는 개별 트레이드를 R-멀티플
(수익÷진입시 리스크) 표본으로 쌓아 **분포의 모양**을 본다 — 손절이 왼꼬리를
−1R에 자르는가, 선별·승자태우기가 오른꼬리를 살찌우는가.

R-멀티플 = 트레이드 수익 ÷ 진입시 리스크(=하드손절 폭). 손절 = −1R,
+30%(손절10%) = +3R. 복리·사이징은 여기서 안 한다 — 그건 분포가 정해진 뒤의
자금관리 문제. 여기선 분포의 모양만 본다.

이건 stochastic이다(deterministic 아님): 한 백테스트 경로는 분포에서 뽑은 한 표본.
판정은 '분포의 모양이 TRAIN·OOS 둘 다에서 같은 방향으로 굽나'로 — 한 경로의 극값이
아니라. 이 모듈은 순수 배열(수익, 진입일, 확신 피처, 손절폭)만 받는다.
research/ 로부터 아무것도 import 하지 않는다 — 라이브러리 경계.

Provenance(git 이력 — 원본은 추출과 함께 삭제): research/distribution.py 의 r_multiples / dist_shape / selection_curve /
conviction_analysis / hold_curve 를 print 기반에서 구조화된 dict 반환으로 일반화.
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def r_multiples(ret: np.ndarray, stop: float) -> np.ndarray:
    """개별 트레이드 수익 → R-멀티플(÷진입시 리스크). 손절=−1R 근처, 큰 승자=고R.

    ``stop`` 은 진입시 하드손절 폭(양수 소수, 예: 0.10=10%). 유한값만 남긴다.
    """
    r = np.asarray(ret, float)
    r = r[np.isfinite(r)]
    return r / stop


def dist_shape(
    R: np.ndarray,
    *,
    left_tail: float = -0.9,
    right_tails: tuple[float, ...] = (3.0, 5.0, 10.0),
) -> dict:
    """개별표본 R-분포의 '모양' — 왼꼬리 절단·오른꼬리 두께·기대값(R). 복리 없음.

    반환 dict:
        n, expectancy_R(기대값), win_rate(승률), payoff(손익비), skew(왜도),
        left_tail_share(≤left_tail 질량 비중 — 리스크가 −1R에 절단되면 건강),
        avg_loss_R(평균 패), pos_mass(양수익 총질량),
        right_tail: {thr: {"freq": 빈도, "share": 양수익질량 중 점유}} — 오른꼬리 두께.
    """
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    n = int(len(R))
    if n == 0:
        return {"n": 0, "expectancy_R": NAN, "win_rate": NAN, "payoff": NAN,
                "skew": NAN, "left_tail": left_tail, "left_tail_share": NAN,
                "avg_loss_R": NAN, "pos_mass": 0.0,
                "right_tail": {thr: {"freq": NAN, "share": NAN} for thr in right_tails}}
    wins, losses = R[R > 0], R[R < 0]
    std = R.std()
    pos_mass = float(wins.sum())
    right: dict[float, dict] = {}
    for thr in right_tails:
        m = R >= thr
        share = float(R[m].sum() / pos_mass) if pos_mass > 0 else 0.0
        right[thr] = {"freq": float(m.mean()), "share": share}
    return {
        "n": n,
        "expectancy_R": float(R.mean()),
        "win_rate": float((R > 0).mean()),
        "payoff": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else NAN,
        "skew": float(((R - R.mean()) ** 3).mean() / std ** 3) if std > 0 else NAN,
        "left_tail": left_tail,
        "left_tail_share": float((R <= left_tail).mean()),
        "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
        "pos_mass": pos_mass,
        "right_tail": right,
    }


def selection_curve(
    R: np.ndarray,
    feature: np.ndarray,
    *,
    fracs: tuple[float, ...] = (1.00, 0.50, 0.30, 0.10),
    tail: float = 3.0,
    min_n: int = 5,
) -> list[dict]:
    """**선별 곡선** — 확신 피처 상위 X%만 잡으면 R-분포가 오른쪽으로 굽나(a-priori).

    전부 잡음(frac=1.0) → 선별(상위%)로 갈수록 기대값R·오른꼬리 빈도가 오르면
    '선별이 분포를 굽힌다' = 진입시점에 알 수 있는 알파. 평균 백테스트가 희석한 그것.

    ``feature`` 는 진입시점 확신 신호(예: 모멘텀 강도). 강한 순으로 정렬해 상위 k개.
    각 fraction 마다 dict 반환: frac, n, expectancy_R, win_rate, tail_freq(≥tail 빈도),
    tail_share(양수익 중 ≥tail 점유), payoff.
    """
    R = np.asarray(R, float)
    feature = np.asarray(feature, float)
    fin = np.isfinite(R) & np.isfinite(feature)
    R, feature = R[fin], feature[fin]
    order = np.argsort(feature)[::-1]  # 확신 강한 순
    rows: list[dict] = []
    for frac in fracs:
        k = min(len(R), max(min_n, int(len(R) * frac)))
        sel = R[order[:k]]
        wins, losses = sel[sel > 0], sel[sel < 0]
        pos = wins.sum()
        tmask = sel >= tail
        rows.append({
            "frac": float(frac),
            "n": int(k),
            "expectancy_R": float(sel.mean()) if k else NAN,
            "win_rate": float((sel > 0).mean()) if k else NAN,
            "tail_freq": float(tmask.mean()) if k else NAN,
            "tail_share": float(sel[tmask].sum() / pos) if pos > 0 else 0.0,
            "payoff": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else NAN,
        })
    return rows


def conviction_analysis(
    feature: np.ndarray,
    R: np.ndarray,
    entry: np.ndarray,
    *,
    train_hi: str,
    nq: int = 5,
    tail: float = 3.0,
    min_per_q: int = 5,
) -> dict:
    """확신 신호(feature) 분위별 R-기대값 — TRAIN/OOS 각각. a-priori 예측력(과최적 방지).

    진입일(``entry``, ISO 문자열 또는 비교가능 배열) < ``train_hi`` 를 TRAIN 으로 분리.
    각 분위 Q1..Qnq 별 (n, expectancy_R, tail_freq) 을 TRAIN·OOS 로 계산.
    단조성 판정(Q_nq > Q_1)이 TRAIN·OOS 둘 다 참이면 예측력 있음(선별 값어치),
    아니면 불안정(꼬리는 대체로 운). 이 verdict 는 서술적 분석 라벨이지
    배포 PASS/FAIL 이 아니다.

    반환: {nq, train: [분위 dict...], oos: [...], train_monotonic, oos_monotonic, verdict}.
    표본이 nq*min_per_q 미만이면 해당 그룹은 None.
    """
    feature = np.asarray(feature)
    R = np.asarray(R, float)
    entry = np.asarray(entry)
    train_m = entry < train_hi
    out: dict[str, list[dict] | None] = {}
    for nm, mask in (("train", train_m), ("oos", ~train_m)):
        f = np.asarray(feature[mask], float)
        r = R[mask]
        fin = np.isfinite(f) & np.isfinite(r)
        f, r = f[fin], r[fin]
        if len(r) < nq * min_per_q:
            out[nm] = None
            continue
        qs = np.quantile(f, np.linspace(0, 1, nq + 1))
        bucket = np.clip(np.digitize(f, qs[1:-1]), 0, nq - 1)
        out[nm] = [{
            "q": q + 1,
            "n": int((bucket == q).sum()),
            "expectancy_R": float(r[bucket == q].mean()) if (bucket == q).any() else NAN,
            "tail_freq": float(np.mean(r[bucket == q] >= tail)) if (bucket == q).any() else NAN,
        } for q in range(nq)]
    tr_mono = oo_mono = None
    verdict = None
    if out["train"] and out["oos"]:
        tr_mono = out["train"][-1]["expectancy_R"] > out["train"][0]["expectancy_R"]
        oo_mono = out["oos"][-1]["expectancy_R"] > out["oos"][0]["expectancy_R"]
        verdict = "predictive" if (tr_mono and oo_mono) else "unstable"
    return {"nq": nq, "train": out["train"], "oos": out["oos"],
            "train_monotonic": tr_mono, "oos_monotonic": oo_mono, "verdict": verdict}


def hold_curve(
    results: dict,
    *,
    right_tails: tuple[float, ...] = (3.0, 5.0),
    top_frac: float = 0.10,
    time_reason: int = 0,
) -> list[dict]:
    """보유상한 스윕 — 더 오래 타면 오른꼬리가 두꺼워지나(승자 태우기). R-분포 모양.

    이 모듈은 시뮬레이터를 모른다 — 호출자가 각 hold 값에 대해 시뮬을 돌린 결과를
    ``results = {hold: {"R": R배열, "reason": 청산사유배열(옵션)}}`` 로 넘긴다
    (이미 OOS 등으로 필터된 배열). 라이브러리 경계 유지.

    각 hold 마다 dict: hold, expectancy_R, top_share(양수익 중 상위 top_frac 점유),
    freq_ge_{thr}R, (reason 있으면) time_exit_share(reason==time_reason 비중).
    """
    rows: list[dict] = []
    for h in sorted(results):
        R = np.asarray(results[h]["R"], float)
        reason = results[h].get("reason")
        fin = np.isfinite(R)
        R = R[fin]
        pos = R[R > 0].sum()
        k = max(1, int(len(R) * top_frac))
        row = {
            "hold": h,
            "expectancy_R": float(R.mean()) if len(R) else NAN,
            "top_share": float(np.sort(R)[::-1][:k].sum() / pos) if pos > 0 else 0.0,
        }
        for thr in right_tails:
            row[f"freq_ge_{thr:g}R"] = float(np.mean(R >= thr)) if len(R) else NAN
        if reason is not None:
            reason = np.asarray(reason)[fin]
            row["time_exit_share"] = float(np.mean(reason == time_reason)) if len(reason) else NAN
        rows.append(row)
    return rows
