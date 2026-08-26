"""취약성 진단 — 양(+)의 기대값이 소수의 꼬리 사건에 얹혀 있으면 통계적 확신은 낮다.

트레이더가 실제 겪는 시퀀스는 '평균'이 아니다. 괴물(상위 몇 건)이 총이익의 대부분을
만들고, 그 몇 건을 빼면 엣지가 사라진다면 — 그 엣지는 재현성이 낮은 운에 가깝다.
이 모듈은 그 취약성을 개별 트레이드 **R-멀티플 분포** 관점에서만 측정한다.
포트폴리오 프레이밍(슬롯·동시보유·연환산·회전율)은 여기 없다 — 오직 표본 분포.

Provenance: scratchpad/trader_view.py 의 괴물 의존도·연속손실·꼬리제거·중앙값·
승리조건부 분포 공식을 라이브러리로 통합(scratchpad 는 Phase 2 에서 삭제됨).
scratchpad/deploy_check.py 의 포트폴리오 지표(회전·연환산-슬롯당·동시보유)는
의도적으로 제외 — 사용자가 포트폴리오 프레이밍을 거부.

모든 함수는 순수 R-멀티플 배열만 받는다. research/ 로부터 아무것도 import 하지 않는다.
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def monster_share(R: np.ndarray, *, k: int = 5) -> float:
    """괴물 의존도 — 총 P&L 중 상위 k 건(최대 승자)이 만든 비중.

    총이익의 대부분을 소수 트레이드가 만들면 엣지의 통계적 확신은 낮다.
    총합이 0이면 정의불가(NAN). R 단위 P&L = R 의 합.
    """
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    total = R.sum()
    if len(R) == 0 or total == 0:
        return NAN
    top = np.sort(R)[::-1][:k].sum()
    return float(top / total)


def max_loss_streak(R: np.ndarray, *, entry: np.ndarray | None = None, loss_thr: float = 0.0) -> int:
    """최장 연속 손실(≤loss_thr) 길이 — 트레이더가 견뎌야 하는 최악의 연패.

    ``entry`` 를 주면 시간순(실제 겪는 순서)으로 정렬 후 계산. 유한값만 센다.
    """
    R = np.asarray(R, float)
    if entry is not None:
        R = R[np.argsort(np.asarray(entry))]  # 시간순 (실제 겪는 순서)
    R = R[np.isfinite(R)]
    loss = R <= loss_thr
    if not loss.any():
        return 0
    # 연패 런 길이 = 손실 누적수 - 직전 승리 시점의 누적수(런별 오프셋).
    csum = np.cumsum(loss)
    return int((csum - np.maximum.accumulate(np.where(loss, 0, csum)))[loss].max())


def tail_removal(R: np.ndarray, *, k: int = 5) -> dict:
    """꼬리제거 민감도 — 상위 k 승자를 빼면 기대값·누적이 얼마나 무너지나.

    반환: {k, n, expectancy_full, expectancy_ex(상위k 제거 후), cum_full, cum_ex}.
    expectancy_ex 가 음수로 뒤집히면 엣지는 그 몇 건에 전적으로 의존.
    """
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    n = int(len(R))
    idx = np.argsort(R)[::-1][:k]
    R2 = np.delete(R, idx)
    return {
        "k": int(k),
        "n": n,
        "expectancy_full": float(R.mean()) if n else NAN,
        "expectancy_ex": float(R2.mean()) if len(R2) else NAN,
        "cum_full": float(R.sum()),
        "cum_ex": float(R2.sum()),
    }


def median_trade(R: np.ndarray) -> float:
    """중앙값 트레이드(R) — 트레이더가 '보통' 겪는 것. 평균이 꼬리에 끌려간 것과 대비."""
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    return float(np.median(R)) if len(R) else NAN


def win_conditional(R: np.ndarray) -> dict:
    """승리조건부 분포 — 이길 때 얼마나 이기고, 질 때 얼마나 지나(대개 손절 −1R).

    반환: {win_rate, median_win, mean_win, max_win, median_loss}.
    손실은 R<=0 로 잡는다(무승부·손절 포함) — trader_view 원본과 동일.
    """
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    wins, losses = R[R > 0], R[R <= 0]
    return {
        "win_rate": float((R > 0).mean()) if len(R) else NAN,
        "median_win": float(np.median(wins)) if len(wins) else NAN,
        "mean_win": float(wins.mean()) if len(wins) else NAN,
        "max_win": float(R.max()) if len(R) else NAN,
        "median_loss": float(np.median(losses)) if len(losses) else NAN,
    }


def fragility_report(R: np.ndarray, *, entry: np.ndarray | None = None,
                     monster_k: int = 5, tail_k: int = 5) -> dict:
    """취약성 지표를 한 dict 로 묶는 편의 함수 — 괴물 의존도·연패·꼬리제거·중앙값·승리조건부."""
    R = np.asarray(R, float)
    n = int(np.isfinite(R).sum())
    return {
        "n": n,
        "monster_share": monster_share(R, k=monster_k),
        "max_loss_streak": max_loss_streak(R, entry=entry),
        "tail_removal": tail_removal(R, k=tail_k),
        "median_trade": median_trade(R),
        "win_conditional": win_conditional(R),
    }
