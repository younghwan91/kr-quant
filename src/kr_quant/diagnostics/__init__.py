"""kr_quant.diagnostics — 백테스트 사후 진단: R-멀티플 분포·취약성·배포 준비도 신호.

라이브러리 경계(leaf 패키지): numpy 만 import 하고, research/ 로부터는 아무것도
import 하지 않는다. 렌즈는 오직 **개별 트레이드 R-멀티플 분포** — 평균-헤드라인·
복리 자본곡선·포트폴리오 프레이밍(슬롯·동시보유·연환산-슬롯당)은 담지 않는다.

- r_distribution: 분포 모양(왼꼬리 절단·오른꼬리 두께), 선별 곡선, 확신 분위 분석, 보유상한 스윕.
- fragility: 괴물 의존도, 최장 연패, 꼬리제거 민감도, 중앙값·승리조건부 분포.
- gate_report: 배포 준비도 신호 REPORTER (PASS/FAIL·임계값 없음).
"""

from __future__ import annotations

from .fragility import (
    fragility_report,
    max_loss_streak,
    median_trade,
    monster_share,
    tail_removal,
    win_conditional,
)
from .gate_report import gate_report
from .r_distribution import (
    conviction_analysis,
    dist_shape,
    hold_curve,
    r_multiples,
    selection_curve,
)

__all__ = [
    # r_distribution
    "r_multiples",
    "dist_shape",
    "selection_curve",
    "conviction_analysis",
    "hold_curve",
    # fragility
    "monster_share",
    "max_loss_streak",
    "tail_removal",
    "median_trade",
    "win_conditional",
    "fragility_report",
    # gate_report
    "gate_report",
]
