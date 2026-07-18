"""배포 준비도 REPORTER — 판정기(verdict machine)가 아니라 신호 보고기.

이 모듈은 배포 준비도를 **숫자·분포**로 보고한다: OOS 기대값 R 의 부트스트랩
신뢰구간, 워크포워드 폴드 일관성 COUNT, 괴물 의존도, 최장 연패, (비용 스윕이
주어지면) 엣지가 죽는 비용. 그게 전부다.

이 모듈은 의도적으로 PASS/FAIL 을 내지 않으며 "monster<50%" 같은 임계값을
하드코딩하지 않는다. 확률적(stochastic) 데이터에 결정론적 판정을 박는 것 자체가
그 자체로 과최적 규칙(Principle 1: stochastic, not deterministic 위반)이기 때문이다.
이 신호들을 읽고 배포 여부를 판단하는 것은 연구자의 몫이다 — 이 함수의 몫이 아니다.

모든 입력은 순수 배열/매핑(R-멀티플, 진입일, 폴드별 기대값, 비용→기대값 곡선).
research/ 로부터 아무것도 import 하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import e
from statistics import NormalDist

import numpy as np

from .fragility import max_loss_streak, monster_share

NAN = float("nan")

_EULER_MASCHERONI = 0.5772156649015329   # γ — E[max of N Gaussians] 근사에 사용
_NORM = NormalDist()                     # 표준정규 (inv_cdf=ppf, cdf)


def _bootstrap_ci(R: np.ndarray, *, n_boot: int, seed: int, ci: float) -> tuple[float, float]:
    """기대값 R(=평균)의 부트스트랩 신뢰구간 (lo, hi). 표본 2개 미만이면 (NAN, NAN)."""
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    if len(R) < 2:
        return (NAN, NAN)
    rng = np.random.default_rng(seed)
    means = R[rng.integers(0, len(R), size=(n_boot, len(R)))].mean(axis=1)
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    return (float(np.percentile(means, lo_pct)), float(np.percentile(means, hi_pct)))


def _cost_edge_dies(cost_curve: Mapping | Sequence) -> float | None:
    """엣지가 죽는 비용 — 기대값 R 이 처음으로 ≤0 이 되는 최소 비용. 안 죽으면 None.

    ``cost_curve``: {비용: 기대값R} 매핑 또는 (비용, 기대값R) 시퀀스. 비용 오름차순으로 검사.
    """
    items = sorted(cost_curve.items()) if isinstance(cost_curve, Mapping) else sorted(cost_curve)
    for cost, exp in items:
        if exp <= 0:
            return float(cost)
    return None


# --- 다중검정 보정 (§4·5·11) — Deflated Sharpe + Harvey-Liu t-haircut ------------
#
# 시도한 config 수 N 을 안 세면 "N개 중 최고 Sharpe" 는 선택편향으로 부풀려진다.
# 아래는 그 부풀림을 **숫자로** 깎는다 (판정 아님 — 리포터 원칙 유지, PASS/FAIL 없음).
#
#   Deflated Sharpe Ratio — Bailey & López de Prado (2014), "The Deflated Sharpe
#     Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
#     Journal of Portfolio Management 40(5), 94–107.
#   t-haircut — Harvey & Liu (2014), "…and the Cross-Section of Expected Returns":
#     다중검정하에서 새 팩터는 t>2.0 이 아니라 t>3.0 수준을 요구한다.


def _expected_max_sharpe_h0(n_trials: int, sr_std: float) -> float:
    """N개 독립 시행 중 **최고 Sharpe 의 기대치** (H0: 참 Sharpe=0) = deflation 벤치마크 SR0.

    참 엣지가 전혀 없어도(모든 시행 SR=0) 표본잡음만으로 최고 시행의 Sharpe 는 0보다
    크다. 그 기대 최대값(Bailey & LdP):

        SR0 ≈ sr_std · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    ``sr_std`` 는 시행별 Sharpe 추정치의 표준편차, γ=Euler-Mascheroni. N≤1 이면 0.
    """
    if n_trials is None or n_trials <= 1 or not np.isfinite(sr_std):
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * e))
    return float(sr_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def _prob_sharpe(sharpe: float, benchmark: float, n_obs: int,
                 skew: float, kurtosis: float) -> float:
    """PSR — 참 Sharpe 가 ``benchmark`` 를 넘을 확률 (Bailey & LdP). ``kurtosis`` 는 비초과(정규=3).

        PSR = Φ( (SR − benchmark)·√(T−1) / √(1 − skew·SR + (kurt−1)/4·SR²) )

    비정규성(음의 왜도·두꺼운 꼬리)과 표본길이 T 를 반영해 Sharpe 유의성을 깎는다.
    표본부족·퇴화(분모≤0)면 NaN.
    """
    if n_obs < 2 or not np.isfinite(sharpe):
        return NAN
    denom = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    if denom <= 0:
        return NAN
    z = (sharpe - benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(_NORM.cdf(z))


def _t_haircut(n_trials: int, *, alpha: float = 0.05) -> dict:
    """Harvey-Liu 스타일 t-haircut — 시행 N 이 늘면 유의 t 문턱이 오른다(Bonferroni).

    단일검정 양측 α 의 문턱 t₀(≈1.96, α=0.05)는 N개 다중검정에서 α/N 로 조여져 문턱이
    상승한다. ``haircut_multiple`` = 조정문턱/기본문턱 → raw t 는 그만큼 약해 보인다.
    (참고: Harvey-Liu 는 새 팩터에 t>3.0 권고 — N≈20 이면 문턱이 대략 3.0.)
    """
    base = _NORM.inv_cdf(1.0 - alpha / 2.0)
    n = max(int(n_trials), 1)
    adj = _NORM.inv_cdf(1.0 - alpha / (2.0 * n))
    return {
        "alpha": float(alpha),
        "method": "Bonferroni (two-sided)",
        "base_hurdle_t": float(base),
        "adjusted_hurdle_t": float(adj),
        "haircut_multiple": float(adj / base),
    }


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
    *,
    sr_std: float | None = None,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014) — 선택편향 보정 리포트.

    관측 Sharpe 를 **N개 시행 중 최고를 뽑았다는 선택편향** + 비정규성 + 표본길이로 깎는다.

    반환 dict:
        n_trials, observed_sharpe, expected_max_sharpe_h0 (=SR0, deflation 벤치마크),
        deflated_sharpe (=SR − SR0 — SR0 아래면 "N개 뽑기의 운"으로 설명 가능),
        prob_sharpe_gt0 (PSR vs 0), prob_deflated_sharpe (PSR vs SR0 = 정통 DSR 확률),
        t_haircut (Harvey-Liu 조정문턱).

    ``sr_std`` (시행별 Sharpe 추정치 표준편차)를 안 주면 H0 하의 Sharpe 추정량
    표준오차 ≈ 1/√(T−1) 로 근사한다. ``sharpe`` 는 표본(건당) Sharpe = mean/std.
    ``kurtosis`` 는 비초과(정규=3). 판정 아님 — 숫자만.
    """
    n_obs = int(n_obs)
    if sr_std is None:
        sr_std = 1.0 / np.sqrt(n_obs - 1) if n_obs > 1 else NAN
    sr0 = _expected_max_sharpe_h0(n_trials, sr_std)
    return {
        "n_trials": int(n_trials),
        "observed_sharpe": float(sharpe),
        "expected_max_sharpe_h0": float(sr0),
        "deflated_sharpe": float(sharpe - sr0),
        "prob_sharpe_gt0": _prob_sharpe(sharpe, 0.0, n_obs, skew, kurtosis),
        "prob_deflated_sharpe": _prob_sharpe(sharpe, sr0, n_obs, skew, kurtosis),
        "t_haircut": _t_haircut(n_trials),
    }


def _deflation_block(R: np.ndarray, n_trials: int) -> dict:
    """OOS R-분포 → 건당 Sharpe·왜도·첨도 계산 후 ``deflated_sharpe(...)`` 로 리포트.

    Sharpe = mean / std(ddof=1). 왜도·첨도는 dist_shape 와 동일한 모집단 모멘트 관례
    (첨도는 비초과, 정규=3). 표본<2 이거나 std=0 이면 NaN 으로 전달(퇴화 방어).
    """
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    n = int(len(R))
    if n < 2 or R.std(ddof=1) == 0:
        sr = skew = kurt = NAN
    else:
        mu = R.mean()
        sr = float(mu / R.std(ddof=1))
        sd = R.std()  # 모집단 std — dist_shape 왜도 관례와 일치
        skew = float(((R - mu) ** 3).mean() / sd ** 3)
        kurt = float(((R - mu) ** 4).mean() / sd ** 4)
    return deflated_sharpe(sr, n_trials, n, skew, kurt)


def gate_report(
    oos_R: np.ndarray,
    *,
    n_trials: int | None = None,
    fold_expectancies: Sequence | None = None,
    entry: np.ndarray | None = None,
    monster_k: int = 5,
    cost_curve: Mapping | Sequence | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """배포 준비도 신호를 구조화된 dict 로 보고한다 — PASS/FAIL 없음, 임계값 없음.

    이 함수는 REPORTER 다. bool 판정 키를 담지 않으며 임계값을 하드코딩하지 않는다.
    반환된 신호를 해석해 배포를 결정하는 것은 연구자의 판단이다 (모듈 docstring 참조).

    Args:
        oos_R: OOS 개별 트레이드 R-멀티플 배열.
        n_trials: 시도한 config 수(옵션). 주어지면 deflation 블록(Deflated Sharpe +
            Harvey-Liu t-haircut)을 계산해 선택편향을 숫자로 깎는다. None 이면 블록 생략
            (기본 동작 불변).
        fold_expectancies: 워크포워드 폴드별 기대값 R 시퀀스(옵션). 주어지면
            fold_consistency 에 폴드 수·양(+)폴드 COUNT 를 보고(판정 아님).
        entry: 진입일 배열(옵션) — 최장 연패를 시간순으로 계산할 때 사용.
        monster_k: 괴물 의존도 상위 k.
        cost_curve: {비용: 기대값R} 매핑/시퀀스(옵션) — 엣지가 죽는 비용 보고.
        n_boot, seed, ci: 기대값 부트스트랩 신뢰구간 파라미터.

    Returns:
        dict: n, expectancy_R, expectancy_ci=(lo, hi), monster_share, max_loss_streak,
        (n_trials 주면) deflation={n_trials, observed_sharpe, expected_max_sharpe_h0,
        deflated_sharpe, prob_sharpe_gt0, prob_deflated_sharpe, t_haircut},
        (fold_expectancies 주면) fold_consistency={n_folds, n_positive, fold_expectancies},
        (cost_curve 주면) cost_edge_dies. bool 판정 키는 포함하지 않는다.
    """
    R = np.asarray(oos_R, float)
    R = R[np.isfinite(R)]
    n = int(len(R))
    report: dict = {
        "n": n,
        "expectancy_R": float(R.mean()) if n else NAN,
        "expectancy_ci": _bootstrap_ci(R, n_boot=n_boot, seed=seed, ci=ci),
        "monster_share": monster_share(R, k=monster_k),
        "max_loss_streak": max_loss_streak(oos_R, entry=entry),
    }
    if n_trials is not None and n_trials >= 1:
        report["deflation"] = _deflation_block(R, n_trials)
    if fold_expectancies is not None:
        fe = np.asarray(fold_expectancies, float)
        report["fold_consistency"] = {
            "n_folds": int(len(fe)),
            "n_positive": int((fe > 0).sum()),
            "fold_expectancies": [float(x) for x in fe],
        }
    if cost_curve is not None:
        report["cost_edge_dies"] = _cost_edge_dies(cost_curve)
    return report
