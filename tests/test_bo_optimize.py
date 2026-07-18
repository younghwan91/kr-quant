"""BO 최적화 유닛테스트 (합성, DB 불요) — 목적함수 부호·train/test 격리·부트스트랩 하단.

핵심 과최적 방지 장치인 (1) train/test 분리 무누출, (2) 거래수 페널티, (3) 보수적 목적함수를 검증.
로직은 ``kr_quant.validation.optimization`` (라이브러리)에 있고, 신호 배선(simulate_fast·SPACE)은
research 실험에서 온다 — 테스트는 그 둘을 이어 목적함수 계약을 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kr_quant.validation.optimization import _boot_lower, make_objective, split_stats


def test_split_stats_train_test_isolation():
    """entry_date 경계로 train/test가 겹치지 않고 정확히 분리된다(정보 누출 방지)."""
    rets = np.array([0.1, -0.2, 0.3, -0.4])
    edates = np.array(["2019-05-01", "2019-08-01", "2023-05-01", "2023-08-01"])
    train = split_stats(rets, edates, hi="2022-01-01")
    test = split_stats(rets, edates, lo="2022-01-01")
    assert set(np.round(train, 3)) == {0.1, -0.2}
    assert set(np.round(test, 3)) == {0.3, -0.4}
    # 겹침 0
    assert len(train) + len(test) == len(rets)


def test_boot_lower_is_below_mean_and_guards_small():
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 0.1, 500)
    lo = _boot_lower(r, seed=0)
    assert lo < r.mean()          # 하단은 평균보다 작다(보수적)
    assert _boot_lower(np.array([0.1, 0.2])) == -1.0   # 소표본 가드


def _synth(year: str, n_codes=60, n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(f"{year}-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    prows, frows = [], []
    for ci in range(n_codes):
        price = 10000.0 + ci
        for d in dates:
            price *= (1 + rng.normal(0.001, 0.02))
            prows.append({"code": f"C{ci:02d}", "date": d, "close": price, "trade_value": 1e9})
            frows.append({"code": f"C{ci:02d}", "date": d, "individual": rng.normal(0, 1000), "volume": 1e5})
    return pd.DataFrame(prows), pd.DataFrame(frows)


def _make_sim(prices, flow):
    """params dict -> (수익, 진입일) 콜러블 — 라이브러리 make_objective 가 받는 신호 배선.

    contrarian_bo.make_sim 과 동일 규약(target=0, cost 기본). 여기선 테스트 자족을 위해 인라인."""
    from research.signals.contrarian_retail import simulate_fast

    def simulate(params: dict):
        return simulate_fast(prices, flow, target=0.0, **params)
    return simulate


def test_objective_counts_train_only_not_oos():
    """목적함수는 TRAIN 트레이드만 센다 — 진입이 전부 OOS(2023)면 페널티(<0) 반환."""
    pytest.importorskip("numba")
    optuna = pytest.importorskip("optuna")
    from research.experiments.contrarian_bo import SPACE
    prices, flow = _synth("2023")          # 모든 진입이 2023(=OOS, train<2022)
    obj = make_objective(_make_sim(prices, flow), SPACE, train_hi="2022-01-01", floor=5)
    ft = optuna.trial.FixedTrial(
        {"window": 5, "top_mom": 0.8, "ext_q": 0.8, "stop": 0.10, "trail": 0.20, "hold": 20})
    val = obj(ft)
    assert val < 0.0    # TRAIN 트레이드 0 → 페널티


def test_objective_positive_region_on_train():
    """진입이 TRAIN(2019)에 충분하면 목적함수가 페널티가 아닌 실제 값을 낸다."""
    pytest.importorskip("numba")
    optuna = pytest.importorskip("optuna")
    from research.experiments.contrarian_bo import SPACE
    prices, flow = _synth("2019")
    obj = make_objective(_make_sim(prices, flow), SPACE, train_hi="2022-01-01", floor=5)
    ft = optuna.trial.FixedTrial(
        {"window": 5, "top_mom": 0.8, "ext_q": 0.8, "stop": 0.10, "trail": 0.20, "hold": 20})
    val = obj(ft)
    assert val > -1.0   # 페널티 바닥(-1)이 아니라 실제 부트스트랩 하단
    assert ft.user_attrs.get("n_train", 0) >= 5
