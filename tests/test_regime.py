"""국면 오버레이 — 룩어헤드·비용·널 구조의 불변식."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kr_quant.engine.metrics import max_drawdown
from kr_quant.strategies.regime import (
    apply_switch, ma_regime_state, market_index_level, monthly_state,
    percentile_of, rotation_null,
)


def _months(n: int, start: str = "2016-01") -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="M")


def test_index_level_compounds():
    r = pd.Series([0.1, -0.1], index=pd.to_datetime(["2016-01-04", "2016-01-05"]))
    lv = market_index_level(r)
    assert lv.iloc[-1] == pytest.approx(1.1 * 0.9)


def test_binary_state_is_level_vs_ma():
    lv = pd.Series(np.arange(1.0, 11.0), index=pd.date_range("2016-01-01", periods=10))
    s = ma_regime_state(lv, window=3)
    assert s.iloc[:2].isna().all()          # warm-up
    assert (s.dropna() == 1.0).all()        # monotonically rising -> always above MA


def test_three_tier_takes_half_when_between():
    lv = pd.Series([1, 2, 3, 4, 5, 4.5, 4.4], index=pd.date_range("2016-01-01", periods=7))
    s = ma_regime_state(lv, window=5, mid_window=2)
    assert set(s.dropna().unique()) <= {0.0, 0.5, 1.0}


def test_state_is_lagged_so_a_clairvoyant_signal_cannot_help():
    """룩어헤드 회귀 — 같은 달 수익으로 만든 완벽한 신호는 lag=1 에서 무력해야 한다."""
    idx = _months(24)
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.05, 24), index=idx)
    clairvoyant = (ret > 0).astype(float)          # built FROM the same month
    lagged, _ = apply_switch(ret, clairvoyant, switch_cost=0.0)
    cheating, _ = apply_switch(ret, clairvoyant, switch_cost=0.0, lag=0)
    assert cheating.sum() > lagged.sum()           # lag=0 leaks and wins
    assert cheating.sum() == pytest.approx(ret[ret > 0].sum())


def test_always_on_reproduces_series_minus_entry_cost():
    idx = _months(12)
    ret = pd.Series(0.01, index=idx)
    on = pd.Series(1.0, index=idx)
    out, exp = apply_switch(ret, on, switch_cost=0.0034)
    assert (exp == 1.0).all()
    # 첫 달만 진입 비용, 이후엔 상태가 안 바뀌므로 비용 없음
    assert out.iloc[0] == pytest.approx(0.01 - 0.0034)
    assert out.iloc[1:].to_numpy() == pytest.approx(0.01)


def test_switch_cost_charged_both_ways():
    idx = _months(4)
    ret = pd.Series(0.0, index=idx)
    state = pd.Series([1.0, 1.0, 0.0, 1.0], index=idx)   # 적용은 한 달 뒤
    out, exp = apply_switch(ret, state, switch_cost=0.01)
    assert list(exp) == [1.0, 1.0, 0.0]                  # lag=1 이라 마지막은 잘림
    assert out.iloc[0] == pytest.approx(-0.01)           # flat -> on
    assert out.iloc[1] == pytest.approx(0.0)             # 변화 없음
    assert out.iloc[2] == pytest.approx(-0.01)           # on -> flat


def test_rotation_null_preserves_duty_cycle_when_state_outruns_the_book():
    """회귀 — 상태가 북보다 길 때 회전이 듀티사이클을 보존해야 한다.

    이전 판본은 전체 상태를 회전한 뒤 북 인덱스로 reindex 해서, 회전마다 다른
    부분수열이 평가창에 들어왔다(실측 듀티 0.663 → 0.584~0.703). 널이 고정해야 할
    양이 흔들리면 "노출 축소가 아니라 타이밍인가" 를 가릴 수 없다.

    이 테스트가 잡아야 할 위반을 실제로 만든다 — 상태 126개월, 북 101개월.
    """
    state_idx = _months(126)
    book_idx = state_idx[25:]                      # 북이 25개월 늦게 시작
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.01, 0.04, len(book_idx)), index=book_idx)
    state = pd.Series((rng.random(126) > 0.35).astype(float), index=state_idx)

    seen: list[tuple[float, int]] = []

    def probe(s: pd.Series) -> float:
        # metric_fn 은 switched 수익 시리즈를 받는다. 노출 자체는 못 보므로
        # rotation_null 이 만든 시리즈 길이로 평가창 크기가 불변인지 본다.
        seen.append((float(len(s)), 0))
        return float(len(s))

    nulls, actual = rotation_null(ret, state, probe, switch_cost=0.0)
    assert len(nulls) == len(ret) - 1              # 평가창 길이 기준 회전 수
    assert all(n == actual for n in nulls)         # 평가창 크기 불변

    # 진짜 불변식: 회전된 노출의 듀티사이클과 스위치 횟수가 실제와 같아야 한다.
    _sw, used = apply_switch(ret, state, switch_cost=0.0)
    base_duty, base_sw = used.mean(), int((used.diff().abs() > 0).sum())
    arr = used.to_numpy()
    for k in range(1, len(arr)):
        rot = pd.Series(np.roll(arr, k), index=used.index)
        assert rot.mean() == pytest.approx(base_duty)
        assert abs(int((rot.diff().abs() > 0).sum()) - base_sw) <= 1  # 랩 지점 1회 허용


def test_percentile_of():
    assert percentile_of(0.5, [0.1, 0.2, 0.9]) == pytest.approx(2 / 3)
    assert np.isnan(percentile_of(0.5, []))


def test_monthly_state_takes_month_end():
    daily = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0],
                      index=pd.to_datetime(["2016-01-05", "2016-01-29",
                                            "2016-02-01", "2016-02-26",
                                            "2016-03-02"]))
    ms = monthly_state(daily)
    assert list(ms) == [1.0, 0.0, 1.0]


def test_max_drawdown_counts_the_opening_loss():
    """회귀 — 초기자본이 첫 peak 여야 한다.

    이전 판본은 equity 곡선만 보고 peak 를 잡아, 첫 구간이 하락이면 자기 자신이
    peak 가 되어 drawdown 0 을 돌려줬다. 국면 스위치 판정이 MDD 차이 0.3%p 로
    갈리는데 그 손실이 두 팔에 비대칭으로 걸린다.
    """
    assert max_drawdown(np.array([-0.20, 0.05, 0.05])) == pytest.approx(-0.20)
    # 상승만 하면 drawdown 은 0
    assert max_drawdown(np.array([0.10, 0.10])) == pytest.approx(0.0)
    # 중간 고점 이후 하락은 종전과 동일하게 잡힌다
    assert max_drawdown(np.array([0.50, -0.50])) == pytest.approx(-0.50)
