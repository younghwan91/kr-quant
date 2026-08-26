"""국면 오버레이 — 룩어헤드·비용·널 구조의 불변식."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_rotation_null_holds_duty_cycle_and_switch_count():
    idx = _months(36)
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.01, 0.04, 36), index=idx)
    state = pd.Series((rng.random(36) > 0.4).astype(float), index=idx)

    def duty(s: pd.Series) -> float:
        return float(len(s))

    nulls, actual = rotation_null(ret, state, duty, switch_cost=0.0)
    assert len(nulls) == 35                       # 비자명 회전 전부
    assert all(n == actual for n in nulls)        # 회전은 표본 수를 안 바꾼다
    # 회전은 켜진 달 수(듀티사이클)를 보존한다
    for k in range(1, 36):
        assert np.roll(state.to_numpy(), k).sum() == state.to_numpy().sum()


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
