"""``delisting_exit`` — 보유 중 상장폐지된 종목의 손실이 실제로 반영되는지.

기본 동작은 ``ret = C[:, t+step]/C[:, t] - 1`` 이라 폐지 후 가격이 NaN 이 되고
``np.nanmean`` 이 그 종목을 조용히 뺀다. 즉 폐지 종목이 **비용 없이 사라진다** —
터지지 않고 수익률만 낙관 쪽으로 틀리는 종류라 합성 패널로 못박는다.
"""

from __future__ import annotations

import numpy as np
import pytest

from kr_quant.engine.sim_crosssectional import staggered_tranche_backtest

STEP = 2
KW = dict(horizon=2, step=STEP, top_n=1, adv_floor=0.0, adv_window=1,
          start_index=1, min_names=1)


def _panels(n_dates=8, n_codes=2):
    dates = [f"2020-01-{d + 1:02d}" for d in range(n_dates)]
    C = np.full((n_codes, n_dates), 100.0)
    V = np.full((n_codes, n_dates), 1e9)
    sig = np.full((n_codes, n_dates), np.nan)
    return C, V, sig, dates


def _run(C, V, sig, dates, **kw):
    periods, _ = staggered_tranche_backtest(C, V, sig, dates, **{**KW, **kw})
    return periods


def test_doomed_name_vanishes_without_the_flag():
    """폐지 종목이 반토막 뒤 사라져도 기본 동작에서는 손실이 안 잡힌다.

    북에 건강한 종목을 하나 더 둬서(top_n=2) 북 평균이 비지 않게 한다 — 비면
    ``nanmean`` 이 NaN 을 내고 그 기간이 통째로 집계에서 빠져, 손실 누락과 구분이
    안 된다.
    """
    C, V, sig, dates = _panels(n_codes=3)
    sig[0, :], sig[1, :], sig[2, :] = 2.0, 1.0, 0.0   # 북 = code0 + code1
    C[0, 3] = 50.0           # 마지막 거래일에 반토막
    C[0, 4:] = np.nan        # 이후 상장폐지

    off = _run(C, V, sig, dates, top_n=2, min_names=2, start_index=2)
    on = _run(C, V, sig, dates, top_n=2, min_names=2, start_index=2,
              delisting_exit=True)
    assert not off.empty and not on.empty
    assert on["net"].sum() < off["net"].sum(), (
        "폐지 손실을 반영하면 성과가 더 나빠야 한다 — 같거나 좋으면 손실이 새고 있다"
    )


def test_whole_book_vanishing_makes_the_period_disappear_without_the_flag():
    """북이 통째로 폐지되면 기본 동작은 그 기간을 NaN 으로 만들어 집계에서 뺀다.

    손실이 0으로 계상되는 것보다 나쁘다 — 아예 없었던 일이 된다.
    """
    C, V, sig, dates = _panels()
    sig[0, :], sig[1, :] = 1.0, 0.0
    C[0, 3] = 50.0
    C[0, 4:] = np.nan

    off = _run(C, V, sig, dates)
    on = _run(C, V, sig, dates, delisting_exit=True)
    assert off["net"].isna().any(), "기본 동작에서는 NaN 기간이 생겨야 한다"
    assert not on["net"].isna().any(), "플래그를 켜면 모든 기간이 실수여야 한다"


def test_exit_uses_last_observed_price():
    """청산가는 마지막 관측 종가(정리매매 종료가)여야 한다."""
    C, V, sig, dates = _panels()
    sig[0, :] = 1.0
    sig[1, :] = 0.0
    C[0, 3] = 25.0           # t=2 진입(100) -> t=4 청산 시점엔 NaN, 마지막 관측=25
    C[0, 4:] = np.nan

    on = _run(C, V, sig, dates, delisting_exit=True, start_index=2)
    # 첫 기간의 북은 code0 단독, 벤치는 code0+code1 평균.
    # code0 수익 = 25/100 - 1 = -0.75 가 반영돼야 하므로 북-벤치 < 0.
    assert on.iloc[0]["net"] < 0


def test_flag_is_a_noop_when_nothing_is_delisted():
    """폐지가 없으면 켜든 끄든 동일해야 한다 — 기존 수치에 영향 없음의 근거."""
    C, V, sig, dates = _panels(n_codes=3)
    rng = np.random.default_rng(0)
    C[:] = 100 * np.cumprod(1 + rng.normal(0, 0.01, C.shape), axis=1)
    sig[:] = rng.normal(0, 1, sig.shape)

    off = _run(C, V, sig, dates, top_n=2, min_names=2)
    on = _run(C, V, sig, dates, top_n=2, min_names=2, delisting_exit=True)
    assert off["net"].to_list() == pytest.approx(on["net"].to_list())


def test_midseries_gap_is_not_treated_as_delisting():
    """중간 거래정지(뒤에 가격이 다시 나옴)는 폐지가 아니다 — 정상 수익률로 이어져야."""
    C, V, sig, dates = _panels(n_codes=3)
    sig[:] = 0.0
    sig[0, :] = 1.0
    C[0, 3] = np.nan          # 하루 결측, 이후 재개
    C[0, 4] = 120.0

    on = _run(C, V, sig, dates, delisting_exit=True, start_index=2)
    # t=2(100) -> t=4(120) 이므로 code0 은 +20%, 손실 처리되면 안 된다.
    assert on.iloc[0]["net"] > 0
