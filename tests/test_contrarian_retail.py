"""개미 반대매매 신호·진단 로직 단위테스트 (합성 데이터, DB 불요).

검증 포인트: (1) 부호 반전(정반대), (2) 룩어헤드 방지 래그, (3) 거래량 정규화,
(4) diagnose가 알려진 부호 관계를 올바르게 잡아내는가(랭크-IC·롱숏 부호).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.contrarian_retail import (
    _mdd,
    _spearman,
    build_contrarian_signal,
    diagnose,
    g_shift,
)


def _flow(codes, dates, individual, volume):
    rows = []
    for c in codes:
        for i, d in enumerate(dates):
            rows.append({"code": c, "date": d, "individual": individual[c][i], "volume": volume[c][i]})
    return pd.DataFrame(rows)


def test_sign_inversion():
    """sign=-1이 순매수 강도를 정확히 반전한다 (정반대)."""
    dates = [f"2020-01-{i:02d}" for i in range(1, 6)]
    flow = _flow(["A"], dates, {"A": [100, 100, 100, 100, 100]}, {"A": [1000] * 5})
    contr = build_contrarian_signal(flow, window=1, lag=0, sign=-1)
    follow = build_contrarian_signal(flow, window=1, lag=0, sign=+1)
    # 개인 순매수 양수(+0.1 강도) → 반대매매는 음, 추종은 양, 정확히 부호 반대·크기 동일
    cs = contr.set_index("date")["signal"]
    fs = follow.set_index("date")["signal"]
    assert np.allclose(cs.to_numpy(), -fs.to_numpy())
    assert (cs < 0).all()  # 개미가 산 종목 → 반대매매 저점수


def test_lookahead_lag():
    """lag=1이면 date t 신호는 t-1까지만 반영 — 미래 유입 없음."""
    dates = [f"2020-01-{i:02d}" for i in range(1, 6)]
    # 마지막 날에만 대량 순매수: lag=1이면 그 정보가 마지막 날 신호엔 안 들어가야
    flow = _flow(["A"], dates, {"A": [0, 0, 0, 0, 9999]}, {"A": [1000] * 5})
    lagged = build_contrarian_signal(flow, window=1, lag=1, sign=-1).set_index("date")["signal"]
    # t=마지막날 신호는 t-1(순매수 0)을 반영 → 0. 9999는 다음날로 새지 않음(다음날 없음)
    assert lagged.iloc[-1] == 0.0
    nolag = build_contrarian_signal(flow, window=1, lag=0, sign=-1).set_index("date")["signal"]
    assert nolag.iloc[-1] != 0.0  # lag=0이면 당일 대량매수가 즉시 반영


def test_volume_normalization():
    """같은 순매수라도 거래량이 크면 강도가 작다 (거래량 정규화 확인)."""
    dates = ["2020-01-01", "2020-01-02"]
    flow = _flow(["A", "B"], dates, {"A": [100, 100], "B": [100, 100]},
                 {"A": [1000, 1000], "B": [10000, 10000]})
    sig = build_contrarian_signal(flow, window=1, lag=0, sign=-1)
    a = sig[sig.code == "A"]["signal"].iloc[0]
    b = sig[sig.code == "B"]["signal"].iloc[0]
    assert abs(a) > abs(b)  # A는 거래량 작아 강도 큼


def test_g_shift_per_code():
    """g_shift는 종목별로 독립 shift — 종목 경계를 넘지 않는다."""
    s = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0])
    codes = pd.Series(["A", "A", "A", "B", "B"])
    shifted = g_shift(s, codes, 1)
    assert pd.isna(shifted.iloc[0])   # A 첫 행
    assert shifted.iloc[1] == 1.0
    assert pd.isna(shifted.iloc[3])   # B 첫 행 — A 값이 새지 않음
    assert shifted.iloc[4] == 10.0


def test_mdd():
    assert _mdd(np.array([])) != _mdd(np.array([]))  # NaN
    # +10% 후 -50% → 낙폭 -50%
    assert _mdd(np.array([0.1, -0.5])) == pytest.approx(-0.5)
    assert _mdd(np.array([0.1, 0.1, 0.1])) == pytest.approx(0.0)  # 단조증가 무낙폭


def test_spearman_known():
    a = np.array([1.0, 2, 3, 4, 5])
    assert _spearman(a, a) == pytest.approx(1.0)
    assert _spearman(a, a[::-1]) == pytest.approx(-1.0)


def test_diagnose_detects_true_sign():
    """합성: 개미 순매수 종목이 미래 저조하도록 설계 → 반대매매 IC·롱숏 양수여야."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=200).strftime("%Y-%m-%d").tolist()
    codes = [f"C{i:02d}" for i in range(40)]
    # 각 종목에 고정된 '개미선호도' — 높을수록 개미가 순매수, 그리고 미래수익은 낮게(반대관계)
    pref = rng.uniform(-1, 1, len(codes))
    prows, frows = [], []
    for ci, c in enumerate(codes):
        price = 10000.0
        for t, d in enumerate(dates):
            # 미래수익 드리프트: 개미선호 높을수록 하락(개미가 사면 떨어짐)
            drift = -pref[ci] * 0.002 + rng.normal(0, 0.01)
            price *= (1 + drift)
            prows.append({"code": c, "date": d, "close": price, "trade_value": 1e9})
            frows.append({"code": c, "date": d, "individual": pref[ci] * 1000 + rng.normal(0, 10),
                          "volume": 100000})
    prices = pd.DataFrame(prows)
    flow = pd.DataFrame(frows)
    contr = build_contrarian_signal(flow, window=5, lag=1, sign=-1)
    out = diagnose(prices, contr, horizon=20, step=5, adv_floor=0.0, start_index=30, min_names=10, top_n=10)
    # 개미가 산 종목이 떨어지도록 만들었으니 반대매매(=-개미) IC와 롱숏은 양수
    assert out["rank_ic"]["mean"] > 0, out["rank_ic"]
    assert out["long_short"]["mean"] > 0, out["long_short"]


def test_diagnose_null_when_no_relationship():
    """개미 순매수와 미래수익이 무관하면 롱숏 스프레드가 유의하지 않다(|t| 작음)."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=200).strftime("%Y-%m-%d").tolist()
    codes = [f"C{i:02d}" for i in range(40)]
    prows, frows = [], []
    for c in codes:
        price = 10000.0
        for d in dates:
            price *= (1 + rng.normal(0, 0.01))  # 순수 랜덤워크, 개미와 무관
            prows.append({"code": c, "date": d, "close": price, "trade_value": 1e9})
            frows.append({"code": c, "date": d, "individual": rng.normal(0, 1000), "volume": 100000})
    contr = build_contrarian_signal(pd.DataFrame(frows), window=5, lag=1, sign=-1)
    out = diagnose(pd.DataFrame(prows), contr, horizon=20, step=5, adv_floor=0.0,
                   start_index=30, min_names=10, top_n=10)
    assert abs(out["long_short"]["t"]) < 3.0  # 무관계 → 유의하지 않음
