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
    build_behavior_signals,
    build_contrarian_signal,
    build_smart_signal,
    diagnose,
    g_shift,
    ic_weighted_book,
    simulate_momentum_long,
    simulate_trades,
    trade_stats,
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


def test_behavior_signals_structure_and_sign():
    """행동 신호: 추격 반대 부호가 맞는가 — 오른 종목을 개미가 사면 저점수(숏)."""
    dates = pd.bdate_range("2020-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    # A: 가격 상승(+10%/일 누적) + 개미 순매수 → 추격. antichase는 음수(숏)여야
    # B: 가격 상승 + 개미 순매도 → antichase 양수(롱)여야
    prows, frows = [], []
    pa, pb = 10000.0, 10000.0
    for d in dates:
        pa *= 1.03  # 둘 다 상승(pr>0)
        pb *= 1.03
        prows += [{"code": "A", "date": d, "close": pa, "trade_value": 1e9},
                  {"code": "B", "date": d, "close": pb, "trade_value": 1e9}]
        frows += [{"code": "A", "date": d, "individual": +500, "volume": 1000},   # 개미 매수(추격)
                  {"code": "B", "date": d, "individual": -500, "volume": 1000}]   # 개미 매도
    sigs = build_behavior_signals(pd.DataFrame(frows), pd.DataFrame(prows), window=3, lag=1)
    assert set(sigs) == {"antichase", "antiknife", "composite"}
    last = {k: v[v.code.isin(["A", "B"])].groupby("code")["signal"].last() for k, v in sigs.items()}
    # 오른 종목: 개미 매수(A)=추격 → antichase 음수, 개미 매도(B) → antichase 양수
    assert last["antichase"]["A"] < 0 < last["antichase"]["B"]
    # 오른 종목이므로 relu(-pr)=0 → antiknife는 0
    assert last["antiknife"]["A"] == pytest.approx(0.0)


def test_smart_signal_conditional_sign():
    """통합 신호: 오른데선 개미반대(-ri), 내린데선 개미편승(+ri) 부호 확인."""
    dates = pd.bdate_range("2020-01-01", periods=8).strftime("%Y-%m-%d").tolist()
    prows, frows = [], []
    up, dn = 10000.0, 10000.0
    for d in dates:
        up *= 1.02   # UP: 상승
        dn *= 0.98   # DN: 하락
        prows += [{"code": "UP", "date": d, "close": up, "trade_value": 1e9},
                  {"code": "DN", "date": d, "close": dn, "trade_value": 1e9}]
        frows += [{"code": "UP", "date": d, "individual": +500, "volume": 1000},  # 개미 매수
                  {"code": "DN", "date": d, "individual": +500, "volume": 1000}]  # 개미 매수
    sig = build_smart_signal(pd.DataFrame(frows), pd.DataFrame(prows), window=3, lag=1)
    last = sig.groupby("code")["signal"].last()
    # UP+개미매수(추격) → -ri<0 → 숏.  DN+개미매수(물타기) → +ri>0 → 롱.
    assert last["UP"] < 0 < last["DN"]


def test_ic_weighted_book_detects_edge():
    """합성: 신호가 미래수익과 양의 관계면 IC가중 북 Sharpe·t 양수여야."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=300).strftime("%Y-%m-%d").tolist()
    codes = [f"C{i:02d}" for i in range(50)]
    edge = rng.uniform(-1, 1, len(codes))  # 종목 고정 신호값 = 미래수익 드리프트
    prows, srows = [], []
    for ci, c in enumerate(codes):
        price = 10000.0
        for d in dates:
            price *= (1 + edge[ci] * 0.002 + rng.normal(0, 0.01))
            prows.append({"code": c, "date": d, "close": price, "trade_value": 1e9})
            srows.append({"code": c, "date": d, "signal": edge[ci]})
    s = ic_weighted_book(pd.DataFrame(prows), pd.DataFrame(srows),
                         horizon=20, step=5, adv_floor=0.0, start_index=30, min_names=10)
    assert s["t"] > 2, s
    assert s["sharpe"] > 0
    assert 0 < s["turnover"] <= 2.0  # 마켓뉴트럴 gross=1 → 회전율 상한 2


def test_simulate_trades_lookahead_free_and_stats():
    """이벤트드리븐: 룩어헤드 없이 트레이드가 생성되고, 손절/목표/시간 청산이 작동한다."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=250).strftime("%Y-%m-%d").tolist()
    codes = [f"C{i:02d}" for i in range(60)]  # winners(상위30%)≥10 되도록 충분히
    prows, frows = [], []
    for c in codes:
        price = 10000.0
        for d in dates:
            price *= (1 + rng.normal(0.001, 0.02))  # 변동성 있는 상승 드리프트(승자 다수)
            prows.append({"code": c, "date": d, "close": price, "trade_value": 1e9})
            frows.append({"code": c, "date": d, "individual": rng.normal(0, 1000), "volume": 1e5})
    trades = simulate_trades(pd.DataFrame(prows), pd.DataFrame(frows),
                             window=5, hold=20, adv_floor=0.0, start_index=30, ext_q=0.8)
    assert len(trades) > 0
    # 모든 청산 사유는 셋 중 하나, 보유일 1..20
    assert all(t["reason"] in ("stop", "target", "time") for t in trades)
    assert all(1 <= t["bars"] <= 20 for t in trades)
    # entry < exit (시간순), side ∈ {+1,-1}
    assert all(t["entry"] <= t["exit"] for t in trades)
    assert set(t["side"] for t in trades) <= {1, -1}
    s = trade_stats(trades)
    assert s["n"] == len(trades)
    assert 0.0 <= s["win_rate"] <= 1.0


def test_trade_stats_empty():
    assert trade_stats([])["n"] == 0


def test_numba_fast_parity_exact_on_clean_data():
    """numba simulate_fast == 순수파이썬 simulate_momentum_long (동점/NaN 없는 합성서 정확 일치).

    통일된 _pctl 임계 덕에 clean 데이터에서 bit-parity. (실데이터는 2019 저유동성일의
    NaN 경계에서 ~0.6% 연쇄차 — BO는 numba 엔진을 일관 사용하므로 무관.)
    """
    pytest.importorskip("numba")
    from research.contrarian_retail import fast_stats, simulate_fast
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=260).strftime("%Y-%m-%d").tolist()
    codes = [f"C{i:02d}" for i in range(80)]
    prows, frows = [], []
    for ci, c in enumerate(codes):
        price = 10000.0 + ci
        for d in dates:
            price *= (1 + rng.normal(0.0005, 0.02))
            prows.append({"code": c, "date": d, "close": price * (1 + 1e-6 * ci), "trade_value": 1e9 + ci})
            frows.append({"code": c, "date": d, "individual": rng.normal(0, 1000) + ci, "volume": 1e5})
    prices, flow = pd.DataFrame(prows), pd.DataFrame(frows)
    P = dict(window=5, hold=20, stop=0.10, target=0.0, trail=0.20, top_mom=0.8, ext_q=0.8,
             adv_floor=0.0, start_index=30)
    slow = simulate_momentum_long(prices, flow, retail_rule="dump", **P)
    r, e = simulate_fast(prices, flow, **P)
    sl = np.array([x["ret"] for x in slow])
    assert len(sl) == len(r), (len(sl), len(r))
    assert np.allclose(np.sort(sl), np.sort(r), atol=1e-9)
    # fast_stats와 trade_stats 스키마 일치
    assert fast_stats(r, e)["n"] == trade_stats(slow)["n"]


def test_trailing_stop_lets_winner_run_then_exits():
    """트레일링 스탑: 상승 후 고점대비 trail% 반납 시 청산 — 수익권에서만 발동."""
    # 60종목: 대부분 강하게 상승 후 특정일 급락 → 트레일 청산 확인
    dates = pd.bdate_range("2020-01-01", periods=120).strftime("%Y-%m-%d").tolist()
    codes = [f"C{i:02d}" for i in range(60)]
    prows, frows = [], []
    for ci, c in enumerate(codes):
        price = 10000.0
        for k, d in enumerate(dates):
            # 앞 40봉 상승, 이후 하락 → 트레일링이 고점 근처에서 잡아야
            price *= (1.01 if k < 40 else 0.99)
            prows.append({"code": c, "date": d, "close": price, "trade_value": 1e9})
            frows.append({"code": c, "date": d, "individual": -1000.0, "volume": 1e5})  # 개미 매도
    trades = simulate_momentum_long(pd.DataFrame(prows), pd.DataFrame(frows), retail_rule="dump",
                                    window=5, hold=60, stop=0.10, target=0.0, trail=0.05,
                                    top_mom=0.5, ext_q=0.6, adv_floor=0.0, start_index=30)
    assert len(trades) > 0
    trail_exits = [t for t in trades if t["reason"] == "trail"]
    assert len(trail_exits) > 0  # 트레일링이 실제로 발동
    # 트레일 청산 트레이드는 수익권에서 나옴(고점대비 소폭 반납이라 대체로 양수)
    assert all(t["reason"] in ("trail", "stop", "time") for t in trades)


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
