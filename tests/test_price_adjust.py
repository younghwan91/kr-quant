"""price_adjust: 기업행동 백조정 유틸 테스트."""

import pandas as pd

from kr_quant.price_adjust import adjust_prices, diagnose


def _series(closes, code="A"):
    n = len(closes)
    return pd.DataFrame({
        "code": [code] * n,
        "date": [f"2021-01-{i + 1:02d}" for i in range(n)],
        "open": closes, "high": closes, "low": closes, "close": closes,
    })


def test_4to1_split_becomes_continuous():
    # 100까지 상승 후 4:1 분할(→25)로 이후 25~30 유지
    closes = [80, 90, 100, 25, 26, 27, 28, 29]
    df = _series(closes)
    adj = adjust_prices(df)["close"].tolist()
    # 분할 이전 구간은 1/4로 백조정되어 연속(100→25)이어야
    assert abs(adj[2] / adj[3] - 1) < 0.2, adj
    assert abs(adj[0] - 20) < 1e-6 and abs(adj[2] - 25) < 1e-6, adj
    # 분할 이후 구간은 그대로
    assert adj[-1] == 29


def test_transient_spike_not_adjusted():
    # 하루만 튀고 되돌아오는 데이터 스파이크는 분할로 보지 않는다
    closes = [100, 101, 300, 102, 103, 104, 105, 106]
    df = _series(closes)
    adj = adjust_prices(df)["close"].tolist()
    assert adj == [float(x) for x in closes], adj


def test_normal_moves_untouched():
    closes = [100, 105, 110, 108, 112, 115, 113, 117]
    df = _series(closes)
    adj = adjust_prices(df)["close"].tolist()
    assert adj == [float(x) for x in closes]


def test_diagnose_flags_split_date():
    df = _series([80, 90, 100, 25, 26, 27, 28, 29])
    ev = diagnose(df)
    assert len(ev) == 1
    assert ev.iloc[0]["date"] == "2021-01-04"  # 100→25 지점
    assert abs(ev.iloc[0]["ratio"] - 0.25) < 0.01


def test_ohlc_scaled_together():
    n = 8
    df = pd.DataFrame({
        "code": ["A"] * n,
        "date": [f"2021-01-{i + 1:02d}" for i in range(n)],
        "open": [80, 90, 100, 25, 26, 27, 28, 29],
        "high": [82, 92, 102, 26, 27, 28, 29, 30],
        "low": [78, 88, 98, 24, 25, 26, 27, 28],
        "close": [80, 90, 100, 25, 26, 27, 28, 29],
    })
    adj = adjust_prices(df)
    # 분할 이전 high/low 도 동일 배수(0.25)로 조정
    assert abs(adj.iloc[2]["high"] - 25.5) < 1e-6
    assert abs(adj.iloc[2]["low"] - 24.5) < 1e-6
